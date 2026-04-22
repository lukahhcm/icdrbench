#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icdrbench.config import load_domains_config
from icdrbench.domain_assignment import build_domain_execution_plan
from scripts.prepare_data.materialize_domain_workflows import (
    FILTER_CALIBRATION_RULES,
    FILTER_STATUS_RULES,
    _apply_mapper_text,
    _evaluate_filter,
    _infer_suffix,
    _labeling_meta,
    _percentile,
    iter_jsonl,
)


def _log(message: str) -> None:
    print(message, flush=True)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    count = 0
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    tmp_path.replace(path)
    return count


def _operator_lookup(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    for variant in plan['execution_variants']:
        by_name.setdefault(variant['name'], variant)
    return by_name


def _op_kind(op_name: str, operators_by_name: dict[str, dict[str, Any]]) -> str:
    variant = operators_by_name.get(op_name)
    if variant is not None:
        return str(variant['kind'])
    if op_name.endswith('_filter'):
        return 'filter'
    return 'mapper'


def _base_params(op_name: str, operators_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    variant = operators_by_name.get(op_name)
    return dict(variant.get('params', {})) if variant is not None else {}


def _mapper_names(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> list[str]:
    return [op_name for op_name in sequence if _op_kind(op_name, operators_by_name) == 'mapper']


def _filter_names(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> list[str]:
    return [op_name for op_name in sequence if _op_kind(op_name, operators_by_name) == 'filter']


def _supporting_records(
    records: list[dict[str, Any]],
    mapper_names: list[str],
    max_records: int,
    salt: str,
) -> list[dict[str, Any]]:
    wanted = set(mapper_names)
    supported = []
    for record in records:
        active = set(_labeling_meta(record).get('active_mapper_names', []))
        if wanted.issubset(active):
            supported.append(record)
    supported.sort(key=lambda row: _stable_id(salt, row.get('id'), row.get('source_name'), row.get('url')))
    return supported[:max_records]


def _record_id(record: dict[str, Any]) -> str:
    for key in ('id', 'source_name', 'url'):
        value = record.get(key)
        if value:
            return str(value)
    return _stable_id(record.get('text', ''), length=20)


def _first_filter_index(sequence: list[str], operators_by_name: dict[str, dict[str, Any]]) -> int | None:
    for idx, op_name in enumerate(sequence):
        if _op_kind(op_name, operators_by_name) == 'filter':
            return idx
    return None


def _text_before_step(
    record: dict[str, Any],
    sequence: list[str],
    step_index: int,
    operators_by_name: dict[str, dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for idx, op_name in enumerate(sequence[:step_index]):
        kind = _op_kind(op_name, operators_by_name)
        if kind == 'filter':
            continue
        before_len = len(text)
        text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
        trace.append(
            {
                'step_index': idx,
                'operator': op_name,
                'kind': kind,
                'input_length': before_len,
                **result,
            }
        )
    return text, trace


def _filter_value(
    record: dict[str, Any],
    sequence: list[str],
    filter_index: int,
    filter_params: dict[str, Any],
    operators_by_name: dict[str, dict[str, Any]],
) -> tuple[float | None, str | None]:
    filter_name = sequence[filter_index]
    value_key = _resolve_filter_value_key(filter_name, filter_params)
    if value_key is None:
        return None, None
    text, _trace = _text_before_step(record, sequence, filter_index, operators_by_name)
    evaluation = _evaluate_filter(filter_name, text, filter_params, _infer_suffix(record))
    value = evaluation.get('stats', {}).get(value_key)
    if isinstance(value, (int, float)):
        return float(value), value_key
    return None, value_key


def _resolve_filter_value_key(op_name: str, params: dict[str, Any]) -> str | None:
    rule = FILTER_STATUS_RULES.get(op_name)
    if rule is None:
        return None
    value_key = rule['value_key']
    return value_key(params) if callable(value_key) else value_key


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _calibrate_filter_params_for_target(
    op_name: str,
    base_params: dict[str, Any],
    values: list[float],
    target_drop_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rule = FILTER_STATUS_RULES.get(op_name)
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    params = dict(base_params)
    if not rule or not calibration or not values:
        return params, {
            'threshold_source': 'not_calibrated',
            'target_drop_rate': target_drop_rate,
            'threshold_value': None,
        }

    if calibration['direction'] == 'min':
        q = target_drop_rate
        threshold = _percentile(values, q)
        if threshold is not None:
            params[rule['min_key']] = _round_float(threshold)
            params.pop(rule['max_key'], None)
    else:
        q = 1.0 - target_drop_rate
        threshold = _percentile(values, q)
        if threshold is not None:
            params[rule['max_key']] = _round_float(threshold)
            params.pop(rule['min_key'], None)

    return params, {
        'threshold_source': 'instance_balanced_quantile',
        'target_drop_rate': target_drop_rate,
        'threshold_quantile': q,
        'threshold_direction': calibration['direction'],
        'threshold_value': _round_float(threshold),
    }


def _execute_workflow(
    record: dict[str, Any],
    sequence: list[str],
    operators_by_name: dict[str, dict[str, Any]],
    filter_params_by_name: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    filter_params_by_name = filter_params_by_name or {}
    suffix = _infer_suffix(record)
    text = str(record.get('text', ''))
    trace = []
    for step_index, op_name in enumerate(sequence):
        kind = _op_kind(op_name, operators_by_name)
        before_len = len(text)
        if kind == 'mapper':
            text, result = _apply_mapper_text(op_name, text, _base_params(op_name, operators_by_name), suffix)
            trace.append(
                {
                    'step_index': step_index,
                    'operator': op_name,
                    'kind': kind,
                    'input_length': before_len,
                    **result,
                }
            )
            continue

        params = dict(filter_params_by_name.get(op_name, _base_params(op_name, operators_by_name)))
        evaluation = _evaluate_filter(op_name, text, params, suffix)
        trace.append(
            {
                'step_index': step_index,
                'operator': op_name,
                'kind': kind,
                'input_length': before_len,
                'keep': evaluation['keep'],
                'status': evaluation['status'],
                'stats': evaluation.get('stats', {}),
            }
        )
        if not evaluation['keep']:
            return {
                'reference_status': 'DROP',
                'reference_text': '',
                'intermediate_text_at_drop': text,
                'trace': trace,
            }

    return {
        'reference_status': 'KEEP',
        'reference_text': text,
        'intermediate_text_at_drop': None,
        'trace': trace,
    }


def _variant_record(
    base: dict[str, Any],
    record: dict[str, Any],
    sequence: list[str],
    filter_params_by_name: dict[str, dict[str, Any]],
    execution: dict[str, Any],
    threshold_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    instance_id = _stable_id(base.get('workflow_variant_id'), _record_id(record))
    return {
        'instance_id': instance_id,
        'source_record_id': _record_id(record),
        'input_text': str(record.get('text', '')),
        'operator_sequence': sequence,
        'filter_params_by_name': filter_params_by_name,
        'threshold_meta': threshold_meta or {},
        'reference_status': execution['reference_status'],
        'reference_text': execution['reference_text'],
        'intermediate_text_at_drop': execution.get('intermediate_text_at_drop'),
        'reference_trace': execution['trace'],
        **base,
    }


def _select_balanced(
    keep_rows: list[dict[str, Any]],
    drop_rows: list[dict[str, Any]],
    max_instances: int,
    target_drop_rate: float,
) -> list[dict[str, Any]]:
    target_drop = max(1, round(max_instances * target_drop_rate))
    target_keep = max_instances - target_drop
    take_drop = min(target_drop, len(drop_rows))
    take_keep = min(target_keep, len(keep_rows))
    return [*keep_rows[:take_keep], *drop_rows[:take_drop]]


def _load_domain_workflows(workflow_library_dir: Path) -> dict[str, dict[str, Any]]:
    workflows = {}
    for path in sorted(workflow_library_dir.glob('*/workflow_library.yaml')):
        with path.open('r', encoding='utf-8') as f:
            payload = yaml.safe_load(f)
        if isinstance(payload, dict) and payload.get('domain'):
            workflows[str(payload['domain'])] = payload
    return workflows


def _active_records_for_mapper(
    records: list[dict[str, Any]],
    mapper_name: str,
    max_records: int,
    salt: str,
) -> list[dict[str, Any]]:
    active_records = []
    for record in records:
        active = set(_labeling_meta(record).get('active_mapper_names', []))
        if mapper_name in active:
            active_records.append(record)
    active_records.sort(key=lambda row: _stable_id(salt, row.get('id'), row.get('source_name'), row.get('url')))
    return active_records[:max_records]


def _atomic_record(
    *,
    domain: str,
    op_name: str,
    op_kind: str,
    record: dict[str, Any],
    params_by_name: dict[str, dict[str, Any]],
    execution: dict[str, Any],
    threshold_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'instance_id': _stable_id('atomic', domain, op_name, _record_id(record)),
        'benchmark_track': 'atomic',
        'domain': domain,
        'operator': op_name,
        'operator_kind': op_kind,
        'source_record_id': _record_id(record),
        'input_text': str(record.get('text', '')),
        'operator_sequence': [op_name],
        'filter_params_by_name': params_by_name,
        'threshold_meta': threshold_meta or {},
        'reference_status': execution['reference_status'],
        'reference_text': execution['reference_text'],
        'intermediate_text_at_drop': execution.get('intermediate_text_at_drop'),
        'reference_trace': execution['trace'],
    }


def _materialize_atomic_mapper(
    domain: str,
    op_name: str,
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = _active_records_for_mapper(records, op_name, args.max_atomic_candidate_records, f'atomic:{domain}:{op_name}')
    rows = []
    for record in candidates:
        try:
            execution = _execute_workflow(record, [op_name], operators_by_name)
        except Exception:
            continue
        if execution['reference_status'] == 'KEEP' and execution['reference_text'] != str(record.get('text', '')):
            rows.append(
                _atomic_record(
                    domain=domain,
                    op_name=op_name,
                    op_kind='mapper',
                    record=record,
                    params_by_name={},
                    execution=execution,
                )
            )
        if len(rows) >= args.max_atomic_instances_per_op:
            break
    return rows, {
        'domain': domain,
        'operator': op_name,
        'operator_kind': 'mapper',
        'status': 'kept' if rows else 'skipped_no_active_outputs',
        'candidate_count': len(candidates),
        'selected_count': len(rows),
        'selected_keep_count': len(rows),
        'selected_drop_count': 0,
        'threshold_meta': {},
        'filter_params': {},
    }


def _materialize_atomic_filter(
    domain: str,
    op_name: str,
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(
        records,
        key=lambda row: _stable_id(f'atomic:{domain}:{op_name}', row.get('id'), row.get('source_name'), row.get('url')),
    )[: args.max_atomic_candidate_records]
    base_params = _base_params(op_name, operators_by_name)
    values = []
    value_key = None
    value_records = []
    for record in candidates:
        try:
            value, current_value_key = _filter_value(record, [op_name], 0, base_params, operators_by_name)
        except Exception:
            continue
        value_key = value_key or current_value_key
        if value is not None:
            values.append(value)
            value_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        op_name,
        base_params,
        values,
        args.target_drop_rate,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': op_name,
        'status_value_key': value_key,
        'value_count': len(values),
        'atomic_filter': True,
    }
    params_by_name = {op_name: calibrated_params}
    keep_rows = []
    drop_rows = []
    for record in value_records:
        try:
            execution = _execute_workflow(record, [op_name], operators_by_name, params_by_name)
        except Exception:
            continue
        row = _atomic_record(
            domain=domain,
            op_name=op_name,
            op_kind='filter',
            record=record,
            params_by_name=params_by_name,
            execution=execution,
            threshold_meta=threshold_meta,
        )
        if execution['reference_status'] == 'KEEP':
            keep_rows.append(row)
        else:
            drop_rows.append(row)

    selected = _select_balanced(keep_rows, drop_rows, args.max_atomic_instances_per_op, args.target_drop_rate)
    has_min_balance = len(keep_rows) >= args.min_atomic_keep and len(drop_rows) >= args.min_atomic_drop
    selected_rows = selected if has_min_balance else []
    return selected_rows, {
        'domain': domain,
        'operator': op_name,
        'operator_kind': 'filter',
        'status': 'kept' if selected_rows else 'skipped_unbalanced',
        'candidate_count': len(candidates),
        'value_count': len(values),
        'keep_count': len(keep_rows),
        'drop_count': len(drop_rows),
        'selected_count': len(selected_rows),
        'selected_keep_count': sum(1 for row in selected_rows if row['reference_status'] == 'KEEP'),
        'selected_drop_count': sum(1 for row in selected_rows if row['reference_status'] == 'DROP'),
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def _materialize_atomic_ops(
    records_by_domain: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    variants_by_key = plan['execution_variants_by_key']
    for domain in sorted(plan['domain_profiles']):
        profile = plan['domain_profiles'][domain]
        records = records_by_domain.get(domain, [])
        mapper_names = [variants_by_key[key]['name'] for key in profile['mapper_keys']]
        filter_names = [variants_by_key[key]['name'] for key in profile['filter_keys']]
        _log(f'atomic {domain}: {len(mapper_names)} mappers, {len(filter_names)} filters, {len(records)} records')

        for op_name in mapper_names:
            op_rows, summary = _materialize_atomic_mapper(domain, op_name, records, operators_by_name, args)
            rows.extend(op_rows)
            summary_rows.append(summary)
            _log(f"  atomic mapper {domain}/{op_name}: {summary['status']} selected={summary['selected_count']}")

        for op_name in filter_names:
            op_rows, summary = _materialize_atomic_filter(domain, op_name, records, operators_by_name, args)
            rows.extend(op_rows)
            summary_rows.append(summary)
            _log(
                f"  atomic filter {domain}/{op_name}: {summary['status']} "
                f"selected={summary['selected_count']} keep={summary.get('selected_keep_count', 0)} "
                f"drop={summary.get('selected_drop_count', 0)}"
            )

    return rows, summary_rows


def _materialize_main_variant(
    domain: str,
    workflow: dict[str, Any],
    variant: dict[str, Any],
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sequence = list(variant['operator_sequence'])
    mapper_names = _mapper_names(sequence, operators_by_name)
    candidates = _supporting_records(records, mapper_names, args.max_candidate_records, variant['workflow_variant_id'])
    base = {
        'benchmark_track': 'main',
        'domain': domain,
        'workflow_id': workflow['workflow_id'],
        'workflow_variant_id': variant['workflow_variant_id'],
        'workflow_type': variant['workflow_type'],
        'order_family_id': None,
        'order_slot': None,
    }

    filter_index = _first_filter_index(sequence, operators_by_name)
    if filter_index is None:
        rows = []
        for record in candidates:
            execution = _execute_workflow(record, sequence, operators_by_name)
            if execution['reference_status'] == 'KEEP' and execution['reference_text'] != str(record.get('text', '')):
                rows.append(_variant_record(base, record, sequence, {}, execution, None))
            if len(rows) >= args.max_instances_per_variant:
                break
        return rows, {
            **base,
            'status': 'kept' if rows else 'skipped_no_active_outputs',
            'candidate_count': len(candidates),
            'keep_count': len(rows),
            'drop_count': 0,
            'selected_keep_count': len(rows),
            'selected_drop_count': 0,
            'selected_count': len(rows),
            'filter_name': None,
            'threshold_meta': {},
        }

    filter_name = sequence[filter_index]
    base_filter_params = dict(variant.get('filter_params') or _base_params(filter_name, operators_by_name))
    values = []
    value_key = None
    value_records = []
    for record in candidates:
        try:
            value, current_value_key = _filter_value(record, sequence, filter_index, base_filter_params, operators_by_name)
        except Exception:
            continue
        value_key = value_key or current_value_key
        if value is not None:
            values.append(value)
            value_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        filter_name,
        base_filter_params,
        values,
        args.target_drop_rate,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': filter_name,
        'status_value_key': value_key,
        'value_count': len(values),
    }
    filter_params_by_name = {filter_name: calibrated_params}
    keep_rows = []
    drop_rows = []
    for record in value_records:
        try:
            execution = _execute_workflow(record, sequence, operators_by_name, filter_params_by_name)
        except Exception:
            continue
        row = _variant_record(base, record, sequence, filter_params_by_name, execution, threshold_meta)
        if execution['reference_status'] == 'KEEP':
            keep_rows.append(row)
        else:
            drop_rows.append(row)

    selected = _select_balanced(keep_rows, drop_rows, args.max_instances_per_variant, args.target_drop_rate)
    has_min_balance = len(keep_rows) >= args.min_keep and len(drop_rows) >= args.min_drop
    selected_rows = selected if has_min_balance else []
    return selected if has_min_balance else [], {
        **base,
        'status': 'kept' if selected_rows else 'skipped_unbalanced',
        'candidate_count': len(candidates),
        'value_count': len(values),
        'keep_count': len(keep_rows),
        'drop_count': len(drop_rows),
        'selected_keep_count': sum(1 for row in selected_rows if row['reference_status'] == 'KEEP'),
        'selected_drop_count': sum(1 for row in selected_rows if row['reference_status'] == 'DROP'),
        'selected_count': len(selected_rows),
        'filter_name': filter_name,
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def _materialize_order_family(
    domain: str,
    workflow: dict[str, Any],
    family: dict[str, Any],
    records: list[dict[str, Any]],
    operators_by_name: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    variants = list(family.get('variants') or [])
    variants_by_slot = {variant['order_slot']: variant for variant in variants}
    if set(variants_by_slot) != {'front', 'middle', 'end'}:
        return [], {
            'domain': domain,
            'workflow_id': workflow['workflow_id'],
            'order_family_id': family.get('order_family_id'),
            'status': 'skipped_missing_slots',
            'selected_group_count': 0,
            'selected_variant_count': 0,
        }

    mapper_names = _mapper_names(variants_by_slot['end']['operator_sequence'], operators_by_name)
    candidates = _supporting_records(records, mapper_names, args.max_candidate_records, family['order_family_id'])
    filter_name = family['filter_name']
    base_params = dict(variants_by_slot['front'].get('filter_params') or _base_params(filter_name, operators_by_name))
    values = []
    value_key = None
    usable_records = []
    for record in candidates:
        record_values = []
        try:
            for slot in ('front', 'middle', 'end'):
                sequence = list(variants_by_slot[slot]['operator_sequence'])
                filter_index = _first_filter_index(sequence, operators_by_name)
                if filter_index is None:
                    continue
                value, current_value_key = _filter_value(record, sequence, filter_index, base_params, operators_by_name)
                value_key = value_key or current_value_key
                if value is not None:
                    record_values.append(value)
        except Exception:
            continue
        if record_values:
            values.extend(record_values)
            usable_records.append(record)

    calibrated_params, threshold_meta = _calibrate_filter_params_for_target(
        filter_name,
        base_params,
        values,
        args.target_drop_rate,
    )
    threshold_meta = {
        **threshold_meta,
        'filter_name': filter_name,
        'status_value_key': value_key,
        'value_count': len(values),
        'shared_across_order_slots': True,
    }
    filter_params_by_name = {filter_name: calibrated_params}
    selected_rows = []
    sensitive_group_count = 0
    for record in usable_records:
        slot_rows = []
        signatures = set()
        try:
            for slot in ('front', 'middle', 'end'):
                variant = variants_by_slot[slot]
                sequence = list(variant['operator_sequence'])
                execution = _execute_workflow(record, sequence, operators_by_name, filter_params_by_name)
                signature = (execution['reference_status'], execution['reference_text'])
                signatures.add(signature)
                base = {
                    'benchmark_track': 'order_sensitivity',
                    'domain': domain,
                    'workflow_id': workflow['workflow_id'],
                    'workflow_variant_id': variant['workflow_variant_id'],
                    'workflow_type': variant['workflow_type'],
                    'order_family_id': family['order_family_id'],
                    'order_slot': slot,
                    'order_group_instance_id': _stable_id(family['order_family_id'], _record_id(record)),
                    'group_success_rule': family.get('group_success_rule', 'all_slots_correct'),
                }
                slot_rows.append(_variant_record(base, record, sequence, filter_params_by_name, execution, threshold_meta))
        except Exception:
            continue
        if len(signatures) < 2:
            continue
        selected_rows.extend(slot_rows)
        sensitive_group_count += 1
        if sensitive_group_count >= args.max_order_groups_per_family:
            break

    keep_count = sum(1 for row in selected_rows if row['reference_status'] == 'KEEP')
    drop_count = sum(1 for row in selected_rows if row['reference_status'] == 'DROP')
    enough_groups = sensitive_group_count >= args.min_order_sensitive_groups
    return selected_rows if enough_groups else [], {
        'domain': domain,
        'workflow_id': workflow['workflow_id'],
        'order_family_id': family['order_family_id'],
        'filter_name': filter_name,
        'status': 'kept' if enough_groups else 'skipped_not_order_sensitive_enough',
        'candidate_count': len(candidates),
        'usable_record_count': len(usable_records),
        'value_count': len(values),
        'selected_group_count': sensitive_group_count if enough_groups else 0,
        'selected_variant_count': len(selected_rows) if enough_groups else 0,
        'keep_count': keep_count if enough_groups else 0,
        'drop_count': drop_count if enough_groups else 0,
        'threshold_meta': threshold_meta,
        'filter_params': calibrated_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Materialize benchmark instances and DJ references without prompts.')
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--workflow-library-dir', default='data/processed/workflow_library')
    parser.add_argument('--filtered-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--output-dir', default='data/benchmark')
    parser.add_argument('--max-candidate-records', type=int, default=512)
    parser.add_argument('--max-instances-per-variant', type=int, default=20)
    parser.add_argument('--max-order-groups-per-family', type=int, default=10)
    parser.add_argument('--min-keep', type=int, default=5)
    parser.add_argument('--min-drop', type=int, default=5)
    parser.add_argument('--min-order-sensitive-groups', type=int, default=5)
    parser.add_argument('--target-drop-rate', type=float, default=0.5)
    parser.add_argument('--max-atomic-candidate-records', type=int, default=256)
    parser.add_argument('--max-atomic-instances-per-op', type=int, default=20)
    parser.add_argument('--min-atomic-keep', type=int, default=5)
    parser.add_argument('--min-atomic-drop', type=int, default=5)
    parser.add_argument('--skip-atomic', action='store_true', help='Only materialize main/order instances.')
    args = parser.parse_args()

    root = ROOT
    workflow_library_dir = (root / args.workflow_library_dir).resolve()
    filtered_path = (root / args.filtered_path).resolve()
    output_dir = (root / args.output_dir).resolve()

    if not workflow_library_dir.exists():
        raise SystemExit(f'workflow library dir not found: {workflow_library_dir}')
    if not filtered_path.exists():
        raise SystemExit(f'filtered corpus not found: {filtered_path}')
    if not 0.0 < args.target_drop_rate < 1.0:
        raise SystemExit('--target-drop-rate must be in (0, 1)')

    domains_cfg = load_domains_config(root / args.domains_config)
    plan = build_domain_execution_plan(domains_cfg)
    operators_by_name = _operator_lookup(plan)

    _log(f'loading filtered records -> {filtered_path}')
    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(filtered_path):
        domain = record.get('domain')
        if domain:
            records_by_domain[str(domain)].append(record)
    _log(
        'loaded records by domain: '
        + ', '.join(f'{domain}={len(records)}' for domain, records in sorted(records_by_domain.items()))
    )

    domain_workflows = _load_domain_workflows(workflow_library_dir)
    _log(f'loaded workflow libraries for {len(domain_workflows)} domains -> {workflow_library_dir}')

    main_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    atomic_rows: list[dict[str, Any]] = []
    main_summary_rows: list[dict[str, Any]] = []
    order_summary_rows: list[dict[str, Any]] = []
    atomic_summary_rows: list[dict[str, Any]] = []

    if not args.skip_atomic:
        atomic_rows, atomic_summary_rows = _materialize_atomic_ops(records_by_domain, plan, operators_by_name, args)

    for domain_index, (domain, domain_yaml) in enumerate(sorted(domain_workflows.items()), start=1):
        workflows = list(domain_yaml.get('workflows') or [])
        records = records_by_domain.get(domain, [])
        _log(f'[{domain_index}/{len(domain_workflows)}] {domain}: {len(workflows)} workflows, {len(records)} records')
        for workflow_index, workflow in enumerate(workflows, start=1):
            workflow_id = workflow['workflow_id']
            main_variants = list(workflow.get('main_workflow_variants') or [])
            order_families = list(workflow.get('order_sensitivity_families') or [])
            _log(
                f'  [{workflow_index}/{len(workflows)}] {domain}/{workflow_id}: '
                f'{len(main_variants)} main variants, {len(order_families)} order families'
            )
            for variant in main_variants:
                rows, summary = _materialize_main_variant(domain, workflow, variant, records, operators_by_name, args)
                main_rows.extend(rows)
                main_summary_rows.append(summary)
                _log(
                    f"    main {variant['workflow_variant_id']}: {summary['status']} "
                    f"selected={summary['selected_count']} keep={summary.get('keep_count', 0)} drop={summary.get('drop_count', 0)}"
                )

            for family in order_families:
                rows, summary = _materialize_order_family(domain, workflow, family, records, operators_by_name, args)
                order_rows.extend(rows)
                order_summary_rows.append(summary)
                _log(
                    f"    order {family['order_family_id']}: {summary['status']} "
                    f"groups={summary.get('selected_group_count', 0)} variants={summary.get('selected_variant_count', 0)}"
                )

    main_count = _write_jsonl(output_dir / 'main.jsonl', main_rows)
    order_count = _write_jsonl(output_dir / 'order_sensitivity.jsonl', order_rows)
    atomic_count = _write_jsonl(output_dir / 'atomic_ops.jsonl', atomic_rows) if not args.skip_atomic else 0
    pd.DataFrame(main_summary_rows).to_csv(output_dir / 'main_summary.csv', index=False)
    pd.DataFrame(order_summary_rows).to_csv(output_dir / 'order_sensitivity_summary.csv', index=False)
    if not args.skip_atomic:
        pd.DataFrame(atomic_summary_rows).to_csv(output_dir / 'atomic_ops_summary.csv', index=False)

    _log(f'wrote main instances: {main_count} -> {output_dir / "main.jsonl"}')
    _log(f'wrote order-sensitivity instances: {order_count} -> {output_dir / "order_sensitivity.jsonl"}')
    if not args.skip_atomic:
        _log(f'wrote atomic-op instances: {atomic_count} -> {output_dir / "atomic_ops.jsonl"}')
    _log(f'wrote summaries -> {output_dir}')


if __name__ == '__main__':
    main()
