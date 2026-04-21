#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icdrbench.config import load_domains_config
from icdrbench.dj_operator_loader import Fields, create_operator
from icdrbench.domain_assignment import build_domain_execution_plan


FILTER_STATUS_RULES: dict[str, dict[str, Any]] = {
    'alphanumeric_filter': {
        'value_key': lambda params: 'alpha_token_ratio' if params.get('tokenization') else 'alnum_ratio',
        'min_key': 'min_ratio',
        'max_key': 'max_ratio',
    },
    'average_line_length_filter': {'value_key': 'avg_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'character_repetition_filter': {'value_key': 'char_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'flagged_words_filter': {'value_key': 'flagged_words_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'maximum_line_length_filter': {'value_key': 'max_line_length', 'min_key': 'min_len', 'max_key': 'max_len'},
    'stopwords_filter': {'value_key': 'stopwords_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'text_length_filter': {'value_key': 'text_len', 'min_key': 'min_len', 'max_key': 'max_len'},
    'word_repetition_filter': {'value_key': 'word_rep_ratio', 'min_key': 'min_ratio', 'max_key': 'max_ratio'},
    'words_num_filter': {'value_key': 'num_words', 'min_key': 'min_num', 'max_key': 'max_num'},
}

FILTER_CALIBRATION_RULES: dict[str, dict[str, Any]] = {
    'alphanumeric_filter': {'direction': 'min', 'quantile': 0.20},
    'average_line_length_filter': {'direction': 'max', 'quantile': 0.80},
    'character_repetition_filter': {'direction': 'max', 'quantile': 0.80},
    'flagged_words_filter': {'direction': 'max', 'quantile': 0.80},
    'maximum_line_length_filter': {'direction': 'max', 'quantile': 0.80},
    'stopwords_filter': {'direction': 'min', 'quantile': 0.20},
    'text_length_filter': {'direction': 'min', 'quantile': 0.20},
    'word_repetition_filter': {'direction': 'max', 'quantile': 0.80},
    'words_num_filter': {'direction': 'min', 'quantile': 0.20},
}

SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS = 80_000
EXPENSIVE_LONG_TEXT_MAPPERS = {
    'remove_repeat_sentences_mapper',
    'remove_words_with_incorrect_substrings_mapper',
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                preview = line[:200]
                raise ValueError(
                    f'Invalid JSONL record in {path} at line {lineno}: {exc.msg}. Line preview: {preview!r}'
                ) from exc


def _call_optional_context(method, payload):
    try:
        return method(payload, context=True)
    except TypeError:
        return method(payload)


def _build_batch(text: str, suffix: str) -> dict[str, Any]:
    return {
        'text': [text],
        Fields.stats: [{}],
        Fields.context: [{}],
        Fields.meta: [{}],
        Fields.suffix: [suffix],
    }


def _build_sample(text: str, suffix: str) -> dict[str, Any]:
    return {
        'text': text,
        Fields.stats: {},
        Fields.context: {},
        Fields.meta: {},
        Fields.suffix: suffix,
    }


def _infer_suffix(record: dict[str, Any]) -> str:
    source_name = record.get('source_name')
    if isinstance(source_name, str) and source_name:
        suffix = Path(source_name).suffix
        if suffix:
            return suffix

    url = record.get('url')
    if isinstance(url, str) and url:
        suffix = Path(url).suffix
        if suffix:
            return suffix

    text = str(record.get('text', ''))
    lowered = text.lower()
    if '\\begin{document}' in lowered or '\\section' in lowered:
        return '.tex'
    if '<html' in lowered or '</html>' in lowered:
        return '.html'
    return ''


def _apply_mapper_text(op_name: str, text: str, params: dict[str, Any], suffix: str) -> tuple[str, dict[str, Any]]:
    if op_name in EXPENSIVE_LONG_TEXT_MAPPERS and len(text) > SAFE_MAX_CHARS_FOR_EXPENSIVE_MAPPERS:
        return text, {
            'active': False,
            'skipped': 'text_too_long_for_expensive_mapper',
            'output_length': len(text),
            'delta_chars': 0,
        }

    op = create_operator(op_name, **params)
    if hasattr(op, 'process_batched') and op.is_batched_op():
        result = op.process_batched(_build_batch(text, suffix))
        output_text = result['text'][0]
    else:
        result = op.process_single(_build_sample(text, suffix))
        output_text = result['text']

    return output_text, {
        'active': output_text != text,
        'output_length': len(output_text),
        'delta_chars': len(output_text) - len(text),
    }


def _evaluate_filter(op_name: str, text: str, params: dict[str, Any], suffix: str) -> dict[str, Any]:
    op = create_operator(op_name, **params)
    if hasattr(op, 'compute_stats_batched') and op.is_batched_op():
        batch = _call_optional_context(op.compute_stats_batched, _build_batch(text, suffix))
        keep_iter = op.process_batched(batch)
        keep = list(keep_iter)[0] if not isinstance(keep_iter, list) else keep_iter[0]
        stats = batch[Fields.stats][0]
    else:
        sample = _build_sample(text, suffix)
        if hasattr(op, 'compute_stats_single'):
            sample = _call_optional_context(op.compute_stats_single, sample)
        keep = op.process_single(sample)
        stats = sample[Fields.stats]

    return {
        'keep': bool(keep),
        'status': 'KEEP' if keep else 'DROP',
        'stats': stats,
    }


def _resolve_status_value_key(op_name: str, params: dict[str, Any]) -> str | None:
    rule = FILTER_STATUS_RULES.get(op_name)
    if rule is None:
        return None
    value_key = rule['value_key']
    return value_key(params) if callable(value_key) else value_key


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _summarize_values(values: list[float]) -> dict[str, Any]:
    return {
        'count': len(values),
        'mean': _round_float(mean(values)) if values else None,
        'min': _round_float(min(values)) if values else None,
        'p10': _round_float(_percentile(values, 0.10)),
        'p20': _round_float(_percentile(values, 0.20)),
        'p50': _round_float(_percentile(values, 0.50)),
        'p80': _round_float(_percentile(values, 0.80)),
        'p90': _round_float(_percentile(values, 0.90)),
        'max': _round_float(max(values)) if values else None,
    }


def _calibrate_filter_params(op_name: str, base_params: dict[str, Any], values: list[float]) -> dict[str, Any]:
    rule = FILTER_STATUS_RULES.get(op_name)
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    params = dict(base_params)
    if not rule or not calibration or not values:
        return params

    threshold = _percentile(values, float(calibration['quantile']))
    if threshold is None:
        return params

    if calibration['direction'] == 'min':
        params[rule['min_key']] = _round_float(threshold)
        params.pop(rule['max_key'], None)
    else:
        params[rule['max_key']] = _round_float(threshold)
        params.pop(rule['min_key'], None)
    return params


def _threshold_rule_label(op_name: str) -> str:
    calibration = FILTER_CALIBRATION_RULES.get(op_name)
    if not calibration:
        return 'not_calibrated'
    pct = int(float(calibration['quantile']) * 100)
    return f"{calibration['direction']}_p{pct}"


def _parse_operator_set(blob: str) -> list[str]:
    return [item.strip() for item in blob.split(' | ') if item.strip()]


def _labeling_meta(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get('meta')
    if not isinstance(meta, dict):
        return {}
    payload = meta.get('icdrbench_domain_labeling')
    return payload if isinstance(payload, dict) else {}


def _workflow_rows_for_domain(domain_dir: Path) -> list[dict[str, Any]]:
    workflow_csv = domain_dir / 'selected_workflows.csv'
    if not workflow_csv.exists():
        return []
    df = pd.read_csv(workflow_csv)
    return df.to_dict(orient='records')


def _ordered_mapper_sequence(
    domain: str,
    operator_set: list[str],
    plan: dict[str, Any],
) -> list[dict[str, Any]]:
    domain_profile = plan['domain_profiles'][domain]
    variants_by_key = plan['execution_variants_by_key']
    wanted = set(operator_set)
    ordered: list[dict[str, Any]] = []
    for key in domain_profile['mapper_keys']:
        variant = variants_by_key[key]
        if variant['name'] in wanted:
            ordered.append(variant)
    seen = {variant['name'] for variant in ordered}
    for op_name in operator_set:
        if op_name in seen:
            continue
        for variant in plan['execution_variants']:
            if variant['kind'] == 'mapper' and variant['name'] == op_name:
                ordered.append(variant)
                break
    return ordered


def _domain_filter_variants(domain: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    domain_profile = plan['domain_profiles'][domain]
    variants_by_key = plan['execution_variants_by_key']
    return [variants_by_key[key] for key in domain_profile['filter_keys']]


def _supporting_records(
    records: list[dict[str, Any]],
    operator_set: list[str],
    max_records: int,
) -> list[dict[str, Any]]:
    wanted = set(operator_set)
    supported = []
    for record in records:
        meta = _labeling_meta(record)
        active = set(meta.get('active_mapper_names', []))
        if wanted.issubset(active):
            supported.append(record)
    return supported[:max_records]


def _replay_mapper_checkpoints(
    record: dict[str, Any],
    ordered_mappers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    suffix = _infer_suffix(record)
    current_text = str(record.get('text', ''))
    checkpoints = [
        {
            'checkpoint_id': 'S0',
            'step_index': 0,
            'after_operator': None,
            'text': current_text,
        }
    ]
    for idx, variant in enumerate(ordered_mappers, start=1):
        current_text, mapper_result = _apply_mapper_text(variant['name'], current_text, dict(variant.get('params', {})), suffix)
        checkpoints.append(
            {
                'checkpoint_id': f'S{idx}',
                'step_index': idx,
                'after_operator': variant['name'],
                'text': current_text,
                'mapper_result': mapper_result,
            }
        )
    return checkpoints


def _collect_checkpoint_filter_stats(
    ordered_mappers: list[dict[str, Any]],
    filter_variants: list[dict[str, Any]],
    support_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    checkpoint_info: dict[str, dict[str, Any]] = {}
    filter_info: dict[str, dict[str, Any]] = {}

    for record in support_records:
        suffix = _infer_suffix(record)
        try:
            checkpoints = _replay_mapper_checkpoints(record, ordered_mappers)
        except Exception:
            continue

        for checkpoint in checkpoints:
            checkpoint_info[checkpoint['checkpoint_id']] = {
                'checkpoint_id': checkpoint['checkpoint_id'],
                'step_index': checkpoint['step_index'],
                'after_operator': checkpoint['after_operator'],
            }
            for filter_variant in filter_variants:
                op_name = filter_variant['name']
                params = dict(filter_variant.get('params', {}))
                value_key = _resolve_status_value_key(op_name, params)
                if value_key is None:
                    continue
                try:
                    evaluation = _evaluate_filter(op_name, checkpoint['text'], params, suffix)
                except Exception:
                    continue
                value = evaluation.get('stats', {}).get(value_key)
                if isinstance(value, (int, float)):
                    raw_values[(op_name, checkpoint['checkpoint_id'])].append(float(value))
                    filter_info[op_name] = {
                        'filter_name': op_name,
                        'filter_params': params,
                        'status_value_key': value_key,
                    }

    checkpoint_rows: list[dict[str, Any]] = []
    attachment_candidates: list[dict[str, Any]] = []
    previous_by_filter: dict[str, dict[str, Any]] = {}

    for filter_name in sorted(filter_info):
        checkpoints = sorted(checkpoint_info.values(), key=lambda row: row['step_index'])
        previous_by_filter[filter_name] = {}
        for checkpoint in checkpoints:
            values = raw_values.get((filter_name, checkpoint['checkpoint_id']), [])
            if not values:
                continue
            summary = _summarize_values(values)
            prev = previous_by_filter[filter_name]
            prev_mean = prev.get('mean')
            mean_value = summary['mean']
            delta = None
            rel_delta = None
            if isinstance(prev_mean, (int, float)) and isinstance(mean_value, (int, float)):
                delta = mean_value - prev_mean
                if abs(prev_mean) > 1e-12:
                    rel_delta = delta / abs(prev_mean)

            base_params = dict(filter_info[filter_name]['filter_params'])
            calibrated_params = _calibrate_filter_params(filter_name, base_params, values)
            candidate = {
                **filter_info[filter_name],
                **checkpoint,
                'support_records': summary['count'],
                'threshold_selection_rule': _threshold_rule_label(filter_name),
                'calibrated_filter_params': calibrated_params,
                'stat_mean': summary['mean'],
                'stat_min': summary['min'],
                'stat_p10': summary['p10'],
                'stat_p20': summary['p20'],
                'stat_p50': summary['p50'],
                'stat_p80': summary['p80'],
                'stat_p90': summary['p90'],
                'stat_max': summary['max'],
                'delta_from_prev_mean': _round_float(delta),
                'relative_delta_from_prev_mean': _round_float(rel_delta),
            }
            checkpoint_rows.append(candidate)
            attachment_candidates.append(
                {
                    **candidate,
                    'selection_score': (
                        summary['count'],
                        abs(rel_delta or 0.0),
                        abs(delta or 0.0),
                        checkpoint['step_index'],
                    ),
                }
            )
            previous_by_filter[filter_name] = summary

    return checkpoint_rows, attachment_candidates


def _select_stage_attachments(
    attachment_candidates: list[dict[str, Any]],
    *,
    stage: str,
    final_step_index: int,
    min_filter_support: int,
    max_filters_per_workflow: int,
) -> list[dict[str, Any]]:
    if stage == 'raw':
        candidates = [row for row in attachment_candidates if row['step_index'] == 0]
    elif stage == 'final':
        candidates = [row for row in attachment_candidates if row['step_index'] == final_step_index]
    elif stage == 'middle':
        candidates = [row for row in attachment_candidates if 0 < row['step_index'] < final_step_index]
    else:
        candidates = list(attachment_candidates)

    candidates = [row for row in candidates if row['support_records'] >= min_filter_support]
    candidates.sort(
        key=lambda row: (
            -row['selection_score'][0],
            -row['selection_score'][1],
            -row['selection_score'][2],
            row['filter_name'],
        )
    )
    selected: list[dict[str, Any]] = []
    seen_filters: set[str] = set()
    for row in candidates:
        if row['filter_name'] in seen_filters:
            continue
        seen_filters.add(row['filter_name'])
        selected.append({k: v for k, v in row.items() if k != 'selection_score'})
        if len(selected) >= max_filters_per_workflow:
            break
    return selected


def _materialize_variants(
    workflow_id: str,
    ordered_mappers: list[dict[str, Any]],
    *,
    raw_attachments: list[dict[str, Any]],
    final_attachments: list[dict[str, Any]],
    middle_attachments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mapper_names = [variant['name'] for variant in ordered_mappers]
    main_variants = [
        {
            'workflow_variant_id': f'{workflow_id}__clean_only',
            'workflow_type': 'clean-only',
            'benchmark_track': 'main',
            'operator_sequence': mapper_names,
            'filter_name': None,
            'filter_checkpoint_id': None,
            'filter_step_index': None,
            'filter_params': None,
        }
    ]

    for idx, attachment in enumerate(raw_attachments, start=1):
        main_variants.append(
            {
                'workflow_variant_id': f'{workflow_id}__filter_then_clean_{idx:02d}',
                'workflow_type': 'filter-then-clean',
                'benchmark_track': 'main',
                'operator_sequence': [attachment['filter_name'], *mapper_names],
                'filter_name': attachment['filter_name'],
                'filter_checkpoint_id': attachment['checkpoint_id'],
                'filter_step_index': attachment['step_index'],
                'filter_params': attachment['calibrated_filter_params'],
                'threshold_selection_rule': attachment['threshold_selection_rule'],
            }
        )

    for idx, attachment in enumerate(final_attachments, start=1):
        main_variants.append(
            {
                'workflow_variant_id': f'{workflow_id}__clean_then_filter_{idx:02d}',
                'workflow_type': 'clean-then-filter',
                'benchmark_track': 'main',
                'operator_sequence': [*mapper_names, attachment['filter_name']],
                'filter_name': attachment['filter_name'],
                'filter_checkpoint_id': attachment['checkpoint_id'],
                'filter_step_index': attachment['step_index'],
                'filter_params': attachment['calibrated_filter_params'],
                'threshold_selection_rule': attachment['threshold_selection_rule'],
            }
        )

    order_variants: list[dict[str, Any]] = []
    for idx, attachment in enumerate(middle_attachments, start=1):
        split_at = int(attachment['step_index'])
        order_variants.append(
            {
                'workflow_variant_id': f'{workflow_id}__clean_filter_clean_s{split_at}_{idx:02d}',
                'workflow_type': 'clean-filter-clean',
                'benchmark_track': 'order_sensitivity',
                'operator_sequence': [
                    *mapper_names[:split_at],
                    attachment['filter_name'],
                    *mapper_names[split_at:],
                ],
                'filter_name': attachment['filter_name'],
                'filter_checkpoint_id': attachment['checkpoint_id'],
                'filter_step_index': attachment['step_index'],
                'filter_params': attachment['calibrated_filter_params'],
                'threshold_selection_rule': attachment['threshold_selection_rule'],
            }
        )
    return main_variants, order_variants


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Materialize main and order-sensitivity workflow drafts from mined clean workflows.'
    )
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--workflow-mining-dir', default='data/processed/workflow_mining')
    parser.add_argument('--filtered-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--output-dir', default='data/processed/workflow_library')
    parser.add_argument('--max-support-records', type=int, default=128)
    parser.add_argument('--min-filter-support', type=int, default=5)
    parser.add_argument('--max-filters-per-workflow', type=int, default=3)
    args = parser.parse_args()

    root = ROOT
    domains_cfg = load_domains_config(root / args.domains_config)
    plan = build_domain_execution_plan(domains_cfg)
    workflow_mining_dir = (root / args.workflow_mining_dir).resolve()
    filtered_path = (root / args.filtered_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not filtered_path.exists():
        raise SystemExit(f'filtered corpus not found: {filtered_path}')
    if not workflow_mining_dir.exists():
        raise SystemExit(f'workflow mining dir not found: {workflow_mining_dir}')

    records_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in iter_jsonl(filtered_path):
        domain = record.get('domain')
        if domain:
            records_by_domain[str(domain)].append(record)

    summary_rows: list[dict[str, Any]] = []
    global_yaml = {'domains': {}}

    for domain_dir in sorted(path for path in workflow_mining_dir.iterdir() if path.is_dir()):
        domain = domain_dir.name
        workflow_rows = _workflow_rows_for_domain(domain_dir)
        if not workflow_rows:
            continue

        domain_records = records_by_domain.get(domain, [])
        ordered_filter_variants = _domain_filter_variants(domain, plan)
        domain_yaml = {'domain': domain, 'workflows': []}
        attachment_rows: list[dict[str, Any]] = []
        checkpoint_rows: list[dict[str, Any]] = []
        order_candidate_rows: list[dict[str, Any]] = []
        variant_rows: list[dict[str, Any]] = []

        for row in workflow_rows:
            workflow_id = str(row['workflow_id'])
            operator_set = _parse_operator_set(str(row['operators']))
            ordered_mappers = _ordered_mapper_sequence(domain, operator_set, plan)
            support_records = _supporting_records(
                domain_records,
                operator_set,
                max_records=args.max_support_records,
            )
            filter_checkpoint_rows, attachment_candidates = _collect_checkpoint_filter_stats(
                ordered_mappers=ordered_mappers,
                filter_variants=ordered_filter_variants,
                support_records=support_records,
            )
            final_step_index = len(ordered_mappers)
            raw_attachments = _select_stage_attachments(
                attachment_candidates,
                stage='raw',
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_filters_per_workflow=args.max_filters_per_workflow,
            )
            final_attachments = _select_stage_attachments(
                attachment_candidates,
                stage='final',
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_filters_per_workflow=args.max_filters_per_workflow,
            )
            middle_attachments = _select_stage_attachments(
                attachment_candidates,
                stage='middle',
                final_step_index=final_step_index,
                min_filter_support=args.min_filter_support,
                max_filters_per_workflow=args.max_filters_per_workflow,
            )
            main_variants, order_variants = _materialize_variants(
                workflow_id,
                ordered_mappers,
                raw_attachments=raw_attachments,
                final_attachments=final_attachments,
                middle_attachments=middle_attachments,
            )
            all_variants = [*main_variants, *order_variants]
            mapper_sequence = ' -> '.join(variant['name'] for variant in ordered_mappers)

            for checkpoint_row in filter_checkpoint_rows:
                checkpoint_rows.append(
                    {
                        'domain': domain,
                        'workflow_id': workflow_id,
                        'mapper_sequence': mapper_sequence,
                        **checkpoint_row,
                    }
                )

            selected_attachments = [
                *[(attachment, 'filter-then-clean', 'main') for attachment in raw_attachments],
                *[(attachment, 'clean-then-filter', 'main') for attachment in final_attachments],
                *[(attachment, 'clean-filter-clean', 'order_sensitivity') for attachment in middle_attachments],
            ]
            for attachment, workflow_type, benchmark_track in selected_attachments:
                attachment_row = {
                    'domain': domain,
                    'workflow_id': workflow_id,
                    'mapper_sequence': mapper_sequence,
                    'workflow_type': workflow_type,
                    'benchmark_track': benchmark_track,
                    **{k: v for k, v in attachment.items() if k != 'selection_score'},
                }
                if benchmark_track == 'order_sensitivity':
                    continue
                else:
                    attachment_rows.append(attachment_row)

            for variant in all_variants:
                variant_rows.append(
                    {
                        'domain': domain,
                        'workflow_id': workflow_id,
                        'workflow_variant_id': variant['workflow_variant_id'],
                        'workflow_type': variant['workflow_type'],
                        'benchmark_track': variant['benchmark_track'],
                        'operator_sequence': ' -> '.join(variant['operator_sequence']),
                        'length': len(variant['operator_sequence']),
                        'filter_name': variant.get('filter_name'),
                        'filter_checkpoint_id': variant.get('filter_checkpoint_id'),
                        'filter_step_index': variant.get('filter_step_index'),
                        'filter_params': variant.get('filter_params'),
                    }
                )

            for attachment, variant in zip(middle_attachments, order_variants):
                order_candidate_rows.append(
                    {
                        'domain': domain,
                        'workflow_id': workflow_id,
                        'workflow_variant_id': variant['workflow_variant_id'],
                        'workflow_type': variant['workflow_type'],
                        'operator_sequence': ' -> '.join(variant['operator_sequence']),
                        'length': len(variant['operator_sequence']),
                        'mapper_sequence': mapper_sequence,
                        **{k: v for k, v in attachment.items() if k != 'selection_score'},
                    }
                )

            domain_yaml['workflows'].append(
                {
                    'workflow_id': workflow_id,
                    'family_id': row.get('family_id'),
                    'selection_source': row.get('selection_source', 'bottom_up_exact_signature'),
                    'support': int(row.get('support', 0) or 0),
                    'support_ratio': float(row.get('support_ratio', 0.0) or 0.0),
                    'mapper_operator_set': operator_set,
                    'ordered_clean_sequence': [variant['name'] for variant in ordered_mappers],
                    'support_records_used_for_filter_scan': len(support_records),
                    'main_workflow_variants': main_variants,
                    'order_sensitivity_variants': order_variants,
                    'selected_filter_attachments': {
                        'filter_then_clean': raw_attachments,
                        'clean_then_filter': final_attachments,
                        'clean_filter_clean': middle_attachments,
                    },
                    'checkpoint_filter_stats_file': 'checkpoint_filter_stats.csv',
                    'curation_status': 'draft_workflow_ready_for_threshold_and_prompt_curation',
                }
            )

            summary_rows.append(
                {
                    'domain': domain,
                    'workflow_id': workflow_id,
                    'support': int(row.get('support', 0) or 0),
                    'mapper_length': len(ordered_mappers),
                    'num_main_variants': len(main_variants),
                    'num_order_sensitivity_variants': len(order_variants),
                    'num_filter_then_clean': len(raw_attachments),
                    'num_clean_then_filter': len(final_attachments),
                    'num_clean_filter_clean': len(middle_attachments),
                }
            )

        global_yaml['domains'][domain] = domain_yaml
        domain_out_dir = output_dir / domain
        domain_out_dir.mkdir(parents=True, exist_ok=True)
        with (domain_out_dir / 'workflow_library.yaml').open('w', encoding='utf-8') as f:
            yaml.safe_dump(domain_yaml, f, sort_keys=False, allow_unicode=True)
        pd.DataFrame(variant_rows).to_csv(domain_out_dir / 'workflow_variants.csv', index=False)
        pd.DataFrame(attachment_rows).to_csv(domain_out_dir / 'filter_attachments.csv', index=False)
        pd.DataFrame(checkpoint_rows).to_csv(domain_out_dir / 'checkpoint_filter_stats.csv', index=False)
        pd.DataFrame(order_candidate_rows).to_csv(domain_out_dir / 'order_sensitivity_candidates.csv', index=False)

    with (output_dir / 'workflow_library.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(global_yaml, f, sort_keys=False, allow_unicode=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / 'workflow_library_summary.csv', index=False)

    print(f'wrote workflow library -> {output_dir}')


if __name__ == '__main__':
    main()
