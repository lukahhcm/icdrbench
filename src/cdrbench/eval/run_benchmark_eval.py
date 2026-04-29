#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from cdrbench.eval.metrics import compute_recipe_metrics
from cdrbench.llm_utils import build_client, parse_json_response, resolve_api_key, resolve_base_url, resolve_model


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_CONFIG = ROOT / 'configs' / 'recipe_prompting.yaml'
DEFAULT_PROGRESS_EVERY = 20
LOCAL_HOSTS = {'127.0.0.1', '0.0.0.0', '::1', 'localhost'}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp_path.replace(path)
    return len(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as f:
        f.write(text)
        if not text.endswith('\n'):
            f.write('\n')
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = yaml.safe_load(f)
    return payload if isinstance(payload, dict) else {}


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _copy_recipe_identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'recipe_id': _first_present(row, 'recipe_id', 'workflow_id'),
        'recipe_variant_id': _first_present(row, 'recipe_variant_id', 'workflow_variant_id'),
        'recipe_type': _first_present(row, 'recipe_type', 'workflow_type'),
    }


def _infer_local_api_key(base_url: str) -> str:
    host = (urlparse(base_url).hostname or '').strip().lower()
    if host in LOCAL_HOSTS:
        return 'EMPTY'
    return resolve_api_key(None)


def _resolved_api_key(explicit_api_key: str | None, base_url: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    try:
        return resolve_api_key(None)
    except RuntimeError:
        return _infer_local_api_key(base_url)


def _prompt_style_config(prompt_cfg: dict[str, Any]) -> dict[str, Any]:
    prompt_styles = prompt_cfg.get('prompt_styles') or {}
    style_cfg = prompt_styles.get('user_natural_v1') or {}
    return style_cfg if isinstance(style_cfg, dict) else {}


def _default_system_prompt(prompt_cfg: dict[str, Any]) -> str:
    style_cfg = _prompt_style_config(prompt_cfg)
    prompt = style_cfg.get('system_prompt')
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return (
        'You are a careful data refinement engine. '
        'Follow the user request exactly and in order. '
        'Return only the required JSON object.'
    )


def _json_schema_hint(prompt_cfg: dict[str, Any]) -> str:
    style_cfg = _prompt_style_config(prompt_cfg)
    output_contract = style_cfg.get('output_contract') or {}
    hint = output_contract.get('json_schema_hint') if isinstance(output_contract, dict) else None
    if isinstance(hint, str) and hint.strip():
        return hint.strip()
    return '{"status":"KEEP","clean_text":"..."} or {"status":"DROP","clean_text":"..."}'


def _select_prompt_variant(row: dict[str, Any], prompt_variant_index: int) -> dict[str, Any]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
        if prompt_variant_index < 0 or prompt_variant_index >= len(prompt_variants):
            raise IndexError(
                f'prompt_variant_index={prompt_variant_index} is out of range for '
                f'instance_id={row.get("instance_id")} with {len(prompt_variants)} variants'
            )
        selected = prompt_variants[prompt_variant_index]
        if isinstance(selected, dict):
            return selected
    user_requirement = row.get('user_requirement')
    if isinstance(user_requirement, str) and user_requirement.strip():
        return {
            'style_id': str(row.get('style_id') or 'single_prompt'),
            'style_label': str(row.get('style_label') or 'Single Prompt'),
            'user_requirement': user_requirement.strip(),
        }
    raise RuntimeError(f'No usable prompt variant found for instance_id={row.get("instance_id")}.')


def _available_prompt_variant_indices(row: dict[str, Any]) -> list[int]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
        return list(range(len(prompt_variants)))
    if isinstance(row.get('user_requirement'), str) and str(row.get('user_requirement')).strip():
        return [0]
    raise RuntimeError(f'No prompt variants available for instance_id={row.get("instance_id")}.')


def _parse_prompt_variant_indices(value: str | None, row: dict[str, Any]) -> list[int]:
    available = _available_prompt_variant_indices(row)
    if value is None or not value.strip():
        return [0]
    if value.strip().lower() == 'all':
        return available
    requested = []
    for part in value.split(','):
        token = part.strip()
        if not token:
            continue
        index = int(token)
        if index not in available:
            raise IndexError(
                f'prompt_variant_index={index} is out of range for '
                f'instance_id={row.get("instance_id")} with available={available}'
            )
        requested.append(index)
    if not requested:
        raise RuntimeError(f'No prompt variant indices resolved for instance_id={row.get("instance_id")}.')
    return sorted(set(requested))


def _render_user_prompt(row: dict[str, Any], user_requirement: str, schema_hint: str) -> str:
    return (
        f"Task:\n{user_requirement}\n\n"
        "Raw input text:\n"
        "<input>\n"
        f"{str(row.get('input_text', ''))}\n"
        "</input>\n\n"
        "Return JSON only.\n"
        f"Use exactly this schema: {schema_hint}\n"
        "Rules:\n"
        "- status must be KEEP or DROP.\n"
        "- If status is KEEP, clean_text must be the final refined text.\n"
        "- If status is DROP, clean_text must be the text state at the point where the sample is rejected.\n"
        "- Do not output markdown, code fences, or explanations.\n"
    )


def _chat_completion(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    request_kwargs: dict[str, Any] = {
        'model': model,
        'temperature': temperature,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
    }
    if max_tokens > 0:
        request_kwargs['max_tokens'] = max_tokens
    completion = client.chat.completions.create(**request_kwargs)
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError('LLM returned empty content.')
    usage = getattr(completion, 'usage', None)
    usage_payload = {}
    if usage is not None:
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = getattr(usage, field, None)
            if value is not None:
                usage_payload[field] = value
    return content, usage_payload


def _extract_prediction_payload(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = parse_json_response(response_text)
    except Exception as exc:  # pragma: no cover - exercised in CLI usage
        return None, f'json_parse_error: {exc}'
    if not isinstance(payload, dict):
        return None, 'json_parse_error: response is not a JSON object'
    return payload, None


def _extract_prediction_fields(
    row: dict[str, Any],
    *,
    explicit_status_field: str | None,
    explicit_text_field: str | None,
) -> tuple[str, str]:
    candidate_objects: list[dict[str, Any]] = [row]
    for key in ('parsed_response', 'response_json', 'prediction_payload'):
        value = row.get(key)
        if isinstance(value, dict):
            candidate_objects.append(value)

    def pick(explicit: str | None, fallbacks: list[str]) -> Any:
        if explicit:
            for obj in candidate_objects:
                if explicit in obj:
                    return obj.get(explicit)
            return None
        for name in fallbacks:
            for obj in candidate_objects:
                if name in obj:
                    return obj.get(name)
        return None

    status_value = pick(explicit_status_field, ['predicted_status', 'status'])
    text_value = pick(explicit_text_field, ['predicted_clean_text', 'clean_text', 'text'])
    return ('' if status_value is None else str(status_value), '' if text_value is None else str(text_value))


def _base_score_row(
    benchmark_row: dict[str, Any],
    predicted_status: str,
    predicted_clean_text: str,
) -> dict[str, Any]:
    keep_fields = [
        'instance_id',
        'benchmark_track',
        'domain',
        'source_domain',
        'order_family_id',
        'order_slot',
        'order_group_instance_id',
        'group_success_rule',
        'operator',
        'operator_kind',
        'source_record_id',
        'input_text',
        'input_length_chars',
        'input_length_bucket',
        'reference_status',
        'reference_text',
    ]
    output_row = _copy_recipe_identity_fields(benchmark_row)
    output_row.update({field: benchmark_row[field] for field in keep_fields if field in benchmark_row})
    output_row['predicted_status'] = predicted_status
    output_row['predicted_clean_text'] = predicted_clean_text
    output_row.update(
        compute_recipe_metrics(
            input_text=benchmark_row.get('input_text', ''),
            reference_status=benchmark_row.get('reference_status', ''),
            reference_text=benchmark_row.get('reference_text', ''),
            predicted_status=predicted_status,
            predicted_clean_text=predicted_clean_text,
        )
    )
    return output_row


def _base_inference_row(eval_row: dict[str, Any]) -> dict[str, Any]:
    keep_fields = [
        'instance_id',
        'benchmark_track',
        'domain',
        'source_domain',
        'order_family_id',
        'order_slot',
        'order_group_instance_id',
        'group_success_rule',
        'operator',
        'operator_kind',
        'source_record_id',
        'input_text',
        'input_length_chars',
        'input_length_bucket',
        'reference_status',
        'reference_text',
        'prompt_variant_count',
        'prompt_candidate_pool_count',
        'prompt_sampling_policy',
        'prompt_sampling_seed',
    ]
    output_row = _copy_recipe_identity_fields(eval_row)
    output_row.update({field: eval_row[field] for field in keep_fields if field in eval_row})
    recipe_prompt_key = _first_present(eval_row, 'recipe_prompt_key', 'workflow_prompt_key')
    if recipe_prompt_key is not None:
        output_row['recipe_prompt_key'] = recipe_prompt_key
    return output_row


def _normalize_variant_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get('variant_predictions')
    if isinstance(variants, list):
        normalized = [variant for variant in variants if isinstance(variant, dict)]
        return sorted(normalized, key=lambda item: int(item.get('prompt_variant_index', 0) or 0))
    return []


def _existing_variant_prediction_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    for variant in _normalize_variant_predictions(row):
        index = int(variant.get('prompt_variant_index', 0) or 0)
        mapping[index] = variant
    return mapping


def _group_rows_by_instance_id(rows: list[dict[str, Any]], key: str = 'instance_id') -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance_id = str(row.get(key) or '')
        if instance_id:
            grouped.setdefault(instance_id, []).append(row)
    return grouped


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_mean([1.0 if bool(row.get(key)) else 0.0 for row in rows])


def _slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)

    summary_rows = []
    for value in sorted(grouped):
        bucket = grouped[value]
        summary_rows.append(
            {
                key: value,
                'count': len(bucket),
                'recipe_success_rate': _rate(bucket, 'recipe_success'),
                'status_accuracy': _rate(bucket, 'status_match'),
                'exact_text_match_rate': _rate(bucket, 'text_exact_match'),
                'canonical_text_match_rate': _rate(bucket, 'text_canonical_match'),
                'avg_refinement_gain': _safe_mean([float(row.get('refinement_gain', 0.0)) for row in bucket]),
            }
        )
    return summary_rows


def _instance_slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)

    summary_rows = []
    for value in sorted(grouped):
        bucket = grouped[value]
        summary_rows.append(
            {
                key: value,
                'count': len(bucket),
                'mean_rs': _safe_mean([float(item.get('mean_rs', 0.0)) for item in bucket]),
                'rs_at_k': _rate(bucket, 'rs_at_k'),
                'mean_rg': _safe_mean([float(item.get('mean_rg', 0.0)) for item in bucket]),
            }
        )
    return summary_rows


def _order_group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = row.get('order_group_instance_id')
        if group_id:
            grouped.setdefault(str(group_id), []).append(row)

    results = []
    for group_id in sorted(grouped):
        bucket = grouped[group_id]
        rule = str(bucket[0].get('group_success_rule') or 'all_slots_correct')
        if rule == 'all_slots_correct':
            success = all(bool(item.get('recipe_success')) for item in bucket)
        else:
            success = all(bool(item.get('recipe_success')) for item in bucket)
        results.append(
            {
                'order_group_instance_id': group_id,
                'group_success_rule': rule,
                'slot_count': len(bucket),
                'order_consistent_success': success,
                'group_success': success,
            }
        )
    return results


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    track_name: str | None,
    model_name: str | None,
    base_url: str | None,
    missing_prediction_count: int = 0,
    unexpected_prediction_count: int = 0,
) -> dict[str, Any]:
    refinement_gains = [float(row.get('refinement_gain', 0.0)) for row in rows]
    input_distances = [float(row.get('edit_distance_input_to_reference', 0)) for row in rows]
    pred_distances = [float(row.get('edit_distance_prediction_to_reference', 0)) for row in rows]
    order_groups = _order_group_summary(rows)
    summary = {
        'track': track_name,
        'model': model_name,
        'base_url': base_url,
        'num_rows': len(rows),
        'missing_prediction_count': missing_prediction_count,
        'unexpected_prediction_count': unexpected_prediction_count,
        'recipe_success_rate': _rate(rows, 'recipe_success'),
        'status_accuracy': _rate(rows, 'status_match'),
        'exact_text_match_rate': _rate(rows, 'text_exact_match'),
        'canonical_text_match_rate': _rate(rows, 'text_canonical_match'),
        'avg_refinement_gain': _safe_mean(refinement_gains),
        'median_refinement_gain': _safe_median(refinement_gains),
        'avg_edit_distance_input_to_reference': _safe_mean(input_distances),
        'avg_edit_distance_prediction_to_reference': _safe_mean(pred_distances),
        'num_order_groups': len(order_groups),
        'order_consistent_success_rate': _rate(order_groups, 'order_consistent_success') if order_groups else 0.0,
        'order_group_success_rate': _rate(order_groups, 'order_consistent_success') if order_groups else 0.0,
        'by_operator': _slice_summary(rows, 'operator'),
        'by_domain': _slice_summary(rows, 'domain'),
        'by_source_domain': _slice_summary(rows, 'source_domain'),
        'by_reference_status': _slice_summary(rows, 'reference_status'),
    }
    return summary


def _build_instance_summary(
    instance_rows: list[dict[str, Any]],
    *,
    track_name: str | None,
    model_name: str | None,
    base_url: str | None,
    variant_rows: list[dict[str, Any]],
    order_group_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mean_rg_values = [float(row.get('mean_rg', 0.0)) for row in instance_rows]
    summary = {
        'track': track_name,
        'model': model_name,
        'base_url': base_url,
        'num_instances': len(instance_rows),
        'num_variant_predictions': len(variant_rows),
        'mean_rs': _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows]),
        'rs_at_k': _rate(instance_rows, 'rs_at_k'),
        'mean_rg': _safe_mean(mean_rg_values),
        'median_rg': _safe_median(mean_rg_values),
        'by_operator': _instance_slice_summary(instance_rows, 'operator'),
        'by_domain': _instance_slice_summary(instance_rows, 'domain'),
        'by_source_domain': _instance_slice_summary(instance_rows, 'source_domain'),
        'by_reference_status': _instance_slice_summary(instance_rows, 'reference_status'),
        'per_prompt_variant': _slice_summary(variant_rows, 'prompt_variant_index'),
    }
    if order_group_rows:
        summary['ocs'] = _rate(order_group_rows, 'ocs')
        summary['ocs_at_k'] = _rate(order_group_rows, 'ocs_at_k')
        summary['rs_front'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'front'])
        summary['rs_middle'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'middle'])
        summary['rs_end'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'end'])
        summary['num_order_groups'] = len(order_group_rows)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(_summary_report_text(summary), flush=True)


def _summary_report_text(summary: dict[str, Any]) -> str:
    track = str(summary.get('track') or 'unknown')
    model = str(summary.get('model') or '')
    prefix = f'[summary] track={track}'
    if model:
        prefix += f' model={model}'

    core_parts = []
    if 'num_instances' in summary:
        core_parts.append(f"num_instances={summary.get('num_instances')}")
    if 'mean_rs' in summary:
        core_parts.append(f"mean_rs={float(summary.get('mean_rs', 0.0)):.4f}")
    if 'rs_at_k' in summary:
        core_parts.append(f"rs_at_k={float(summary.get('rs_at_k', 0.0)):.4f}")
    if 'mean_rg' in summary:
        core_parts.append(f"mean_rg={float(summary.get('mean_rg', 0.0)):.4f}")
    if 'ocs' in summary:
        core_parts.append(f"ocs={float(summary.get('ocs', 0.0)):.4f}")
    if 'ocs_at_k' in summary:
        core_parts.append(f"ocs_at_k={float(summary.get('ocs_at_k', 0.0)):.4f}")
    if 'rs_front' in summary:
        core_parts.append(f"rs_front={float(summary.get('rs_front', 0.0)):.4f}")
    if 'rs_middle' in summary:
        core_parts.append(f"rs_middle={float(summary.get('rs_middle', 0.0)):.4f}")
    if 'rs_end' in summary:
        core_parts.append(f"rs_end={float(summary.get('rs_end', 0.0)):.4f}")
    return prefix + ' ' + ' '.join(core_parts)


def _paper_metrics_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        'track': summary.get('track'),
        'model': summary.get('model'),
        'num_instances': summary.get('num_instances'),
        'mean_rs': summary.get('mean_rs'),
        'rs_at_k': summary.get('rs_at_k'),
        'mean_rg': summary.get('mean_rg'),
    }
    if 'ocs' in summary:
        payload['ocs'] = summary.get('ocs')
    if 'ocs_at_k' in summary:
        payload['ocs_at_k'] = summary.get('ocs_at_k')
    if 'rs_front' in summary:
        payload['rs_front'] = summary.get('rs_front')
    if 'rs_middle' in summary:
        payload['rs_middle'] = summary.get('rs_middle')
    if 'rs_end' in summary:
        payload['rs_end'] = summary.get('rs_end')
    return payload


def _aggregate_instance_metrics(benchmark_row: dict[str, Any], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = _base_inference_row(benchmark_row)
    recipe_success_values = [1.0 if bool(row.get('recipe_success')) else 0.0 for row in variant_rows]
    refinement_gain_values = [float(row.get('refinement_gain', 0.0)) for row in variant_rows]
    variant_metric_summaries = []
    recipe_success_by_index: dict[int, bool] = {}
    for row in variant_rows:
        index = int(row.get('prompt_variant_index', 0) or 0)
        recipe_success_by_index[index] = bool(row.get('recipe_success'))
        variant_metric_summaries.append(
            {
                'prompt_variant_index': index,
                'prompt_style_id': row.get('prompt_style_id'),
                'prompt_style_label': row.get('prompt_style_label'),
                'recipe_success': bool(row.get('recipe_success')),
                'status_match': bool(row.get('status_match')),
                'text_exact_match': bool(row.get('text_exact_match')),
                'refinement_gain': float(row.get('refinement_gain', 0.0)),
                'prediction_error': row.get('prediction_error'),
            }
        )

    instance_row = {
        **base,
        'num_prompt_variants': len(variant_rows),
        'mean_rs': _safe_mean(recipe_success_values),
        'rs_at_k': any(recipe_success_values),
        'mean_rg': _safe_mean(refinement_gain_values),
        'prompt_variant_metrics': variant_metric_summaries,
        'recipe_success_prompt0': bool(recipe_success_by_index.get(0, False)),
    }
    return instance_row


def _build_order_group_rows(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in instance_rows:
        group_id = row.get('order_group_instance_id')
        if group_id:
            grouped.setdefault(str(group_id), []).append(row)

    results = []
    for group_id in sorted(grouped):
        bucket = grouped[group_id]
        results.append(
            {
                'order_group_instance_id': group_id,
                'slot_count': len(bucket),
                'ocs': all(bool(row.get('recipe_success_prompt0')) for row in bucket),
                'ocs_at_k': all(bool(row.get('rs_at_k')) for row in bucket),
            }
        )
    return results


def _track_name_from_path(path: Path) -> str:
    stem = path.stem
    return stem if stem else 'unknown'


def _predict(args: argparse.Namespace) -> None:
    eval_path = (ROOT / args.eval_path).resolve()
    output_path = (ROOT / args.output_path).resolve()
    output_dir = output_path.parent
    prompt_cfg = _load_yaml((ROOT / args.prompt_config).resolve())
    system_prompt = _default_system_prompt(prompt_cfg)
    schema_hint = _json_schema_hint(prompt_cfg)
    rows = _read_jsonl(eval_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    skipped_for_input_length = 0
    if args.max_input_chars > 0:
        filtered_rows = []
        for row in rows:
            input_length_chars = row.get('input_length_chars')
            if not isinstance(input_length_chars, int):
                input_length_chars = len(str(row.get('input_text', '')))
            if int(input_length_chars) > args.max_input_chars:
                skipped_for_input_length += 1
                continue
            filtered_rows.append(row)
        rows = filtered_rows

    base_url = resolve_base_url(args.base_url)
    api_key = _resolved_api_key(args.api_key, base_url)
    model = resolve_model(args.model)
    client = build_client(api_key=api_key, base_url=base_url)

    existing_rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for row in _read_jsonl(output_path):
            instance_id = str(row.get('instance_id') or '')
            if instance_id:
                existing_rows_by_id[instance_id] = row

    output_rows = []
    started = time.time()
    total_rows = len(rows)
    new_count = 0
    print(
        f'start predict track={_track_name_from_path(eval_path)} model={model} '
        f'num_rows={total_rows} skipped_input_too_long={skipped_for_input_length} '
        f'progress_every={args.progress_every} resume={bool(args.resume)} '
        f'max_input_chars={args.max_input_chars}',
        flush=True,
    )
    for index, row in enumerate(rows, start=1):
        instance_id = str(row.get('instance_id') or '')
        selected_prompt_variant_indices = _parse_prompt_variant_indices(
            args.prompt_variant_indices if args.prompt_variant_indices is not None else str(args.prompt_variant_index),
            row,
        )
        existing_row = existing_rows_by_id.get(instance_id, {})
        existing_variant_predictions = _existing_variant_prediction_map(existing_row)
        variant_predictions = dict(existing_variant_predictions)

        missing_indices = [
            prompt_variant_index
            for prompt_variant_index in selected_prompt_variant_indices
            if prompt_variant_index not in existing_variant_predictions
        ]

        for prompt_variant_index in missing_indices:
            prompt_variant = _select_prompt_variant(row, prompt_variant_index)
            user_requirement = str(prompt_variant.get('user_requirement') or '').strip()
            user_prompt = _render_user_prompt(row, user_requirement, schema_hint)

            prediction_payload = None
            prediction_error = None
            response_text = ''
            usage_payload: dict[str, Any] = {}
            for attempt in range(1, args.max_retries + 2):
                try:
                    response_text, usage_payload = _chat_completion(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    prediction_payload, prediction_error = _extract_prediction_payload(response_text)
                    if prediction_error is None:
                        break
                except Exception as exc:  # pragma: no cover - exercised in CLI usage
                    prediction_error = f'request_error: {exc}'
                if attempt <= args.max_retries:
                    time.sleep(args.retry_sleep_seconds)

            predicted_status = ''
            predicted_clean_text = ''
            if isinstance(prediction_payload, dict):
                predicted_status, predicted_clean_text = _extract_prediction_fields(
                    {'parsed_response': prediction_payload},
                    explicit_status_field=None,
                    explicit_text_field=None,
                )

            variant_predictions[prompt_variant_index] = {
                'prompt_variant_index': prompt_variant_index,
                'prompt_style_id': str(prompt_variant.get('style_id') or ''),
                'prompt_style_label': str(prompt_variant.get('style_label') or ''),
                'user_requirement': user_requirement,
                'request_model': model,
                'request_base_url': base_url,
                'raw_response': response_text,
                'parsed_response': prediction_payload,
                'prediction_error': prediction_error,
                'prediction_valid_json': prediction_error is None,
                'response_usage': usage_payload,
                'predicted_status': predicted_status,
                'predicted_clean_text': predicted_clean_text,
            }
            new_count += 1

        inference_row = {
            **_base_inference_row(row),
            'request_model': model,
            'request_base_url': base_url,
            'selected_prompt_variant_indices': selected_prompt_variant_indices,
            'variant_predictions': [variant_predictions[key] for key in sorted(variant_predictions)],
        }
        output_rows.append(inference_row)

        if index % args.progress_every == 0 or index == total_rows:
            elapsed = time.time() - started
            print(
                f'progress predict row={index}/{total_rows} '
                f'new_variant_predictions={new_count} elapsed_sec={elapsed:.1f}',
                flush=True,
            )

    _write_jsonl(output_path, output_rows)
    summary = {
        'track': _track_name_from_path(eval_path),
        'model': model,
        'base_url': base_url,
        'num_instances': len(output_rows),
        'num_new_variant_predictions': new_count,
        'num_skipped_for_input_length': skipped_for_input_length,
        'max_input_chars': args.max_input_chars,
        'requested_prompt_variant_indices': (
            args.prompt_variant_indices if args.prompt_variant_indices is not None else str(args.prompt_variant_index)
        ),
    }
    _write_json(output_dir / 'summary.json', summary)
    print(f'wrote predictions -> {output_path}', flush=True)
    print(f'wrote summary -> {output_dir / "summary.json"}', flush=True)


def _score_rows_from_inference_row(
    inference_row: dict[str, Any],
    *,
    explicit_status_field: str | None,
    explicit_text_field: str | None,
) -> list[dict[str, Any]]:
    scored_rows = []
    variant_predictions = _normalize_variant_predictions(inference_row)
    if variant_predictions:
        for variant_prediction in variant_predictions:
            predicted_status, predicted_clean_text = _extract_prediction_fields(
                variant_prediction,
                explicit_status_field=explicit_status_field,
                explicit_text_field=explicit_text_field,
            )
            scored_row = _base_score_row(inference_row, predicted_status, predicted_clean_text)
            scored_row.update(
                {
                    'prompt_variant_index': int(variant_prediction.get('prompt_variant_index', 0) or 0),
                    'prompt_style_id': variant_prediction.get('prompt_style_id'),
                    'prompt_style_label': variant_prediction.get('prompt_style_label'),
                    'user_requirement': variant_prediction.get('user_requirement'),
                    'request_model': variant_prediction.get('request_model'),
                    'request_base_url': variant_prediction.get('request_base_url'),
                    'raw_response': variant_prediction.get('raw_response'),
                    'parsed_response': variant_prediction.get('parsed_response'),
                    'prediction_error': variant_prediction.get('prediction_error'),
                    'prediction_valid_json': variant_prediction.get('prediction_valid_json'),
                    'response_usage': variant_prediction.get('response_usage'),
                }
            )
            scored_rows.append(scored_row)
        return scored_rows

    predicted_status, predicted_clean_text = _extract_prediction_fields(
        inference_row,
        explicit_status_field=explicit_status_field,
        explicit_text_field=explicit_text_field,
    )
    scored_row = _base_score_row(inference_row, predicted_status, predicted_clean_text)
    passthrough = {
        key: value
        for key, value in inference_row.items()
        if key not in scored_row
    }
    scored_row.update(passthrough)
    scored_rows.append(scored_row)
    return scored_rows


def _score(args: argparse.Namespace) -> None:
    predictions_path = (ROOT / args.predictions_path).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = _read_jsonl(predictions_path)

    existing_instance_rows_by_id: dict[str, dict[str, Any]] = {}
    existing_variant_rows_by_id: dict[str, list[dict[str, Any]]] = {}
    instance_metrics_path = output_dir / 'instance_metrics.jsonl'
    scored_variant_predictions_path = output_dir / 'scored_variant_predictions.jsonl'
    if args.resume:
        if instance_metrics_path.exists():
            for row in _read_jsonl(instance_metrics_path):
                instance_id = str(row.get('instance_id') or '')
                if instance_id:
                    existing_instance_rows_by_id[instance_id] = row
        if scored_variant_predictions_path.exists():
            existing_variant_rows_by_id = _group_rows_by_instance_id(_read_jsonl(scored_variant_predictions_path))

    scored_variant_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    reused_instance_count = 0
    new_instance_count = 0
    started = time.time()
    if args.benchmark_path:
        benchmark_rows = _read_jsonl((ROOT / args.benchmark_path).resolve())
        print(
            f'start score track={_track_name_from_path((ROOT / args.benchmark_path).resolve())} '
            f'num_instances={len(benchmark_rows)} progress_every={args.progress_every} resume={bool(args.resume)}',
            flush=True,
        )
        benchmark_map = {str(row.get('instance_id') or ''): row for row in benchmark_rows if row.get('instance_id')}
        unexpected_prediction_count = 0
        seen_prediction_ids = set()
        missing_prediction_count = 0

        for prediction_row in prediction_rows:
            prediction_instance_id = str(prediction_row.get(args.prediction_instance_field) or '')
            if prediction_instance_id:
                seen_prediction_ids.add(prediction_instance_id)
                if prediction_instance_id not in benchmark_map:
                    unexpected_prediction_count += 1

        for benchmark_row in benchmark_rows:
            instance_id = str(benchmark_row.get('instance_id') or '')
            if args.resume and instance_id in existing_instance_rows_by_id:
                instance_rows.append(existing_instance_rows_by_id[instance_id])
                scored_variant_rows.extend(existing_variant_rows_by_id.get(instance_id, []))
                reused_instance_count += 1
                current_count = reused_instance_count + new_instance_count
                if current_count % args.progress_every == 0 or current_count == len(benchmark_rows):
                    elapsed = time.time() - started
                    print(
                        f'progress score instance={current_count}/{len(benchmark_rows)} '
                        f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                        f'elapsed_sec={elapsed:.1f}',
                        flush=True,
                    )
                continue
            prediction_row = next(
                (row for row in prediction_rows if str(row.get(args.prediction_instance_field) or '') == instance_id),
                None,
            )
            if prediction_row is None:
                missing_prediction_count += 1
                instance_row = {
                    **_base_inference_row(benchmark_row),
                    'num_prompt_variants': 0,
                    'mean_rs': 0.0,
                    'rs_at_k': False,
                    'mean_rg': 0.0,
                    'prompt_variant_metrics': [],
                    'recipe_success_prompt0': False,
                    'missing_prediction': True,
                }
                instance_rows.append(instance_row)
                new_instance_count += 1
                current_count = reused_instance_count + new_instance_count
                if current_count % args.progress_every == 0 or current_count == len(benchmark_rows):
                    elapsed = time.time() - started
                    print(
                        f'progress score instance={current_count}/{len(benchmark_rows)} '
                        f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                        f'elapsed_sec={elapsed:.1f}',
                        flush=True,
                    )
                continue

            merged_row = {**benchmark_row, **prediction_row}
            instance_variant_rows = _score_rows_from_inference_row(
                merged_row,
                explicit_status_field=args.prediction_status_field,
                explicit_text_field=args.prediction_text_field,
            )
            scored_variant_rows.extend(instance_variant_rows)
            instance_row = _aggregate_instance_metrics(benchmark_row, instance_variant_rows)
            instance_row['missing_prediction'] = False
            instance_rows.append(instance_row)
            new_instance_count += 1
            current_count = reused_instance_count + new_instance_count
            if current_count % args.progress_every == 0 or current_count == len(benchmark_rows):
                elapsed = time.time() - started
                print(
                    f'progress score instance={current_count}/{len(benchmark_rows)} '
                    f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                    f'elapsed_sec={elapsed:.1f}',
                    flush=True,
                )

        order_group_rows = _build_order_group_rows(instance_rows)
        summary = _build_instance_summary(
            instance_rows,
            track_name=_track_name_from_path((ROOT / args.benchmark_path).resolve()),
            model_name=args.model,
            base_url=args.base_url,
            variant_rows=scored_variant_rows,
            order_group_rows=order_group_rows,
        )
        summary['missing_prediction_count'] = missing_prediction_count
        summary['unexpected_prediction_count'] = unexpected_prediction_count
    else:
        total_rows = len(prediction_rows)
        print(
            f'start score track={_track_name_from_path(predictions_path)} '
            f'num_instances={total_rows} progress_every={args.progress_every} resume={bool(args.resume)}',
            flush=True,
        )
        for index, prediction_row in enumerate(prediction_rows, start=1):
            instance_id = str(prediction_row.get(args.prediction_instance_field) or prediction_row.get('instance_id') or '')
            if args.resume and instance_id in existing_instance_rows_by_id:
                instance_rows.append(existing_instance_rows_by_id[instance_id])
                scored_variant_rows.extend(existing_variant_rows_by_id.get(instance_id, []))
                reused_instance_count += 1
                current_count = reused_instance_count + new_instance_count
                if current_count % args.progress_every == 0 or current_count == total_rows:
                    elapsed = time.time() - started
                    print(
                        f'progress score instance={current_count}/{total_rows} '
                        f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                        f'elapsed_sec={elapsed:.1f}',
                        flush=True,
                    )
                continue
            instance_variant_rows = _score_rows_from_inference_row(
                prediction_row,
                explicit_status_field=args.prediction_status_field,
                explicit_text_field=args.prediction_text_field,
            )
            scored_variant_rows.extend(instance_variant_rows)
            instance_rows.append(_aggregate_instance_metrics(prediction_row, instance_variant_rows))
            new_instance_count += 1
            current_count = reused_instance_count + new_instance_count
            if current_count % args.progress_every == 0 or current_count == total_rows:
                elapsed = time.time() - started
                print(
                    f'progress score instance={current_count}/{total_rows} '
                    f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                    f'elapsed_sec={elapsed:.1f}',
                    flush=True,
                )
        order_group_rows = _build_order_group_rows(instance_rows)
        summary = _build_instance_summary(
            instance_rows,
            track_name=_track_name_from_path(predictions_path),
            model_name=args.model,
            base_url=args.base_url,
            variant_rows=scored_variant_rows,
            order_group_rows=order_group_rows,
        )

    _write_jsonl(output_dir / 'scored_variant_predictions.jsonl', scored_variant_rows)
    _write_jsonl(output_dir / 'instance_metrics.jsonl', instance_rows)
    _write_json(output_dir / 'summary.json', summary)
    _write_json(output_dir / 'paper_metrics.json', _paper_metrics_payload(summary))
    _write_text(output_dir / 'report.txt', _summary_report_text(summary))
    if args.write_csv_slices:
        _write_csv(output_dir / 'by_operator.csv', summary['by_operator'])
        _write_csv(output_dir / 'by_domain.csv', summary['by_domain'])
        _write_csv(output_dir / 'by_source_domain.csv', summary['by_source_domain'])
        _write_csv(output_dir / 'by_reference_status.csv', summary['by_reference_status'])
        if order_group_rows:
            _write_csv(output_dir / 'order_groups.csv', order_group_rows)
        _write_csv(output_dir / 'per_prompt_variant.csv', summary.get('per_prompt_variant', []))
    print(f'wrote scored variant predictions -> {output_dir / "scored_variant_predictions.jsonl"}', flush=True)
    print(f'wrote instance metrics -> {output_dir / "instance_metrics.jsonl"}', flush=True)
    print(f'wrote summary -> {output_dir / "summary.json"}', flush=True)
    print(f'wrote paper metrics -> {output_dir / "paper_metrics.json"}', flush=True)
    print(f'wrote report -> {output_dir / "report.txt"}', flush=True)
    _print_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run or score CDR-Bench evaluations. Atomic track is the initial target, but the scorer also works for main and order tracks.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    predict_parser = subparsers.add_parser('predict', help='Call an OpenAI-compatible API on eval-ready JSONL and save raw predictions.')
    predict_parser.add_argument('--eval-path', required=True, help='Eval-ready JSONL, e.g. data/benchmark/atomic_ops/atomic_ops.jsonl')
    predict_parser.add_argument('--output-path', required=True, help='Prediction JSONL output path.')
    predict_parser.add_argument('--prompt-config', default=str(DEFAULT_PROMPT_CONFIG.relative_to(ROOT)))
    predict_parser.add_argument('--model', default=None)
    predict_parser.add_argument('--base-url', default=None)
    predict_parser.add_argument('--api-key', default=None)
    predict_parser.add_argument('--temperature', type=float, default=0.0)
    predict_parser.add_argument('--max-tokens', type=int, default=4096)
    predict_parser.add_argument('--prompt-variant-index', type=int, default=0)
    predict_parser.add_argument(
        '--prompt-variant-indices',
        default=None,
        help='Comma-separated prompt variant indices to run, or "all". Overrides --prompt-variant-index when set.',
    )
    predict_parser.add_argument('--max-samples', type=int, default=0)
    predict_parser.add_argument(
        '--max-input-chars',
        type=int,
        default=0,
        help='Skip inference for samples whose input_length_chars exceeds this threshold. 0 disables the filter.',
    )
    predict_parser.add_argument('--max-retries', type=int, default=2)
    predict_parser.add_argument('--retry-sleep-seconds', type=float, default=2.0)
    predict_parser.add_argument('--progress-every', type=int, default=DEFAULT_PROGRESS_EVERY)
    predict_parser.add_argument('--resume', action='store_true')
    predict_parser.set_defaults(func=_predict)

    infer_parser = subparsers.add_parser('infer', help='Alias for predict; saves raw model outputs for later metric computation.')
    for action in predict_parser._actions[1:]:
        infer_parser._add_action(action)
    infer_parser.set_defaults(func=_predict)

    score_parser = subparsers.add_parser('score', help='Score existing prediction JSONL against benchmark references.')
    score_parser.add_argument('--predictions-path', required=True)
    score_parser.add_argument('--benchmark-path', default=None)
    score_parser.add_argument('--output-dir', required=True)
    score_parser.add_argument('--prediction-instance-field', default='instance_id')
    score_parser.add_argument('--prediction-status-field', default=None)
    score_parser.add_argument('--prediction-text-field', default=None)
    score_parser.add_argument('--model', default=None)
    score_parser.add_argument('--base-url', default=None)
    score_parser.add_argument('--progress-every', type=int, default=DEFAULT_PROGRESS_EVERY)
    score_parser.add_argument('--resume', action='store_true')
    score_parser.add_argument(
        '--write-csv-slices',
        action='store_true',
        help='Write slice-analysis CSV files such as by_operator.csv and by_domain.csv. Disabled by default.',
    )
    score_parser.set_defaults(func=_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
