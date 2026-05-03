#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from cdrbench.eval.metrics import compute_recipe_metrics


ROOT = Path(__file__).resolve().parents[3]
DOMAIN_METADATA = {
    'web': {'label': 'Web Refinement', 'abbr': 'WR', 'order': 0},
    'arxiv': {'label': 'LaTeX Refinement', 'abbr': 'LR', 'order': 1},
    'knowledge_base': {'label': 'RAG Preparation', 'abbr': 'RP', 'order': 2},
    'pii': {'label': 'Privacy Redaction', 'abbr': 'PR', 'order': 3},
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        handle.write(text)
        if not text.endswith('\n'):
            handle.write('\n')
    tmp_path.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with tmp_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _load_domain_metadata() -> dict[str, dict[str, Any]]:
    metadata = {key: dict(value) for key, value in DOMAIN_METADATA.items()}
    path = ROOT / 'configs' / 'domains.yaml'
    if not path.exists():
        return metadata
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    domains = payload.get('domains') if isinstance(payload, dict) else None
    if not isinstance(domains, dict):
        return metadata
    for key, value in domains.items():
        if not isinstance(key, str):
            continue
        description = value.get('description') if isinstance(value, dict) else None
        if key not in metadata:
            metadata[key] = {
                'label': description.strip() if isinstance(description, str) and description.strip() else key.replace('_', ' ').title(),
                'abbr': key.upper(),
                'order': 100 + len(metadata),
            }
    return metadata


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return _safe_mean([1.0 if bool(row.get(key)) else 0.0 for row in rows])


def _mean_optional(values: list[Any]) -> float:
    normalized: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            continue
    return _safe_mean(normalized)


def _is_format_instability_error(error_text: Any) -> bool:
    if error_text is None:
        return False
    text = str(error_text)
    return text == 'empty_response' or text.startswith('json_parse_error:')


def _normalize_variant_predictions(row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = row.get('variant_predictions')
    if not isinstance(variants, list):
        return []
    normalized = [variant for variant in variants if isinstance(variant, dict)]
    return sorted(normalized, key=lambda item: int(item.get('prompt_variant_index', 0) or 0))


def _group_rows_by_instance_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        instance_id = str(row.get('instance_id') or '')
        if instance_id:
            grouped.setdefault(instance_id, []).append(row)
    return grouped


def _first_non_empty_str(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _track_name_from_predictions_path(path: Path) -> str:
    if path.stem == 'predictions' and path.parent.name:
        return path.parent.name
    return path.stem or 'unknown'


def _infer_labels(prediction_rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for prediction_row in prediction_rows:
        model = _first_non_empty_str(prediction_row.get('request_model'))
        base_url = _first_non_empty_str(prediction_row.get('request_base_url'))
        if model or base_url:
            return model, base_url
        for variant_prediction in _normalize_variant_predictions(prediction_row):
            model = _first_non_empty_str(variant_prediction.get('request_model'))
            base_url = _first_non_empty_str(variant_prediction.get('request_base_url'))
            if model or base_url:
                return model, base_url
    return None, None


def _base_identity(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
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
        'recipe_id',
        'recipe_variant_id',
        'recipe_type',
    ]
    return {key: row[key] for key in keys if key in row}


def _score_variant(prediction_row: dict[str, Any], variant_prediction: dict[str, Any]) -> dict[str, Any]:
    prediction_error = variant_prediction.get('prediction_error')
    valid_prediction = prediction_error is None
    predicted_status = '' if variant_prediction.get('predicted_status') is None else str(variant_prediction.get('predicted_status'))
    predicted_clean_text = '' if variant_prediction.get('predicted_clean_text') is None else str(variant_prediction.get('predicted_clean_text'))
    if not predicted_status and 'parsed_response' in variant_prediction and isinstance(variant_prediction['parsed_response'], dict):
        payload = variant_prediction['parsed_response']
        predicted_status = '' if payload.get('status') is None else str(payload.get('status'))
        predicted_clean_text = '' if payload.get('clean_text') is None else str(payload.get('clean_text'))

    scored = _base_identity(prediction_row)
    scored['predicted_status'] = predicted_status
    scored['predicted_clean_text'] = predicted_clean_text
    if valid_prediction:
        scored.update(
            compute_recipe_metrics(
                input_text=prediction_row.get('input_text', ''),
                reference_status=prediction_row.get('reference_status', ''),
                reference_text=prediction_row.get('reference_text', ''),
                predicted_status=predicted_status,
                predicted_clean_text=predicted_clean_text,
            )
        )
    else:
        scored.update(
            {
                'normalized_reference_status': '' if prediction_row.get('reference_status') is None else str(prediction_row.get('reference_status')).strip().upper(),
                'normalized_predicted_status': '' if predicted_status is None else str(predicted_status).strip().upper(),
                'status_match': False,
                'text_exact_match': False,
                'recipe_success': False,
                'text_match': False,
                'edit_distance_input_to_reference': None,
                'edit_distance_prediction_to_reference': None,
                'refinement_gain': None,
            }
        )
    scored.update(
        {
            'prompt_variant_index': int(variant_prediction.get('prompt_variant_index', 0) or 0),
            'prompt_style_id': variant_prediction.get('prompt_style_id'),
            'prompt_style_label': variant_prediction.get('prompt_style_label'),
            'user_requirement': variant_prediction.get('user_requirement'),
            'request_model': variant_prediction.get('request_model'),
            'request_base_url': variant_prediction.get('request_base_url'),
            'raw_response': variant_prediction.get('raw_response'),
            'parsed_response': variant_prediction.get('parsed_response'),
            'prediction_error': prediction_error,
            'prediction_valid_json': variant_prediction.get('prediction_valid_json'),
            'retry_attempted': bool(variant_prediction.get('retry_attempted')),
            'format_instability_error': _is_format_instability_error(prediction_error),
            'response_usage': variant_prediction.get('response_usage'),
        }
    )
    return scored


def _score_prediction_row(prediction_row: dict[str, Any]) -> list[dict[str, Any]]:
    variants = _normalize_variant_predictions(prediction_row)
    if not variants:
        variants = [prediction_row]
    return [_score_variant(prediction_row, variant) for variant in variants]


def _aggregate_instance_metrics(prediction_row: dict[str, Any], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recipe_success_values = [1.0 if bool(row.get('recipe_success')) else 0.0 for row in variant_rows]
    refinement_gain_values = [row.get('refinement_gain') for row in variant_rows]
    prompt_variant_metrics = []
    recipe_success_by_index: dict[int, bool] = {}
    for row in variant_rows:
        index = int(row.get('prompt_variant_index', 0) or 0)
        recipe_success_by_index[index] = bool(row.get('recipe_success'))
        prompt_variant_metrics.append(
            {
                'prompt_variant_index': index,
                'prompt_style_id': row.get('prompt_style_id'),
                'prompt_style_label': row.get('prompt_style_label'),
                'recipe_success': bool(row.get('recipe_success')),
                'status_match': bool(row.get('status_match')),
                'text_exact_match': bool(row.get('text_exact_match')),
                'refinement_gain': None if row.get('refinement_gain') is None else float(row.get('refinement_gain', 0.0)),
                'prediction_error': row.get('prediction_error'),
                'format_instability_error': bool(row.get('format_instability_error')),
            }
        )

    instance_row = _base_identity(prediction_row)
    instance_row.update(
        {
            'num_prompt_variants': len(variant_rows),
            'mean_rs': _safe_mean(recipe_success_values),
            'rs_at_k': any(recipe_success_values),
            'mean_rg': _mean_optional(refinement_gain_values),
            'num_valid_rg_variants': sum(1 for value in refinement_gain_values if value is not None),
            'num_invalid_variants': sum(1 for row in variant_rows if row.get('prediction_error') is not None),
            'num_format_error_variants': sum(1 for row in variant_rows if bool(row.get('format_instability_error'))),
            'prompt_variant_metrics': prompt_variant_metrics,
            'recipe_success_prompt0': bool(recipe_success_by_index.get(0, False)),
        }
    )
    return instance_row


def _build_order_group_rows(instance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in instance_rows:
        group_id = row.get('order_group_instance_id')
        if group_id:
            grouped.setdefault(str(group_id), []).append(row)
    return [
        {
            'order_group_instance_id': group_id,
            'slot_count': len(bucket),
            'ocs': all(bool(row.get('recipe_success_prompt0')) for row in bucket),
            'ocs_at_k': all(bool(row.get('rs_at_k')) for row in bucket),
        }
        for group_id, bucket in sorted(grouped.items())
    ]


def _instance_slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)
    return [
        {
            key: value,
            'count': len(bucket),
            'mean_rs': _safe_mean([float(item.get('mean_rs', 0.0)) for item in bucket]),
            'rs_at_k': _rate(bucket, 'rs_at_k'),
            'mean_rg': _mean_optional([item.get('mean_rg') for item in bucket]),
        }
        for value, bucket in sorted(grouped.items())
    ]


def _attach_domain_metadata(rows: list[dict[str, Any]], *, key: str, metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        copied = dict(row)
        raw_value = str(row.get(key) or 'UNKNOWN')
        meta = metadata.get(
            raw_value,
            {
                'label': raw_value.replace('_', ' ').title(),
                'abbr': raw_value.upper(),
                'order': 999,
            },
        )
        copied['domain_label'] = str(meta.get('label') or raw_value)
        copied['domain_abbr'] = str(meta.get('abbr') or raw_value.upper())
        copied['domain_order'] = int(meta.get('order', 999) or 999)
        enriched.append(copied)
    return sorted(enriched, key=lambda row: (int(row.get('domain_order', 999) or 999), str(row.get(key) or '')))


def _variant_slice_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or 'UNKNOWN')
        grouped.setdefault(value, []).append(row)
    return [
        {
            key: value,
            'count': len(bucket),
            'recipe_success_rate': _rate(bucket, 'recipe_success'),
            'status_accuracy': _rate(bucket, 'status_match'),
            'exact_text_match_rate': _rate(bucket, 'text_exact_match'),
            'avg_refinement_gain': _mean_optional([row.get('refinement_gain') for row in bucket]),
            'valid_json_rate': _rate(bucket, 'prediction_valid_json'),
            'format_error_rate': _rate(bucket, 'format_instability_error'),
        }
        for value, bucket in sorted(grouped.items())
    ]


def _summary_report_text(summary: dict[str, Any]) -> str:
    parts = [f"[summary] track={summary.get('track') or 'unknown'}"]
    if summary.get('model'):
        parts.append(f"model={summary['model']}")
    parts.append(f"num_instances={summary.get('num_instances', 0)}")
    parts.append(f"mean_rs={float(summary.get('mean_rs', 0.0)):.4f}")
    parts.append(f"rs_at_k={float(summary.get('rs_at_k', 0.0)):.4f}")
    parts.append(f"mean_rg={float(summary.get('mean_rg', 0.0)):.4f}")
    parts.append(f"valid_json_rate={float(summary.get('valid_json_rate', 0.0)):.4f}")
    parts.append(f"empty_response_rate={float(summary.get('empty_response_rate', 0.0)):.4f}")
    parts.append(f"format_error_rate={float(summary.get('format_error_rate', 0.0)):.4f}")
    if 'ocs' in summary:
        parts.append(f"ocs={float(summary.get('ocs', 0.0)):.4f}")
        parts.append(f"ocs_at_k={float(summary.get('ocs_at_k', 0.0)):.4f}")
    return ' '.join(parts)


def _paper_metrics_payload(summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        'track': summary.get('track'),
        'model': summary.get('model'),
        'num_instances': summary.get('num_instances'),
        'mean_rs': summary.get('mean_rs'),
        'rs_at_k': summary.get('rs_at_k'),
        'mean_rg': summary.get('mean_rg'),
        'valid_json_rate': summary.get('valid_json_rate'),
        'empty_response_rate': summary.get('empty_response_rate'),
        'format_error_rate': summary.get('format_error_rate'),
    }
    for key in ('ocs', 'ocs_at_k', 'rs_front', 'rs_middle', 'rs_end'):
        if key in summary:
            payload[key] = summary.get(key)
    return payload


def _build_summary(
    predictions_path: Path,
    instance_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    order_group_rows: list[dict[str, Any]],
    model_name: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    domain_metadata = _load_domain_metadata()
    mean_rg_values = [row.get('mean_rg') for row in instance_rows if int(row.get('num_valid_rg_variants', 0) or 0) > 0]
    by_domain = _attach_domain_metadata(_instance_slice_summary(instance_rows, 'domain'), key='domain', metadata=domain_metadata)
    by_source_domain = _instance_slice_summary(instance_rows, 'source_domain')
    summary = {
        'track': _track_name_from_predictions_path(predictions_path),
        'model': model_name,
        'base_url': base_url,
        'num_instances': len(instance_rows),
        'num_variant_predictions': len(variant_rows),
        'mean_rs': _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows]),
        'rs_at_k': _rate(instance_rows, 'rs_at_k'),
        'mean_rg': _mean_optional(mean_rg_values),
        'median_rg': _safe_median([float(value) for value in mean_rg_values if value is not None]),
        'valid_json_rate': _rate(variant_rows, 'prediction_valid_json'),
        'empty_response_rate': _safe_mean([1.0 if str(row.get('prediction_error') or '') == 'empty_response' else 0.0 for row in variant_rows]),
        'format_error_rate': _rate(variant_rows, 'format_instability_error'),
        'by_operator': _instance_slice_summary(instance_rows, 'operator'),
        'by_domain': by_domain,
        'by_source_domain': by_source_domain,
        'by_reference_status': _instance_slice_summary(instance_rows, 'reference_status'),
        'per_prompt_variant': _variant_slice_summary(variant_rows, 'prompt_variant_index'),
    }
    if order_group_rows:
        summary['ocs'] = _rate(order_group_rows, 'ocs')
        summary['ocs_at_k'] = _rate(order_group_rows, 'ocs_at_k')
        summary['rs_front'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'front'])
        summary['rs_middle'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'middle'])
        summary['rs_end'] = _safe_mean([float(row.get('mean_rs', 0.0)) for row in instance_rows if str(row.get('order_slot') or '') == 'end'])
        summary['num_order_groups'] = len(order_group_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Score CDR-Bench predictions and write reports.')
    parser.add_argument('--predictions-path', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--progress-every', type=int, default=20)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--write-csv-slices', action='store_true')
    args = parser.parse_args()

    predictions_path = Path(args.predictions_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else predictions_path.parent.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = _read_jsonl(predictions_path)
    inferred_model, inferred_base_url = _infer_labels(prediction_rows)

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
    total_rows = len(prediction_rows)
    started = time.time()
    print(
        f'start score track={_track_name_from_predictions_path(predictions_path)} '
        f'num_instances={total_rows} progress_every={args.progress_every} resume={bool(args.resume)}',
        flush=True,
    )

    for index, prediction_row in enumerate(prediction_rows, start=1):
        instance_id = str(prediction_row.get('instance_id') or '')
        if args.resume and instance_id in existing_instance_rows_by_id:
            instance_rows.append(existing_instance_rows_by_id[instance_id])
            scored_variant_rows.extend(existing_variant_rows_by_id.get(instance_id, []))
            reused_instance_count += 1
        else:
            instance_variant_rows = _score_prediction_row(prediction_row)
            scored_variant_rows.extend(instance_variant_rows)
            instance_rows.append(_aggregate_instance_metrics(prediction_row, instance_variant_rows))
            new_instance_count += 1

        if index % args.progress_every == 0 or index == total_rows:
            elapsed = time.time() - started
            print(
                f'progress score instance={index}/{total_rows} '
                f'reused_instances={reused_instance_count} new_instances={new_instance_count} '
                f'elapsed_sec={elapsed:.1f}',
                flush=True,
            )

    order_group_rows = _build_order_group_rows(instance_rows)
    summary = _build_summary(
        predictions_path=predictions_path,
        instance_rows=instance_rows,
        variant_rows=scored_variant_rows,
        order_group_rows=order_group_rows,
        model_name=inferred_model,
        base_url=inferred_base_url,
    )

    _write_jsonl(scored_variant_predictions_path, scored_variant_rows)
    _write_jsonl(instance_metrics_path, instance_rows)
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

    print(f'wrote scored variant predictions -> {scored_variant_predictions_path}', flush=True)
    print(f'wrote instance metrics -> {instance_metrics_path}', flush=True)
    print(f'wrote summary -> {output_dir / "summary.json"}', flush=True)
    print(f'wrote paper metrics -> {output_dir / "paper_metrics.json"}', flush=True)
    print(f'wrote report -> {output_dir / "report.txt"}', flush=True)
    print(_summary_report_text(summary), flush=True)


if __name__ == '__main__':
    main()
