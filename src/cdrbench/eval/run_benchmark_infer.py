#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from cdrbench.infer.openai_infer import make_api_infer, make_vllm_infer
from cdrbench.llm_utils import parse_json_response, resolve_api_key, resolve_base_url, resolve_model


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT_CONFIG = ROOT / 'configs' / 'recipe_prompting.yaml'
DEFAULT_PROGRESS_EVERY = 20
LOCAL_HOSTS = {'127.0.0.1', '0.0.0.0', '::1', 'localhost'}
FATAL_REQUEST_ERROR_PATTERNS = (
    re.compile(r'\b404\b'),
    re.compile(r'model[^a-z0-9]+not[^a-z0-9]+found', re.IGNORECASE),
    re.compile(r'not[^a-z0-9]+found', re.IGNORECASE),
    re.compile(r'no[^a-z0-9]+such[^a-z0-9]+model', re.IGNORECASE),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    tmp_path.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    tmp_path.replace(path)


def _write_progress_snapshot(output_path: Path, output_dir: Path, output_rows: list[dict[str, Any]], model: str, base_url: str, track_name: str) -> None:
    _write_jsonl(output_path, output_rows)
    _write_json(
        output_dir / 'summary.json',
        {
            'track': track_name,
            'model': model,
            'base_url': base_url,
            'num_instances': len(output_rows),
        },
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
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


def _resolved_api_key(explicit_api_key: str | None, base_url: str) -> str:
    if explicit_api_key:
        return explicit_api_key
    host = (urlparse(base_url).hostname or '').strip().lower()
    if host in LOCAL_HOSTS:
        return 'EMPTY'
    return resolve_api_key(None)


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
                f'prompt_variant_index={index} is out of range for instance_id={row.get("instance_id")} with available={available}'
            )
        requested.append(index)
    return sorted(set(requested))


def _select_prompt_variant(row: dict[str, Any], prompt_variant_index: int) -> dict[str, Any]:
    prompt_variants = row.get('prompt_variants')
    if isinstance(prompt_variants, list) and prompt_variants:
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


def _extract_prediction_payload(response_text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not response_text.strip():
        return None, 'empty_response'
    try:
        payload = parse_json_response(response_text)
    except Exception as exc:
        return None, f'json_parse_error: {exc}'
    if not isinstance(payload, dict):
        return None, 'json_parse_error: response is not a JSON object'
    return payload, None


def _is_retryable_prediction_error(prediction_error: str | None) -> bool:
    if not prediction_error:
        return False
    return prediction_error == 'empty_response' or prediction_error.startswith('json_parse_error:')


def _extract_prediction_fields(prediction_payload: dict[str, Any] | None) -> tuple[str, str]:
    if not isinstance(prediction_payload, dict):
        return '', ''
    status_value = prediction_payload.get('predicted_status', prediction_payload.get('status'))
    text_value = prediction_payload.get('predicted_clean_text', prediction_payload.get('clean_text', prediction_payload.get('text')))
    return ('' if status_value is None else str(status_value), '' if text_value is None else str(text_value))


def _existing_variant_prediction_map(row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    variants = row.get('variant_predictions')
    if not isinstance(variants, list):
        return {}
    mapping: dict[int, dict[str, Any]] = {}
    for variant in variants:
        if isinstance(variant, dict):
            mapping[int(variant.get('prompt_variant_index', 0) or 0)] = variant
    return mapping


def _row_input_length_chars(row: dict[str, Any]) -> int:
    value = row.get('input_length_chars')
    if isinstance(value, int):
        return value
    return len(str(row.get('input_text', '')))


def _build_infer_backend(args: argparse.Namespace, model: str, base_url: str, api_key: str) -> Any:
    common_kwargs = {
        'model': model,
        'api_base': base_url,
        'concurrency': max(1, int(args.concurrency)),
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'num_runs': 1,
        'max_retries': max(1, int(args.max_retries) + 1),
        'retry_delay': float(args.retry_sleep_seconds),
    }
    host = (urlparse(base_url).hostname or '').strip().lower()
    if host in LOCAL_HOSTS:
        return make_vllm_infer(**common_kwargs)
    return make_api_infer(**common_kwargs, api_key=api_key)


def _is_fatal_request_error(error_text: str | None) -> bool:
    if not error_text:
        return False
    return any(pattern.search(error_text) for pattern in FATAL_REQUEST_ERROR_PATTERNS)


def _log_prediction_issue(
    *,
    track_name: str,
    instance_id: str,
    prompt_variant_index: int,
    prediction_error: str | None,
    response_text: str,
) -> None:
    if not prediction_error:
        return
    response_preview = response_text.strip().replace('\n', ' ')
    if len(response_preview) > 240:
        response_preview = response_preview[:240] + '...'
    print(
        f'warn infer track={track_name} instance_id={instance_id or "UNKNOWN"} '
        f'prompt_variant_index={prompt_variant_index} error={prediction_error} '
        f'response_preview={response_preview!r}',
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CDR-Bench inference on eval-ready JSONL and save raw predictions.')
    parser.add_argument('--eval-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--prompt-config', default=str(DEFAULT_PROMPT_CONFIG.relative_to(ROOT)))
    parser.add_argument('--model', default=None)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=0)
    parser.add_argument('--prompt-variant-index', type=int, default=0)
    parser.add_argument('--prompt-variant-indices', default=None)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--max-input-chars', type=int, default=0)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--retry-sleep-seconds', type=float, default=2.0)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--progress-every', type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    eval_path = Path(args.eval_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_dir = output_path.parent
    prompt_cfg = _load_yaml((ROOT / args.prompt_config).resolve())
    system_prompt = _default_system_prompt(prompt_cfg)
    schema_hint = _json_schema_hint(prompt_cfg)
    rows = _read_jsonl(eval_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if args.max_input_chars > 0:
        rows = [row for row in rows if _row_input_length_chars(row) <= args.max_input_chars]

    base_url = resolve_base_url(args.base_url)
    api_key = _resolved_api_key(args.api_key, base_url)
    model = resolve_model(args.model)
    infer_backend = _build_infer_backend(args, model, base_url, api_key)

    existing_rows_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        for row in _read_jsonl(output_path):
            instance_id = str(row.get('instance_id') or '')
            if instance_id:
                existing_rows_by_id[instance_id] = row

    output_rows = []
    variant_jobs: list[tuple[int, dict[str, Any], int, dict[str, Any], dict[int, dict[str, Any]], list[int]]] = []
    started = time.time()
    for row_index, row in enumerate(rows):
        instance_id = str(row.get('instance_id') or '')
        selected_prompt_variant_indices = _parse_prompt_variant_indices(
            args.prompt_variant_indices if args.prompt_variant_indices is not None else str(args.prompt_variant_index),
            row,
        )
        existing_variant_predictions = _existing_variant_prediction_map(existing_rows_by_id.get(instance_id, {}))
        variant_predictions = dict(existing_variant_predictions)
        for prompt_variant_index in selected_prompt_variant_indices:
            if prompt_variant_index in existing_variant_predictions:
                continue
            prompt_variant = _select_prompt_variant(row, prompt_variant_index)
            user_requirement = str(prompt_variant.get('user_requirement') or '').strip()
            variant_jobs.append(
                (
                    row_index,
                    row,
                    prompt_variant_index,
                    {
                        'prompt_style_id': str(prompt_variant.get('style_id') or ''),
                        'prompt_style_label': str(prompt_variant.get('style_label') or ''),
                        'user_requirement': user_requirement,
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': _render_user_prompt(row, user_requirement, schema_hint)},
                        ],
                    },
                    variant_predictions,
                    selected_prompt_variant_indices,
                )
            )
        output_rows.append(
            {
                **_base_inference_row(row),
                'request_model': model,
                'request_base_url': base_url,
                'selected_prompt_variant_indices': selected_prompt_variant_indices,
                'variant_predictions': [existing_variant_predictions[key] for key in sorted(existing_variant_predictions)],
            }
        )

    track_name = eval_path.stem or 'unknown'
    print(
        f'start infer track={track_name} model={model} '
        f'num_rows={len(rows)} progress_every={args.progress_every} resume={bool(args.resume)} '
        f'concurrency={max(1, int(args.concurrency))} base_url={base_url}',
        flush=True,
    )

    if variant_jobs:
        first_job = variant_jobs[0]
        preflight_result = infer_backend.infer_one(first_job[3]['messages'])
        if preflight_result.error and _is_fatal_request_error(preflight_result.error):
            raise SystemExit(
                f'preflight infer request failed for model={model} base_url={base_url}: {preflight_result.error}'
            )

        remaining_jobs = variant_jobs[1:]
        infer_results = [preflight_result]
        if remaining_jobs:
            infer_results.extend(infer_backend.infer([job[3]['messages'] for job in remaining_jobs]))

        for index, infer_result in enumerate(infer_results, start=1):
            row_index, row, prompt_variant_index, meta, variant_predictions, selected_prompt_variant_indices = variant_jobs[index - 1]
            instance_id = str(row.get('instance_id') or '')
            response_text = infer_result.text
            prediction_payload = None
            prediction_error = None
            if infer_result.error is not None:
                prediction_error = f'request_error: {infer_result.error}'
            else:
                prediction_payload, prediction_error = _extract_prediction_payload(response_text)
            retry_attempted = False
            if _is_retryable_prediction_error(prediction_error):
                retry_attempted = True
                print(
                    f'retry infer track={track_name} instance_id={instance_id or "UNKNOWN"} '
                    f'prompt_variant_index={prompt_variant_index} reason={prediction_error}',
                    flush=True,
                )
                retry_result = infer_backend.infer_one(meta['messages'])
                response_text = retry_result.text
                if retry_result.error is not None:
                    prediction_payload = None
                    prediction_error = f'request_error: {retry_result.error}'
                else:
                    prediction_payload, prediction_error = _extract_prediction_payload(response_text)
            _log_prediction_issue(
                track_name=track_name,
                instance_id=instance_id,
                prompt_variant_index=prompt_variant_index,
                prediction_error=prediction_error,
                response_text=response_text,
            )
            predicted_status, predicted_clean_text = _extract_prediction_fields(prediction_payload)
            variant_predictions[prompt_variant_index] = {
                'prompt_variant_index': prompt_variant_index,
                'prompt_style_id': meta['prompt_style_id'],
                'prompt_style_label': meta['prompt_style_label'],
                'user_requirement': meta['user_requirement'],
                'request_model': model,
                'request_base_url': base_url,
                'raw_response': response_text,
                'parsed_response': prediction_payload,
                'prediction_error': prediction_error,
                'prediction_valid_json': prediction_error is None,
                'retry_attempted': retry_attempted,
                'response_usage': {},
                'predicted_status': predicted_status,
                'predicted_clean_text': predicted_clean_text,
            }
            output_rows[row_index]['variant_predictions'] = [variant_predictions[key] for key in sorted(variant_predictions)]
            output_rows[row_index]['selected_prompt_variant_indices'] = selected_prompt_variant_indices
            if index % args.progress_every == 0 or index == len(variant_jobs):
                elapsed = time.time() - started
                _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
                print(f'progress infer variant={index}/{len(variant_jobs)} elapsed_sec={elapsed:.1f}', flush=True)

    _write_progress_snapshot(output_path, output_dir, output_rows, model, base_url, track_name)
    print(f'wrote predictions -> {output_path}', flush=True)
    print(f'wrote summary -> {output_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
