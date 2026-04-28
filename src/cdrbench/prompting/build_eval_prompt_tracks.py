#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
EVAL_PROGRESS_EVERY = 200

TRACK_FILES = {
    'main': 'main.jsonl',
    'order_sensitivity': 'order_sensitivity.jsonl',
    'atomic_ops': 'atomic_ops.jsonl',
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _stable_id(*parts: Any, length: int = 16) -> str:
    blob = '||'.join(_stable_json(part) if isinstance(part, (dict, list)) else str(part) for part in parts)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:length]


def _workflow_key(row: dict[str, Any]) -> str:
    operator_sequence = list(row.get('operator_sequence') or ([row['operator']] if row.get('operator') else []))
    return _stable_id(
        row.get('benchmark_track'),
        row.get('domain'),
        row.get('workflow_type'),
        row.get('order_slot'),
        operator_sequence,
        row.get('filter_params_by_name') or {},
    )


def _sample_prompt_variants(
    candidates: list[dict[str, Any]],
    *,
    workflow_prompt_key: str,
    instance_id: str,
    sample_count: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    candidates_by_style: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        style_id = str(candidate.get('style_id') or '')
        if not style_id:
            continue
        candidates_by_style.setdefault(style_id, []).append(candidate)

    sampled_style_ids = sorted(
        candidates_by_style,
        key=lambda style_id: _stable_id(
            'prompt-style-sample',
            sample_seed,
            workflow_prompt_key,
            instance_id,
            style_id,
        ),
    )[:sample_count]

    prompt_variants = []
    for style_id in sampled_style_ids:
        style_candidates = sorted(
            candidates_by_style[style_id],
            key=lambda candidate: _stable_id(
                'prompt-candidate-sample',
                sample_seed,
                workflow_prompt_key,
                instance_id,
                style_id,
                candidate.get('candidate_id') or candidate.get('user_requirement') or '',
            ),
        )
        candidate = style_candidates[0]
        prompt_variants.append(
            {
                'style_id': str(candidate.get('style_id') or ''),
                'style_label': str(candidate.get('style_label') or ''),
                'user_requirement': str(candidate.get('user_requirement') or ''),
            }
        )
    return prompt_variants


def _eval_row(
    row: dict[str, Any],
    *,
    workflow_prompt_key: str,
    candidates: list[dict[str, Any]],
    prompt_variants_per_sample: int,
    prompt_sampling_seed: int,
) -> dict[str, Any]:
    prompt_variants = _sample_prompt_variants(
        candidates,
        workflow_prompt_key=workflow_prompt_key,
        instance_id=str(row.get('instance_id') or ''),
        sample_count=prompt_variants_per_sample,
        sample_seed=prompt_sampling_seed,
    )
    keep_fields = [
        'instance_id',
        'benchmark_track',
        'domain',
        'source_domain',
        'workflow_id',
        'workflow_variant_id',
        'workflow_type',
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
    output_row = {field: row[field] for field in keep_fields if field in row}
    output_row.update(
        {
            'workflow_prompt_key': workflow_prompt_key,
            'prompt_candidate_pool_count': len(candidates),
            'prompt_variant_count': len(prompt_variants),
            'prompt_sampling_policy': 'deterministic_distinct_styles_without_replacement',
            'prompt_sampling_seed': prompt_sampling_seed,
            'prompt_variants': prompt_variants,
        }
    )
    return output_row


def main() -> None:
    parser = argparse.ArgumentParser(description='Build eval-ready prompt track files from an accepted recipe prompt library.')
    parser.add_argument('--benchmark-dir', default='data/benchmark')
    parser.add_argument('--prompt-library', default='data/benchmark_prompts/recipe_prompt_library.jsonl')
    parser.add_argument('--output-dir', default='data/benchmark_prompts/eval')
    parser.add_argument('--tracks', nargs='*', default=list(TRACK_FILES), choices=sorted(TRACK_FILES))
    parser.add_argument('--prompt-variants-per-sample', type=int, default=3)
    parser.add_argument('--prompt-sampling-seed', type=int, default=0)
    parser.add_argument(
        '--min-prompt-variants-per-sample',
        type=int,
        default=3,
        help='Skip samples whose recipe prompt pool cannot supply at least this many distinct styles.',
    )
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    prompt_library_path = (ROOT / args.prompt_library).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    library_rows = _read_jsonl(prompt_library_path)
    library_by_key = {
        str(row.get('workflow_prompt_key')): list(row.get('candidates') or [])
        for row in library_rows
        if row.get('workflow_prompt_key')
    }

    summary_rows = []
    for track in args.tracks:
        input_path = benchmark_dir / TRACK_FILES[track]
        if not input_path.exists():
            print(f'skip missing {track}: {input_path}', flush=True)
            continue
        rows = _read_jsonl(input_path)
        total_rows = len(rows)
        output_rows = []
        missing_pool_rows = 0
        insufficient_style_rows = 0
        print(f'start eval track={track} input_rows={total_rows}', flush=True)
        for row_index, row in enumerate(rows, start=1):
            workflow_prompt_key = _workflow_key(row)
            candidates = list(library_by_key.get(workflow_prompt_key) or [])
            if not candidates:
                missing_pool_rows += 1
                if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                    print(
                        f'progress eval track={track} row={row_index}/{total_rows} '
                        f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                        f'insufficient_styles={insufficient_style_rows}',
                        flush=True,
                    )
                continue
            distinct_style_count = len({str(candidate.get('style_id') or '') for candidate in candidates if candidate.get('style_id')})
            if distinct_style_count < args.min_prompt_variants_per_sample:
                insufficient_style_rows += 1
                if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                    print(
                        f'progress eval track={track} row={row_index}/{total_rows} '
                        f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                        f'insufficient_styles={insufficient_style_rows}',
                        flush=True,
                    )
                continue
            output_rows.append(
                _eval_row(
                    row,
                    workflow_prompt_key=workflow_prompt_key,
                    candidates=candidates,
                    prompt_variants_per_sample=args.prompt_variants_per_sample,
                    prompt_sampling_seed=args.prompt_sampling_seed,
                )
            )
            if row_index % EVAL_PROGRESS_EVERY == 0 or row_index == total_rows:
                print(
                    f'progress eval track={track} row={row_index}/{total_rows} '
                    f'kept={len(output_rows)} missing_pool={missing_pool_rows} '
                    f'insufficient_styles={insufficient_style_rows}',
                    flush=True,
                )

        count = _write_jsonl(output_dir / TRACK_FILES[track], output_rows)
        summary_rows.append(
            {
                'track': track,
                'input_rows': len(rows),
                'kept_rows': count,
                'missing_pool_rows': missing_pool_rows,
                'insufficient_style_rows': insufficient_style_rows,
                'prompt_variants_per_sample': args.prompt_variants_per_sample,
                'prompt_sampling_seed': args.prompt_sampling_seed,
                'min_prompt_variants_per_sample': args.min_prompt_variants_per_sample,
            }
        )
        print(f'wrote eval track {track}: {count} rows -> {output_dir / TRACK_FILES[track]}', flush=True)

    _write_jsonl(output_dir / 'prompt_eval_build_summary.jsonl', summary_rows)
    print(f'wrote eval build summary -> {output_dir / "prompt_eval_build_summary.jsonl"}', flush=True)


if __name__ == '__main__':
    main()
