#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


RECIPE_TYPES = ('clean-only', 'filter-then-clean', 'clean-then-filter')


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == '.jsonl':
        return _read_jsonl(path)
    return _read_csv(path)


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
    with tmp_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _variant_rank_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        _to_int(row.get('candidate_count')),
        _to_int(row.get('selected_count')),
        _to_int(row.get('value_count')),
        str(row.get('recipe_variant_id') or ''),
    )


def _row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get('source_record_id') or ''), str(row.get('instance_id') or ''))


def _take_balanced_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return sorted(rows, key=_row_sort_key)

    ordered = sorted(rows, key=_row_sort_key)
    keep_rows = [row for row in ordered if str(row.get('reference_status') or '').upper() == 'KEEP']
    drop_rows = [row for row in ordered if str(row.get('reference_status') or '').upper() == 'DROP']
    other_rows = [
        row for row in ordered if str(row.get('reference_status') or '').upper() not in {'KEEP', 'DROP'}
    ]

    if not keep_rows or not drop_rows:
        return ordered[:max_rows]

    target_keep = max_rows // 2
    target_drop = max_rows // 2
    selected = [*keep_rows[:target_keep], *drop_rows[:target_drop]]

    keep_remaining = keep_rows[target_keep:]
    drop_remaining = drop_rows[target_drop:]
    if max_rows % 2 == 1:
        if len(keep_remaining) >= len(drop_remaining) and keep_remaining:
            selected.append(keep_remaining[0])
            keep_remaining = keep_remaining[1:]
        elif drop_remaining:
            selected.append(drop_remaining[0])
            drop_remaining = drop_remaining[1:]

    remainder = [*keep_remaining, *drop_remaining, *other_rows]
    if len(selected) < max_rows:
        selected.extend(remainder[: max_rows - len(selected)])

    return sorted(selected[:max_rows], key=_row_sort_key)


def _resolve_source_file(source_dir: Path, filename: str) -> Path:
    direct = source_dir / filename
    if direct.exists():
        return direct
    fallback = source_dir.parent / filename
    if fallback.exists():
        return fallback
    raise SystemExit(f'missing required source file: expected {direct} or {fallback}')


def _resolve_source_file_candidates(source_dir: Path, filenames: list[str]) -> Path:
    for filename in filenames:
        direct = source_dir / filename
        if direct.exists():
            return direct
        fallback = source_dir.parent / filename
        if fallback.exists():
            return fallback
    expected = ' or '.join(str(source_dir / filename) for filename in filenames)
    fallback_expected = ' or '.join(str(source_dir.parent / filename) for filename in filenames)
    raise SystemExit(f'missing required source file: expected one of {expected} or {fallback_expected}')


def _select_best_variants(summary_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    kept_rows = [row for row in summary_rows if str(row.get('status') or '').strip() == 'kept']
    by_recipe_and_type: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in kept_rows:
        recipe_id = str(row.get('recipe_id') or '').strip()
        recipe_type = str(row.get('recipe_type') or '').strip()
        variant_id = str(row.get('recipe_variant_id') or '').strip()
        if not recipe_id or not variant_id or recipe_type not in RECIPE_TYPES:
            continue
        by_recipe_and_type[(recipe_id, recipe_type)].append(row)

    selected_variants: dict[str, dict[str, str]] = {}
    manifest_rows: list[dict[str, Any]] = []
    recipe_ids = sorted({recipe_id for recipe_id, _ in by_recipe_and_type})
    for recipe_id in recipe_ids:
        for recipe_type in RECIPE_TYPES:
            candidates = by_recipe_and_type.get((recipe_id, recipe_type), [])
            if not candidates:
                continue
            best = max(candidates, key=_variant_rank_key)
            variant_id = str(best['recipe_variant_id'])
            selected_variants[variant_id] = best
            manifest_rows.append(
                {
                    'recipe_id': recipe_id,
                    'recipe_type': recipe_type,
                    'recipe_variant_id': variant_id,
                    'candidate_count': _to_int(best.get('candidate_count')),
                    'value_count': _to_int(best.get('value_count')),
                    'full_selected_count': _to_int(best.get('selected_count')),
                    'keep_count': _to_int(best.get('keep_count')),
                    'drop_count': _to_int(best.get('drop_count')),
                }
            )
    return selected_variants, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a smaller engineering subset from the full main benchmark.'
    )
    parser.add_argument('--source-dir', default='data/benchmark_full/main')
    parser.add_argument('--output-dir', default='data/benchmark/main')
    parser.add_argument('--rows-per-variant', type=int, default=10)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.rows_per_variant <= 0:
        raise SystemExit('--rows-per-variant must be > 0')

    main_path = _resolve_source_file(source_dir, 'main.jsonl')
    main_summary_path = _resolve_source_file_candidates(
        source_dir,
        [
            'main_summary.csv',
            'prompt_eval_build_summary.csv',
            'prompt_eval_build_summary.jsonl',
        ],
    )

    summary_rows = _read_table(main_summary_path)
    selected_variants_by_id, manifest_rows = _select_best_variants(summary_rows)
    if not selected_variants_by_id:
        raise SystemExit(f'no kept main variants found in {main_summary_path}')

    full_main_rows = _read_jsonl(main_path)
    rows_by_variant_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_main_rows:
        variant_id = str(row.get('recipe_variant_id') or '')
        if variant_id in selected_variants_by_id:
            rows_by_variant_id[variant_id].append(row)

    subset_rows: list[dict[str, Any]] = []
    kept_manifest_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        variant_id = str(manifest_row['recipe_variant_id'])
        sampled_rows = _take_balanced_rows(rows_by_variant_id.get(variant_id, []), args.rows_per_variant)
        subset_rows.extend(sampled_rows)
        kept_manifest_rows.append(
            {
                **manifest_row,
                'subset_selected_count': len(sampled_rows),
                'subset_keep_count': sum(
                    1 for row in sampled_rows if str(row.get('reference_status') or '').upper() == 'KEEP'
                ),
                'subset_drop_count': sum(
                    1 for row in sampled_rows if str(row.get('reference_status') or '').upper() == 'DROP'
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / 'main.jsonl', subset_rows)
    _write_csv(output_dir / 'main_summary.csv', kept_manifest_rows)
    _write_csv(output_dir / 'selected_variants.csv', kept_manifest_rows)

    print(
        f'wrote engineering main subset: recipes={len({row["recipe_id"] for row in kept_manifest_rows})} '
        f'variants={len(kept_manifest_rows)} rows={len(subset_rows)} -> {output_dir / "main.jsonl"}',
        flush=True,
    )
    print(f'wrote subset summary -> {output_dir / "main_summary.csv"}', flush=True)
    print(f'wrote selected variants manifest -> {output_dir / "selected_variants.csv"}', flush=True)


if __name__ == '__main__':
    main()
