#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ORDER_SLOTS = ('front', 'middle', 'end')


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


def _resolve_source_file(source_dir: Path, filename: str) -> Path:
    direct = source_dir / filename
    if direct.exists():
        return direct
    fallback = source_dir.parent / filename
    if fallback.exists():
        return fallback
    raise SystemExit(f'missing required source file: expected {direct} or {fallback}')


def _family_rank_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        _to_int(row.get('selected_group_count')),
        _to_int(row.get('selected_variant_count')),
        _to_int(row.get('candidate_count')),
        _to_int(row.get('value_count')),
        str(row.get('order_family_id') or ''),
    )


def _group_sort_key(rows: list[dict[str, Any]]) -> tuple[str, str]:
    source_record_id = ''
    group_id = ''
    for row in rows:
        if not source_record_id:
            source_record_id = str(row.get('source_record_id') or '')
        if not group_id:
            group_id = str(row.get('order_group_instance_id') or '')
    return source_record_id, group_id


def _normalize_group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    by_slot = {str(row.get('order_slot') or ''): row for row in rows}
    if set(by_slot) != set(ORDER_SLOTS):
        return None
    return [by_slot['front'], by_slot['middle'], by_slot['end']]


def _select_best_families(summary_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    kept_rows = [row for row in summary_rows if str(row.get('status') or '').strip() == 'kept']
    by_recipe: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kept_rows:
        recipe_id = str(row.get('recipe_id') or '').strip()
        family_id = str(row.get('order_family_id') or '').strip()
        if recipe_id and family_id:
            by_recipe[recipe_id].append(row)

    selected_families: dict[str, dict[str, str]] = {}
    manifest_rows: list[dict[str, Any]] = []
    for recipe_id in sorted(by_recipe):
        best = max(by_recipe[recipe_id], key=_family_rank_key)
        family_id = str(best['order_family_id'])
        selected_families[family_id] = best
        manifest_rows.append(
            {
                'recipe_id': recipe_id,
                'order_family_id': family_id,
                'filter_name': best.get('filter_name'),
                'candidate_count': _to_int(best.get('candidate_count')),
                'usable_record_count': _to_int(best.get('usable_record_count')),
                'value_count': _to_int(best.get('value_count')),
                'full_selected_group_count': _to_int(best.get('selected_group_count')),
                'full_selected_variant_count': _to_int(best.get('selected_variant_count')),
                'keep_count': _to_int(best.get('keep_count')),
                'drop_count': _to_int(best.get('drop_count')),
            }
        )
    return selected_families, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a smaller engineering subset from the full order-sensitivity benchmark.'
    )
    parser.add_argument('--source-dir', default='data/benchmark_full/order_sensitivity')
    parser.add_argument('--output-dir', default='data/benchmark/order_sensitivity')
    parser.add_argument('--groups-per-family', type=int, default=5)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.groups_per_family <= 0:
        raise SystemExit('--groups-per-family must be > 0')

    benchmark_path = _resolve_source_file(source_dir, 'order_sensitivity.jsonl')
    summary_path = _resolve_source_file(source_dir, 'order_sensitivity_summary.csv')

    summary_rows = _read_csv(summary_path)
    selected_families_by_id, manifest_rows = _select_best_families(summary_rows)
    if not selected_families_by_id:
        raise SystemExit(f'no kept order families found in {summary_path}')

    full_rows = _read_jsonl(benchmark_path)
    group_rows_by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in full_rows:
        family_id = str(row.get('order_family_id') or '')
        group_id = str(row.get('order_group_instance_id') or '')
        if family_id in selected_families_by_id and group_id:
            group_rows_by_family[family_id][group_id].append(row)

    subset_rows: list[dict[str, Any]] = []
    kept_manifest_rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        family_id = str(manifest_row['order_family_id'])
        normalized_groups = []
        for group_id, rows in group_rows_by_family.get(family_id, {}).items():
            normalized = _normalize_group_rows(rows)
            if normalized is not None:
                normalized_groups.append((group_id, normalized))
        normalized_groups.sort(key=lambda item: _group_sort_key(item[1]))
        selected_groups = normalized_groups[: args.groups_per_family]
        family_subset_rows = [row for _, rows in selected_groups for row in rows]
        subset_rows.extend(family_subset_rows)
        kept_manifest_rows.append(
            {
                **manifest_row,
                'subset_selected_group_count': len(selected_groups),
                'subset_selected_variant_count': len(family_subset_rows),
                'subset_keep_count': sum(
                    1 for row in family_subset_rows if str(row.get('reference_status') or '').upper() == 'KEEP'
                ),
                'subset_drop_count': sum(
                    1 for row in family_subset_rows if str(row.get('reference_status') or '').upper() == 'DROP'
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / 'order_sensitivity.jsonl', subset_rows)
    _write_csv(output_dir / 'order_sensitivity_summary.csv', kept_manifest_rows)
    _write_csv(output_dir / 'selected_families.csv', kept_manifest_rows)

    print(
        f'wrote engineering order subset: recipes={len({row["recipe_id"] for row in kept_manifest_rows})} '
        f'families={len(kept_manifest_rows)} groups={sum(row["subset_selected_group_count"] for row in kept_manifest_rows)} '
        f'rows={len(subset_rows)} -> {output_dir / "order_sensitivity.jsonl"}',
        flush=True,
    )
    print(f'wrote subset summary -> {output_dir / "order_sensitivity_summary.csv"}', flush=True)
    print(f'wrote selected families manifest -> {output_dir / "selected_families.csv"}', flush=True)


if __name__ == '__main__':
    main()
