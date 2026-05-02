#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


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


def _resolve_optional_file(base_dir: Path, filename: str) -> Path | None:
    candidate = base_dir / filename
    return candidate if candidate.exists() else None


def _operator_rank_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (
        _to_int(row.get('selected_count')),
        _to_int(row.get('candidate_count')),
        _to_int(row.get('keep_count')) + _to_int(row.get('drop_count')),
        str(row.get('operator') or ''),
    )


def _summary_manifest(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        operator = str(row.get('operator') or '').strip()
        if not operator:
            continue
        best = manifest.get(operator)
        if best is None or _operator_rank_key(row) > _operator_rank_key(best):
            manifest[operator] = {
                'operator': operator,
                'operator_kind': row.get('operator_kind'),
                'candidate_count': _to_int(row.get('candidate_count')),
                'full_selected_count': _to_int(row.get('selected_count')),
                'keep_count': _to_int(row.get('keep_count') or row.get('selected_keep_count')),
                'drop_count': _to_int(row.get('drop_count') or row.get('selected_drop_count')),
            }
    return manifest


def _rows_manifest(full_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        operator = str(row.get('operator') or '').strip()
        if not operator:
            continue
        by_operator[operator].append(row)
        meta.setdefault(operator, {'operator': operator, 'operator_kind': row.get('operator_kind')})
    manifest: dict[str, dict[str, Any]] = {}
    for operator, rows in by_operator.items():
        manifest[operator] = {
            **meta[operator],
            'full_selected_count': len(rows),
            'keep_count': sum(1 for row in rows if str(row.get('reference_status') or '').upper() == 'KEEP'),
            'drop_count': sum(1 for row in rows if str(row.get('reference_status') or '').upper() == 'DROP'),
        }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a smaller engineering subset from the full atomic benchmark.')
    parser.add_argument('--source-dir', default='data/benchmark_full/atomic_ops')
    parser.add_argument('--output-dir', default='data/benchmark/atomic_ops')
    parser.add_argument('--processed-summary-dir', default='data/processed/benchmark_instances')
    parser.add_argument('--rows-per-operator', type=int, default=6)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    processed_summary_dir = Path(args.processed_summary_dir).resolve()
    if args.rows_per_operator <= 0:
        raise SystemExit('--rows-per-operator must be > 0')

    atomic_path = _resolve_source_file(source_dir, 'atomic_ops.jsonl')
    full_rows = _read_jsonl(atomic_path)

    summary_path = _resolve_optional_file(processed_summary_dir, 'atomic_ops_summary.csv')
    manifest_by_operator = _summary_manifest(_read_csv(summary_path)) if summary_path is not None else {}
    if not manifest_by_operator:
        manifest_by_operator = _rows_manifest(full_rows)
    if not manifest_by_operator:
        raise SystemExit(f'no usable atomic operators found in {atomic_path}')

    rows_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        operator = str(row.get('operator') or '')
        if operator in manifest_by_operator:
            rows_by_operator[operator].append(row)

    subset_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for operator in sorted(manifest_by_operator):
        sampled_rows = _take_balanced_rows(rows_by_operator.get(operator, []), args.rows_per_operator)
        subset_rows.extend(sampled_rows)
        meta = manifest_by_operator[operator]
        manifest_rows.append(
            {
                **meta,
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
    _write_jsonl(output_dir / 'atomic_ops.jsonl', subset_rows)
    _write_csv(output_dir / 'atomic_ops_summary.csv', manifest_rows)
    _write_csv(output_dir / 'selected_operators.csv', manifest_rows)

    print(
        f'wrote engineering atomic subset: operators={len(manifest_rows)} rows={len(subset_rows)} '
        f'-> {output_dir / "atomic_ops.jsonl"}',
        flush=True,
    )
    print(f'wrote subset summary -> {output_dir / "atomic_ops_summary.csv"}', flush=True)
    print(f'wrote selected operators manifest -> {output_dir / "selected_operators.csv"}', flush=True)


if __name__ == '__main__':
    main()
