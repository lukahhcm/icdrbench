#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from djbench.config import load_domains_config
from djbench.domain_labeling import (
    build_domain_execution_plan,
    domain_operator_catalog_frame,
    process_corpus,
)


def _resolve_path(root: Path, raw_path_value: str) -> Path:
    raw_path = Path(raw_path_value)
    if raw_path.is_absolute():
        return raw_path
    return root / raw_path


def _count_jsonl_lines(path: Path) -> int:
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Tag corpora with all domain operators, infer per-record domains, and keep only records with enough active mappers.'
    )
    parser.add_argument('--corpora-config', default='configs/corpora.yaml')
    parser.add_argument('--domains-config', default='configs/domains.yaml')
    parser.add_argument('--corpora', nargs='*', default=None)
    parser.add_argument('--tagged-dir', default='data/processed/domain_tags')
    parser.add_argument('--filtered-dir', default='data/processed/domain_filtered')
    parser.add_argument('--combined-path', default='data/processed/domain_filtered/all.jsonl')
    parser.add_argument('--summary-path', default='outputs/domain_labeling_summary.csv')
    parser.add_argument('--assignments-path', default='outputs/domain_assignment_counts.csv')
    parser.add_argument('--catalog-path', default='outputs/domain_operator_catalog.csv')
    parser.add_argument('--min-active-mappers', type=int, default=2)
    parser.add_argument('--max-records', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='Resume from existing per-corpus tagged/filtered outputs.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    corpora_cfg = load_domains_config(root / args.corpora_config)['corpora']
    domains_cfg = load_domains_config(root / args.domains_config)
    selected = set(args.corpora) if args.corpora else None

    plan = build_domain_execution_plan(domains_cfg)
    catalog = domain_operator_catalog_frame(plan)
    catalog_path = root / args.catalog_path
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_path, index=False)
    print(f'wrote operator catalog -> {catalog_path}')

    summary_rows = []
    assignment_rows = []
    combined_handle = None
    combined_path = root / args.combined_path

    try:
        for corpus_name, corpus_cfg in corpora_cfg.items():
            if selected and corpus_name not in selected:
                continue
            raw_path = _resolve_path(root, corpus_cfg['raw_path'])
            if not raw_path.exists():
                print(f'skip {corpus_name}: missing {raw_path}')
                continue

            if combined_handle is None:
                combined_path.parent.mkdir(parents=True, exist_ok=True)
                combined_mode = 'a' if args.resume and combined_path.exists() else 'w'
                combined_handle = open(combined_path, combined_mode, encoding='utf-8')

            tagged_path = root / args.tagged_dir / f'{corpus_name}.jsonl'
            filtered_path = root / args.filtered_dir / f'{corpus_name}.jsonl'
            total_records_hint = _count_jsonl_lines(raw_path)
            if args.max_records is not None:
                total_records_hint = min(total_records_hint, args.max_records)
            summary_row, corpus_assignment_rows = process_corpus(
                corpus_name=corpus_name,
                raw_path=raw_path,
                tagged_path=tagged_path,
                filtered_path=filtered_path,
                plan=plan,
                field_map=corpus_cfg.get('field_map'),
                defaults=corpus_cfg.get('defaults'),
                min_active_mappers=args.min_active_mappers,
                max_records=args.max_records,
                progress_every=args.progress_every,
                total_records_hint=total_records_hint,
                resume=args.resume,
                combined_handle=combined_handle,
            )
            summary_rows.append(summary_row)
            assignment_rows.extend(corpus_assignment_rows)
            print(
                f"{corpus_name}: kept {summary_row['kept_records']} / {summary_row['total_records']} "
                f"(assigned {summary_row['assigned_records']}) -> {filtered_path}"
            )
    finally:
        if combined_handle is not None:
            combined_handle.close()

    if not summary_rows:
        print('no corpora were processed')
        return

    summary = pd.DataFrame(summary_rows)
    summary_path = root / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f'wrote summary -> {summary_path}')

    assignments = pd.DataFrame(
        assignment_rows,
        columns=['corpus', 'scope', 'assigned_domain', 'count'],
    )
    assignments_path = root / args.assignments_path
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    print(f'wrote assignment counts -> {assignments_path}')

    print(f'wrote combined filtered corpus -> {combined_path}')


if __name__ == '__main__':
    main()
