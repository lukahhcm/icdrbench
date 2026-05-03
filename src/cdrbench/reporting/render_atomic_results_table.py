#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _latex_escape(text: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    return ''.join(replacements.get(ch, ch) for ch in text)


def _slugify_model_name(name: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', name.strip())
    return slug.strip('_') or 'unknown_model'


def _format_rate(value: Any, *, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '--'
    return f'{number * 100:.{digits}f}'


def _format_rg(value: Any, *, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '--'
    return f'{number:.{digits}f}'


def _discover_atomic_metrics(score_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(score_root.glob('*/atomic_ops/paper_metrics.json')):
        payload = _load_json(path)
        if str(payload.get('track') or '') != 'atomic_ops':
            continue
        model = str(payload.get('model') or path.parents[1].name)
        rows.append(
            {
                'model': model,
                'mean_rs': payload.get('mean_rs'),
                'rs_at_k': payload.get('rs_at_k'),
                'mean_rg': payload.get('mean_rg'),
                'num_instances': payload.get('num_instances'),
                'source_path': str(path),
            }
        )
    return rows


def _render_table(rows: list[dict[str, Any]], *, caption: str, label: str) -> str:
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Model & RS & RS@3 & Mean RG \\',
        r'\midrule',
    ]

    for row in rows:
        model = _latex_escape(str(row.get('model') or 'unknown'))
        rs = _format_rate(row.get('mean_rs'))
        rs_at_3 = _format_rate(row.get('rs_at_k'))
        mean_rg = _format_rg(row.get('mean_rg'))
        lines.append(f'{model} & {rs} & {rs_at_3} & {mean_rg} \\\\')

    lines.extend(
        [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
    )
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render a LaTeX table summarizing atomic benchmark results from paper_metrics.json files.'
    )
    parser.add_argument('--score-root', default='data/evaluation/score', help='Root directory containing per-run score outputs.')
    parser.add_argument(
        '--output-path',
        default='data/evaluation/reports/atomic_results_table.tex',
        help='Output LaTeX file path.',
    )
    parser.add_argument(
        '--caption',
        default='Atomic benchmark results across models. RS denotes mean recipe success, RS@3 denotes pass-at-3 recipe success across prompt variants, and Mean RG denotes mean bounded refinement gain.',
    )
    parser.add_argument('--label', default='tab:atomic-results')
    parser.add_argument(
        '--sort-by',
        choices=('mean_rs', 'rs_at_k', 'mean_rg', 'model'),
        default='mean_rs',
        help='Primary sort key for table rows.',
    )
    args = parser.parse_args()

    score_root = (ROOT / args.score_root).resolve()
    output_path = (ROOT / args.output_path).resolve()
    rows = _discover_atomic_metrics(score_root)
    if not rows:
        raise SystemExit(f'No atomic paper_metrics.json files found under: {score_root}')

    if args.sort_by == 'model':
        rows.sort(key=lambda row: str(row.get('model') or ''))
    else:
        rows.sort(key=lambda row: (float(row.get(args.sort_by) or 0.0), str(row.get('model') or '')), reverse=True)

    table_text = _render_table(rows, caption=args.caption, label=args.label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text, encoding='utf-8')
    print(f'wrote atomic LaTeX table -> {output_path}', flush=True)


if __name__ == '__main__':
    main()
