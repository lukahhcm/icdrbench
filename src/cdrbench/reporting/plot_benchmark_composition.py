#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_jsonl_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(_read_jsonl(path))


def _load_csv_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _track_jsonl_path(benchmark_dir: Path, track: str) -> Path:
    direct = benchmark_dir / f"{track}.jsonl"
    nested = benchmark_dir / track / f"{track}.jsonl"
    return direct if direct.exists() else nested


def _track_csv_path(benchmark_dir: Path, track: str, filename: str) -> Path:
    direct = benchmark_dir / filename
    nested = benchmark_dir / track / filename
    return direct if direct.exists() else nested


def _preferred_column(df: pd.DataFrame, *columns: str) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None


def _value_counts(df: pd.DataFrame, column: str, *, unique_key: str | None = None) -> pd.Series:
    if df.empty or column not in df.columns:
        return pd.Series(dtype=int)
    frame = df.copy()
    frame[column] = frame[column].fillna("unknown").astype(str)
    if unique_key and unique_key in frame.columns:
        counts = frame.groupby(column)[unique_key].nunique().sort_values(ascending=False)
        counts.name = "count"
        return counts
    counts = frame[column].value_counts().sort_values(ascending=False)
    counts.name = "count"
    return counts


def _sort_status(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    order = {"kept": 0}
    ordered_index = sorted(series.index.tolist(), key=lambda item: (order.get(str(item), 1), str(item)))
    return series.reindex(ordered_index)


def _counts_to_records(series: pd.Series, label_name: str) -> list[dict[str, Any]]:
    return [{label_name: str(idx), "count": int(val)} for idx, val in series.items()]


def _autopct(values: list[int]):
    total = sum(values)

    def inner(pct: float) -> str:
        if total <= 0:
            return ""
        absolute = int(round(pct * total / 100.0))
        if pct < 4:
            return ""
        return f"{pct:.0f}%\n(n={absolute})"

    return inner


def _draw_donut(ax, series: pd.Series, title: str, colors: list[str] | None = None) -> None:
    ax.set_title(title, fontsize=11, pad=12)
    if series.empty or int(series.sum()) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return

    labels = [str(idx) for idx in series.index]
    values = [int(v) for v in series.values]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=_autopct(values),
        startangle=90,
        counterclock=False,
        pctdistance=0.77,
        labeldistance=1.08,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_color("#222222")
    ax.text(0, 0, f"n={sum(values)}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.axis("equal")


def _draw_bar(ax, series: pd.Series, title: str, color: str, *, horizontal: bool = False) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    if series.empty or int(series.sum()) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return

    labels = [str(idx) for idx in series.index]
    values = [int(v) for v in series.values]
    if horizontal:
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=color, alpha=0.9)
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=9)
        for bar, value in zip(bars, values):
            ax.text(value + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=8)
    else:
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=color, alpha=0.9)
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _draw_stacked_domain_bars(ax, counts_by_track: dict[str, pd.Series], title: str, palette: list[str]) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    tracks = list(counts_by_track.keys())
    domains = sorted({str(idx) for series in counts_by_track.values() for idx in series.index})
    if not domains or not tracks:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return

    color_map = {domain: palette[idx % len(palette)] for idx, domain in enumerate(domains)}
    bottoms = np.zeros(len(tracks))
    x = np.arange(len(tracks))
    for domain in domains:
        values = np.array([int(counts_by_track[track].get(domain, 0)) for track in tracks])
        ax.bar(x, values, bottom=bottoms, label=domain, color=color_map[domain], alpha=0.95)
        bottoms += values

    ax.set_xticks(x, tracks, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for xi, total in zip(x, bottoms):
        ax.text(xi, total, str(int(total)), ha="center", va="bottom", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _track_row_count(df: pd.DataFrame, unique_key: str | None = None) -> int:
    if df.empty:
        return 0
    if unique_key and unique_key in df.columns:
        return int(df[unique_key].nunique())
    return int(len(df))


def _order_structure_series(order_df: pd.DataFrame) -> pd.Series:
    row_count = len(order_df) if not order_df.empty else 0
    group_count = order_df["order_group_instance_id"].nunique() if "order_group_instance_id" in order_df.columns else 0
    family_count = order_df["order_family_id"].nunique() if "order_family_id" in order_df.columns else 0
    return pd.Series(
        {
            "families": int(family_count),
            "groups": int(group_count),
            "rows": int(row_count),
        }
    )


def _top_atomic_operators(atomic_df: pd.DataFrame, k: int = 12) -> pd.Series:
    series = _value_counts(atomic_df, "operator")
    if len(series) <= k:
        return series
    top = series.iloc[:k].copy()
    top.loc["others"] = int(series.iloc[k:].sum())
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark composition charts for paper-ready inspection.")
    parser.add_argument("--benchmark-dir", default="data/benchmark")
    parser.add_argument("--recipe-library-dir", "--workflow-library-dir", dest="recipe_library_dir", default="data/processed/recipe_library")
    parser.add_argument("--output-dir", default="data/evaluation/reports/plots")
    args = parser.parse_args()

    benchmark_dir = (ROOT / args.benchmark_dir).resolve()
    recipe_library_dir = (ROOT / args.recipe_library_dir).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = recipe_library_dir / "recipe_library_summary.csv"
    if not summary_path.exists():
        summary_path = recipe_library_dir / "workflow_library_summary.csv"
    recipe_summary = _load_csv_frame(summary_path)
    main_df = _load_jsonl_frame(_track_jsonl_path(benchmark_dir, "main"))
    order_df = _load_jsonl_frame(_track_jsonl_path(benchmark_dir, "order_sensitivity"))
    atomic_df = _load_jsonl_frame(_track_jsonl_path(benchmark_dir, "atomic_ops"))
    main_summary = _load_csv_frame(_track_csv_path(benchmark_dir, "main", "main_summary.csv"))
    order_summary = _load_csv_frame(_track_csv_path(benchmark_dir, "order_sensitivity", "order_sensitivity_summary.csv"))
    atomic_summary = _load_csv_frame(_track_csv_path(benchmark_dir, "atomic_ops", "atomic_ops_summary.csv"))

    recipe_domains = _value_counts(recipe_summary, "domain")
    main_domains = _value_counts(main_df, "domain")
    main_type_column = _preferred_column(main_df, "recipe_type", "workflow_type") or "recipe_type"
    summary_type_label = "recipe_type"
    main_types = _value_counts(main_df, main_type_column)
    main_status = _sort_status(_value_counts(main_summary, "status"))
    order_domains = _value_counts(order_df, "domain", unique_key="order_group_instance_id")
    order_status = _sort_status(_value_counts(order_summary, "status"))
    atomic_domains = _value_counts(atomic_df, "source_domain")
    atomic_status = _sort_status(_value_counts(atomic_summary, "status"))
    track_size_series = pd.Series(
        {
            "atomic_ops": _track_row_count(atomic_df),
            "main": _track_row_count(main_df),
            "order_sensitivity": _track_row_count(order_df),
        }
    )
    domain_by_track = {
        "atomic_ops": atomic_domains,
        "main": main_domains,
        "order_sensitivity": order_domains,
    }
    order_structure = _order_structure_series(order_df)
    atomic_top_ops = _top_atomic_operators(atomic_df, k=10)

    palette = [
        "#0B7285",
        "#74C0FC",
        "#F59F00",
        "#E8590C",
        "#2B8A3E",
        "#C2255C",
        "#5F3DC4",
        "#495057",
    ]
    status_palette = ["#2B8A3E", "#F08C00", "#C92A2A", "#868E96", "#1C7ED6", "#9C36B5"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    fig.suptitle("CDR-Bench Benchmark Composition", fontsize=18, fontweight="bold")

    _draw_bar(axes[0, 0], track_size_series, "Track Sizes", "#0B7285")
    _draw_stacked_domain_bars(axes[0, 1], domain_by_track, "Domain Composition by Track", palette)
    _draw_donut(axes[0, 2], recipe_domains, "Recipe Library Domains", palette)

    _draw_donut(axes[1, 0], main_types, "Main Track by Recipe Type", palette)
    _draw_donut(axes[1, 1], main_status, "Main Variant Status", status_palette)
    _draw_bar(axes[1, 2], order_structure, "Order Track Structure", "#E8590C")

    _draw_donut(axes[2, 0], order_status, "Order Family Status", status_palette)
    _draw_bar(axes[2, 1], atomic_top_ops, "Atomic Top Operators", "#5F3DC4", horizontal=True)
    _draw_donut(axes[2, 2], atomic_status, "Atomic Operator Status", status_palette)

    png_path = output_dir / "benchmark_composition_overview.png"
    pdf_path = output_dir / "benchmark_composition_overview.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary_payload = {
        "recipe_library_domains": _counts_to_records(recipe_domains, "domain"),
        "track_sizes": _counts_to_records(track_size_series, "track"),
        "main_domains": _counts_to_records(main_domains, "domain"),
        "main_recipe_types": _counts_to_records(main_types, summary_type_label),
        "main_variant_status": _counts_to_records(main_status, "status"),
        "order_track_structure": _counts_to_records(order_structure, "component"),
        "order_group_domains": _counts_to_records(order_domains, "domain"),
        "order_family_status": _counts_to_records(order_status, "status"),
        "atomic_source_domains": _counts_to_records(atomic_domains, "source_domain"),
        "atomic_top_operators": _counts_to_records(atomic_top_ops, "operator"),
        "atomic_operator_status": _counts_to_records(atomic_status, "status"),
    }
    (output_dir / "benchmark_composition_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(_counts_to_records(recipe_domains, "domain")).to_csv(output_dir / "recipe_library_domains.csv", index=False)
    pd.DataFrame(_counts_to_records(track_size_series, "track")).to_csv(output_dir / "track_sizes.csv", index=False)
    pd.DataFrame(_counts_to_records(main_domains, "domain")).to_csv(output_dir / "main_domains.csv", index=False)
    pd.DataFrame(_counts_to_records(main_types, summary_type_label)).to_csv(output_dir / "main_recipe_types.csv", index=False)
    pd.DataFrame(_counts_to_records(main_status, "status")).to_csv(output_dir / "main_variant_status.csv", index=False)
    pd.DataFrame(_counts_to_records(order_structure, "component")).to_csv(output_dir / "order_track_structure.csv", index=False)
    pd.DataFrame(_counts_to_records(order_domains, "domain")).to_csv(output_dir / "order_group_domains.csv", index=False)
    pd.DataFrame(_counts_to_records(order_status, "status")).to_csv(output_dir / "order_family_status.csv", index=False)
    pd.DataFrame(_counts_to_records(atomic_domains, "source_domain")).to_csv(output_dir / "atomic_source_domains.csv", index=False)
    pd.DataFrame(_counts_to_records(atomic_top_ops, "operator")).to_csv(output_dir / "atomic_top_operators.csv", index=False)
    pd.DataFrame(_counts_to_records(atomic_status, "status")).to_csv(output_dir / "atomic_operator_status.csv", index=False)

    print(f"wrote benchmark composition plots -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
