"""Generate PNG visualizations for docstring model metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import textwrap

DATA_PATH = Path("docstring_metrics_runs.csv")
FIGURES_DIR = Path("figures")


def read_rows() -> List[Dict[str, str]]:
    with DATA_PATH.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def summarise_by_model(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for row in rows:
        key = row["model_key"]
        model = summary.setdefault(
            key,
            {
                "label": row["model_label"],
                "size_mb": float(row["model_size_mb"]),
                "emissions": [],
                "inference": [],
                "accuracy": [],
            },
        )
        model["emissions"].append(float(row["emissions_kg"]))
        model["inference"].append(float(row["inference_time_s"]))
        model["accuracy"].append(1.0 if row["accuracy"].upper() == "PASS" else 0.0)
    for model in summary.values():
        model["avg_emissions"] = mean(model["emissions"])
        model["avg_inference"] = mean(model["inference"])
        model["pass_rate"] = mean(model["accuracy"])
    return summary


def bar_chart(
    keys: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filepath: Path,
    value_format: str,
    display_labels: Optional[List[str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(keys))
    bars = ax.bar(positions, values, color="#4f8bc9")
    ax.set_xticks(list(positions))
    labels = display_labels if display_labels is not None else keys
    wrapped_labels = [textwrap.fill(label, width=18) for label in labels]
    ax.set_xticklabels(wrapped_labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    value_labels = [value_format.format(value) for value in values]
    ax.bar_label(bars, labels=value_labels, padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def scatter_model_size_vs_emissions(
    summary: Dict[str, Dict[str, float]],
    filepath: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = []
    emissions = []
    labels = []
    for key, data in summary.items():
        sizes.append(data["size_mb"])
        emissions.append(data["avg_emissions"])
        labels.append(data["label"])
    ax.scatter(sizes, emissions, s=160, color="#d95f02")
    ax.margins(x=0.15, y=0.25)
    ax.set_title("Average Emissions vs Model Size")
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Average Emissions (kg CO₂eq)")
    ax.grid(True, linestyle="--", alpha=0.4)
    for x, y, label in zip(sizes, emissions, labels):
        wrapped = textwrap.fill(label, width=24)
        ax.annotate(
            wrapped,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#cccccc"),
        )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def summary_table_png(
    summary: Dict[str, Dict[str, float]],
    ordered_keys: List[str],
    filepath: Path,
) -> None:
    headers = ["Model", "Avg CO2 (kg)", "Avg Latency (s)", "Model Size (MB)", "Pass Rate"]
    rows = []
    for key in ordered_keys:
        data = summary[key]
        rows.append(
            [
                textwrap.fill(data["label"], width=32),
                f"{data['avg_emissions']:.2e}",
                f"{data['avg_inference']:.2f}",
                f"{data['size_mb']:.0f}",
                f"{data['pass_rate']:.0%}",
            ]
        )

    fig_height = 1.2 + 0.6 * len(rows)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.5, 0.14, 0.16, 0.16, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_fontsize(11)
            cell.set_text_props(weight="bold")
        if col_idx == 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = read_rows()
    summary = summarise_by_model(rows)
    ordered_keys = sorted(summary.keys(), key=lambda k: summary[k]["avg_emissions"])

    emissions = [summary[k]["avg_emissions"] for k in ordered_keys]
    display_labels = [summary[k]["label"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        emissions,
        title="Average Emissions by Model",
        ylabel="Emissions (kg CO₂eq)",
        filepath=FIGURES_DIR / "emissions_by_model_key.png",
        value_format="{:.2e}",
        display_labels=display_labels,
    )

    inference = [summary[k]["avg_inference"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        inference,
        title="Average Inference Time by Model",
        ylabel="Time (seconds)",
        filepath=FIGURES_DIR / "inference_time_by_model.png",
        value_format="{:.2f}",
        display_labels=display_labels,
    )

    sizes = [summary[k]["size_mb"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        sizes,
        title="Model Size by Model Key",
        ylabel="Size (MB)",
        filepath=FIGURES_DIR / "model_size_by_model.png",
        value_format="{:.0f}",
        display_labels=display_labels,
    )

    accuracy = [summary[k]["pass_rate"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        accuracy,
        title="Accuracy Rate by Model",
        ylabel="Pass Rate",
        filepath=FIGURES_DIR / "accuracy_by_model.png",
        value_format="{:.0%}",
        display_labels=display_labels,
    )

    scatter_model_size_vs_emissions(summary, FIGURES_DIR / "model_size_vs_emissions.png")
    summary_table_png(summary, ordered_keys, FIGURES_DIR / "model_summary_table.png")


if __name__ == "__main__":
    main()
