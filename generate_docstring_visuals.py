"""Generate PNG visualizations for docstring model metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

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
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = range(len(keys))
    bars = ax.bar(positions, values, color="#4f8bc9")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    labels = [value_format.format(value) for value in values]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
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
    scatter = ax.scatter(sizes, emissions, s=160, color="#d95f02")
    ax.set_title("Average Emissions vs Model Size")
    ax.set_xlabel("Model Size (MB)")
    ax.set_ylabel("Average Emissions (kg CO₂eq)")
    ax.grid(True, linestyle="--", alpha=0.4)
    for x, y, label in zip(sizes, emissions, labels):
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = read_rows()
    summary = summarise_by_model(rows)
    ordered_keys = sorted(summary.keys(), key=lambda k: summary[k]["avg_emissions"])

    emissions = [summary[k]["avg_emissions"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        emissions,
        title="Average Emissions by Model",
        ylabel="Emissions (kg CO₂eq)",
        filepath=FIGURES_DIR / "emissions_by_model_key.png",
        value_format="{:.2e}",
    )

    inference = [summary[k]["avg_inference"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        inference,
        title="Average Inference Time by Model",
        ylabel="Time (seconds)",
        filepath=FIGURES_DIR / "inference_time_by_model.png",
        value_format="{:.2f}",
    )

    sizes = [summary[k]["size_mb"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        sizes,
        title="Model Size by Model Key",
        ylabel="Size (MB)",
        filepath=FIGURES_DIR / "model_size_by_model.png",
        value_format="{:.0f}",
    )

    accuracy = [summary[k]["pass_rate"] for k in ordered_keys]
    bar_chart(
        ordered_keys,
        accuracy,
        title="Accuracy Rate by Model",
        ylabel="Pass Rate",
        filepath=FIGURES_DIR / "accuracy_by_model.png",
        value_format="{:.0%}",
    )

    scatter_model_size_vs_emissions(summary, FIGURES_DIR / "model_size_vs_emissions.png")


if __name__ == "__main__":
    main()
