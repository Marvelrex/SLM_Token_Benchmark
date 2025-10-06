# viz_green_prompt_metrics.py
# Usage: run in a Python environment with matplotlib and pandas installed.

import os
import math
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List
import textwrap

from matplotlib import gridspec

# -------------------------
# 1) Config
# -------------------------
CSV_PATH = "green_prompt_results_three_models.csv"
OUT_DIR = "./green_prompt_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 2) Helpers
# -------------------------

# --- add near the top, after imports ---


# Put near your imports (or reuse if you already added these earlier)



PROMPT_LABELS = {
    "complete_prompt_text": "Complete",
    "concise_prompt_text": "Concise",
    "ultra_concise_prompt_text": "Ultra",
}
def _pretty(label: str, max_width: int = 14) -> str:
    lbl = PROMPT_LABELS.get(label, label.replace("_", " "))
    return "\n".join(textwrap.wrap(lbl, width=max_width)) or lbl


def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return first matching column name (case-insensitive) from candidates."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of the candidate columns found: {candidates}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map various possible headers to a canonical schema:
    Model, Quantization, Prompt, Hardware/Compute, T_in, T_out, T_total, Inference_Time_s, Accuracy
    """
    colmap: Dict[str, List[str]] = {
        "Model": ["model", "Model", "model_name"],
        "Quantization": ["quantization", "Quantization", "quant", "dtype", "bits"],
        "Prompt": ["prompt", "Prompt", "prompt_name"],
        "Hardware": ["hardware", "Hardware", "compute", "Compute", "cloud_local", "CloudLocal", "Cloud/Local"],
        "T_in": ["T_in", "Tin", "InputTokenCount", "Input_Tokens", "Input_Token_Count"],
        "T_out": ["T_out", "Tout", "OutputTokenCount", "Output_Tokens", "Output_Token_Count"],
        "T_total": ["T_total", "Ttotal", "TotalTokenCount", "Total_Tokens", "Total_Token_Count"],
        "Inference_Time_s": ["Inference_Time_s", "InferenceTime", "Time_s", "T(s)", "Inference_Time"],
        "Accuracy": ["Accuracy", "Acc", "PassFail", "P_F", "Passed"]
    }

    canon = {}
    for canonical, candidates in colmap.items():
        try:
            canon[canonical] = find_col(df, candidates)
        except KeyError:
            # Only Hardware/Compute is optional
            if canonical == "Hardware":
                canon[canonical] = None
            else:
                raise

    # Ensure required cols exist
    req = ["Model", "Quantization", "Prompt", "T_in", "T_out", "T_total", "Inference_Time_s", "Accuracy"]
    for r in req:
        if canon.get(r) is None:
            raise KeyError(f"Missing required column mapping for: {r}")

    # Build a normalized frame
    out = pd.DataFrame()
    out["Model"] = df[canon["Model"]].astype(str)
    out["Quantization"] = df[canon["Quantization"]].astype(str)
    out["Prompt"] = df[canon["Prompt"]].astype(str)

    if canon["Hardware"] is not None:
        out["Compute"] = df[canon["Hardware"]].astype(str)
    else:
        out["Compute"] = "Local"  # per your note: all entries are Local

    # Numeric conversions with safe coercion
    for k_src, k_dst in [
        (canon["T_in"], "T_in"),
        (canon["T_out"], "T_out"),
        (canon["T_total"], "T_total"),
        (canon["Inference_Time_s"], "Inference_Time_s"),
    ]:
        out[k_dst] = pd.to_numeric(df[k_src], errors="coerce")

    # Accuracy normalization: map to 0/1
    acc_raw = df[canon["Accuracy"]]

    def acc_to_binary(x):
        s = str(x).strip().lower()
        if s.startswith('pass'): return 1
        if s.startswith('fail'): return 0
        return np.nan
    out["Accuracy_binary"] = acc_raw.map(acc_to_binary)

    return out

def grouped_averages(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Average over 5 epochs per (Model, Quantization, Prompt, Compute)."""
    grp_cols = ["Model", "Quantization", "Prompt", "Compute"]
    metrics = ["T_in", "T_out", "T_total", "Inference_Time_s", "Accuracy_binary"]
    g = (
        df_norm
        .groupby(grp_cols, dropna=False)[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={
            "T_in": "Tin_avg",
            "T_out": "Tout_avg",
            "T_total": "Ttotal_avg",
            "Inference_Time_s": "Time_avg",
            "Accuracy_binary": "Accuracy_pct"
        })
    )
    g["Accuracy_pct"] = g["Accuracy_pct"] * 100.0
    return g

def _bar_positions(n_groups, n_bars, bar_width=0.2, group_spacing=0.4):
    """
    Compute x positions for grouped bars.
    Returns group centers and a list of arrays for each bar's x positions.
    """
    group_centers = np.arange(n_groups) * (n_bars * bar_width + group_spacing)
    offsets = (np.arange(n_bars) - (n_bars - 1)/2.0) * bar_width
    bar_x = [group_centers + off for off in offsets]
    return group_centers, bar_x

# --- replace your plot_grouped_bars() with this version ---
def plot_grouped_bars(df_avg: pd.DataFrame, value_col: str, ylabel: str, title: str, outfile: str):
    """
    Grouped bars: X = Prompt, grouped by Quantization, one figure per Model.
    Uses shorter/wrapped x-tick labels and small rotation to prevent overlap.
    """
    models = sorted(df_avg["Model"].unique())
    prompts = sorted(df_avg["Prompt"].unique())
    quants  = sorted(df_avg["Quantization"].unique())

    # Make pretty labels once
    pretty_prompts = [_pretty(p) for p in prompts]

    # Wider figure if there are many/long labels
    n_models = len(models)
    fig_height = max(3.8, 3.0)
    fig_width  = max(6.0 * n_models, 4.5 * n_models)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        sub = df_avg[df_avg["Model"] == model]

        # Build matrix: rows=prompts, cols=quants
        data = np.full((len(prompts), len(quants)), np.nan)
        for i, p in enumerate(prompts):
            for j, q in enumerate(quants):
                val = sub[(sub["Prompt"] == p) & (sub["Quantization"] == q)][value_col]
                if len(val):
                    data[i, j] = float(val.values[0])

        # Compute grouped bar positions
        n_groups = len(prompts)
        n_bars   = len(quants)
        bar_width     = 0.22
        group_spacing = 0.55
        group_centers, bar_x = _bar_positions(n_groups, n_bars, bar_width, group_spacing)

        # Draw bars
        for j in range(n_bars):
            ax.bar(bar_x[j], data[:, j], width=bar_width, label=quants[j])

        # X ticks with pretty (wrapped) labels
        ax.set_xticks(group_centers)
        ax.set_xticklabels(pretty_prompts)

        # Light rotation + right alignment helps when labels are multi-line
        for tick in ax.get_xticklabels():
            tick.set_rotation(10)     # small angle; bump to 20–30 if needed
            tick.set_ha("center")     # center works well with multi-line labels

        ax.set_title(model)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.05)

        # Optional: numbers on bars
        for j in range(n_bars):
            for x, y in zip(bar_x[j], data[:, j]):
                if not (isinstance(y, float) and math.isnan(y)):
                    txt = f"{y:.0f}" if abs(y) >= 10 else f"{y:.2f}"
                    ax.text(x, y, txt, ha="center", va="bottom", fontsize=8)

        ax.legend(title="Quantization", fontsize=8, title_fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def plot_accuracy_heatmaps(df_avg: pd.DataFrame, outfile: str):
    import numpy as np
    import matplotlib.pyplot as plt

    models  = sorted(df_avg["Model"].unique())
    prompts = sorted(df_avg["Prompt"].unique())
    quants  = sorted(df_avg["Quantization"].unique())

    pretty_prompts = [_pretty(p) for p in prompts]

    n_models = len(models)
    fig_w = max(10, 4.8 * n_models)
    fig_h = 4.4
    fig, axes = plt.subplots(1, n_models, figsize=(fig_w, fig_h), constrained_layout=True)
    if n_models == 1:
        axes = [axes]

    # give more breathing room between rows/cols and reserve space for titles
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.08, hspace=0.15)

    cmap = plt.colormaps.get_cmap("viridis").copy()
    try:
        cmap.set_bad("#e6e6e6")
    except Exception:
        pass
    vmin, vmax = 0, 100
    im_last = None

    for ax, model in zip(axes, models):
        sub = df_avg[df_avg["Model"] == model]
        mat = np.full((len(quants), len(prompts)), np.nan, dtype=float)
        for r, q in enumerate(quants):
            for c, p in enumerate(prompts):
                s = sub[(sub["Quantization"] == q) & (sub["Prompt"] == p)]["Accuracy_pct"]
                if len(s):
                    mat[r, c] = float(s.values[0])

        im_last = ax.imshow(np.ma.masked_invalid(mat), aspect="auto",
                            vmin=vmin, vmax=vmax, cmap=cmap)

        ax.set_xticks(np.arange(len(prompts)))
        ax.set_xticklabels(pretty_prompts, rotation=10, ha="center")
        ax.set_yticks(np.arange(len(quants)))
        ax.set_yticklabels(quants)
        ax.set_xlabel("Prompt")
        ax.set_ylabel("Quantization")

        # SHORT subplot title (model only), smaller font, more pad
        ax.set_title(model, fontsize=11, pad=12)

        # annotate cells
        for r in range(len(quants)):
            for c in range(len(prompts)):
                v = mat[r, c]
                if not (isinstance(v, float) and np.isnan(v)):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9)

    # single shared colorbar
    cbar = fig.colorbar(im_last, ax=axes, location="right", shrink=0.9, pad=0.03)
    cbar.set_label("Accuracy (%)")

    # GLOBAL title; place it a bit higher to avoid any crowding
    fig.suptitle("Accuracy (%) by Quantization and Prompt", fontsize=13, y=1.02)

    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- helper: robust map for Pass/Fail strings (handles "Fail: ...")
def _acc_to_bin(x):
    s = str(x).strip().lower()
    if s.startswith('pass'): return 1
    if s.startswith('fail'): return 0
    return np.nan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_quant_time_bar(csv_path: str, model_name: str, save_path: str | None = None):
    # Load & basic column mapping
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    Model   = cols.get('model', 'Model')
    Quant   = cols.get('quantization', 'Quantization')
    Prompt  = cols.get('prompt', 'Prompt')
    TimeCol = cols.get('inference_time_s', 'Inference_Time_s')
    AccCol  = cols.get('accuracy', 'Accuracy')

    sub = df[df[Model].astype(str) == str(model_name)].copy()
    if sub.empty:
        raise ValueError(f"No rows for model: {model_name}")

    # Accuracy -> 0/1 (handles "Pass" and "Fail: ...")
    acc = sub[AccCol].astype(str).str.strip().str.lower()
    sub['acc_bin'] = np.where(acc.str.startswith('pass'), 1,
                        np.where(acc.str.startswith('fail'), 0, np.nan))

    # Average the 5 epochs at (Quant, Prompt), then average across prompts per Quant
    gp = (sub.groupby([Quant, Prompt], dropna=False)
              .agg(time_mean=(TimeCol, 'mean'),
                   acc_pct =('acc_bin', lambda x: x.mean()*100.0))
              .reset_index())

    gq = (gp.groupby(Quant, dropna=False)
             .agg(time_mean=('time_mean', 'mean'),
                  time_std =('time_mean', 'std'),
                  acc_pct  =('acc_pct', 'mean'))
             .reset_index())

    # Order quantization sensibly (unquantized/fp16 -> q8 -> q4)
    def _sort_key(q):
        s = str(q).lower()
        if '4' in s and ('q4' in s or '4-bit' in s or 'int4' in s): return 2
        if '8' in s and ('q8' in s or '8-bit' in s or 'int8' in s): return 1
        return 0
    gq = gq.sort_values(by=Quant, key=lambda s: s.map(_sort_key))

    # --- Bar plot only ---
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    x = np.arange(len(gq))
    bars = ax.bar(x, gq['time_mean'], yerr=gq['time_std'], capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(q) for q in gq[Quant]])
    ax.set_ylabel("Average Inference Time (s)")
    ax.set_title(f"{model_name}: Avg Time by Quantization")
    ax.grid(axis='y', alpha=0.3)

    # Show accuracy % above bars
    for xi, h, acc in zip(x, gq['time_mean'], gq['acc_pct']):
        if np.isfinite(h):
            ax.text(xi, h, f"{acc:.0f}%", ha='center', va='bottom', fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -------------------------
# 3) Main
# -------------------------
def main():
    # Load CSV
    df_raw = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df_raw)} rows from {CSV_PATH}")

    # Normalize schema
    df_norm = normalize_columns(df_raw)

    # Fill missing compute label with 'Local' if any
    if "Compute" not in df_norm.columns:
        df_norm["Compute"] = "Local"

    # Aggregate over epochs
    df_avg = grouped_averages(df_norm)

    # Save aggregated data
    agg_path = os.path.join(OUT_DIR, "aggregated_means_by_model_quant_prompt.csv")
    df_avg.to_csv(agg_path, index=False)
    print(f"Saved aggregated means to: {agg_path}")

    # Print small preview
    print(df_avg.head(10))

    # Plots
    plot_grouped_bars(
        df_avg, value_col="Ttotal_avg",
        ylabel="Avg Total Tokens",
        title="Average Total Token Count by Configuration",
        outfile=os.path.join(OUT_DIR, "bar_total_tokens.png"),
    )

    plot_grouped_bars(
        df_avg, value_col="Time_avg",
        ylabel="Avg Inference Time (s)",
        title="Average Inference Time by Configuration",
        outfile=os.path.join(OUT_DIR, "bar_inference_time.png"),
    )

    plot_accuracy_heatmaps(df_avg, outfile=os.path.join(OUT_DIR, "heatmap_accuracy.png"))
    for model_name in pd.read_csv(CSV_PATH)["Model"].unique():
        safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', model_name)
        plot_quant_time_bar(CSV_PATH, model_name=model_name,save_path=f"quant_inference_time_{safe_name}.png")


if __name__ == "__main__":
    main()
