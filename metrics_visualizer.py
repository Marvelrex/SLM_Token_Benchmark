# viz_green_prompt_metrics.py
# Usage: run in a Python environment with matplotlib and pandas installed.

import os
import re
import math
import textwrap
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1) Config
# -------------------------
CSV_PATH_A = "green_prompt_results_three_models.csv"            # primary (e.g., Local)
CSV_PATH_B = "cloud_mistral7b_quant_benchmark.csv"              # optional second csv (e.g., Cloud)
LABEL_A = "Local"
LABEL_B = "Cloud"

OUT_DIR = "./green_prompt_figs"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# 2) Helpers
# -------------------------

PROMPT_LABELS = {
    "complete_prompt_text": "Complete",
    "concise_prompt_text": "Concise",
    "ultra_concise_prompt_text": "Ultra",
}
def _pretty(label: str, max_width: int = 14) -> str:
    lbl = PROMPT_LABELS.get(str(label), str(label).replace("_", " "))
    return "\n".join(textwrap.wrap(lbl, width=max_width)) or lbl

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"None of the candidate columns found: {candidates}")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map various possible headers to a canonical schema:
    Model, Quantization, Prompt, Compute, T_in, T_out, T_total, Inference_Time_s, Accuracy
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
        "Accuracy": ["Accuracy", "Acc", "PassFail", "P_F", "Passed"],
    }

    canon = {}
    for canonical, candidates in colmap.items():
        try:
            canon[canonical] = find_col(df, candidates)
        except KeyError:
            if canonical == "Hardware":
                canon[canonical] = None
            else:
                raise

    out = pd.DataFrame()
    out["Model"] = df[canon["Model"]].astype(str)
    out["Quantization"] = df[canon["Quantization"]].astype(str)
    out["Prompt"] = df[canon["Prompt"]].astype(str)
    out["Compute"] = (df[canon["Hardware"]].astype(str) if canon["Hardware"] else "Local")

    # Vectorized numeric coercion
    for src, dst in [
        (canon["T_in"], "T_in"),
        (canon["T_out"], "T_out"),
        (canon["T_total"], "T_total"),
        (canon["Inference_Time_s"], "Inference_Time_s"),
    ]:
        out[dst] = pd.to_numeric(df[src], errors="coerce")

    # Accuracy normalization: robust Pass/Fail mapping
    acc = df[canon["Accuracy"]].astype(str).str.strip().str.lower()
    out["Accuracy_binary"] = np.where(acc.str.startswith("pass"), 1,
                               np.where(acc.str.startswith("fail"), 0, np.nan))
    return out

def grouped_averages(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Average over epochs per (Model, Quantization, Prompt, Compute)."""
    grp_cols = ["Model", "Quantization", "Prompt", "Compute"]
    metrics = ["T_in", "T_out", "T_total", "Inference_Time_s", "Accuracy_binary"]
    g = (df_norm.groupby(grp_cols, dropna=False)[metrics]
                .mean(numeric_only=True)
                .reset_index()
                .rename(columns={
                    "T_in": "Tin_avg",
                    "T_out": "Tout_avg",
                    "T_total": "Ttotal_avg",
                    "Inference_Time_s": "Time_avg",
                    "Accuracy_binary": "Accuracy_pct",
                }))
    g["Accuracy_pct"] = g["Accuracy_pct"] * 100.0
    return g

def _bar_positions(n_groups, n_bars, bar_width=0.22, group_spacing=0.55):
    centers = np.arange(n_groups) * (n_bars * bar_width + group_spacing)
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2.0) * bar_width
    return centers, [centers + off for off in offsets]

def _quant_sort_key(q: str) -> int:
    s = str(q).lower()
    if "fp16" in s or "f16" in s: return 0
    if "8" in s and ("q8" in s or "8-bit" in s or "int8" in s): return 1
    if "4" in s and ("q4" in s or "4-bit" in s or "int4" in s): return 2
    return 3

# -------------------------
# 3) Plots
# -------------------------

def plot_grouped_bars(df_avg: pd.DataFrame, value_col: str, ylabel: str, title: str, outfile: str):
    models = sorted(df_avg["Model"].unique())
    prompts = sorted(df_avg["Prompt"].unique())
    quants  = sorted(df_avg["Quantization"].unique(), key=_quant_sort_key)

    pretty_prompts = [_pretty(p) for p in prompts]

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(max(6.0*n_models, 6.0), 3.8), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        sub = df_avg[df_avg["Model"] == model]
        data = np.full((len(prompts), len(quants)), np.nan)

        for i, p in enumerate(prompts):
            for j, q in enumerate(quants):
                v = sub[(sub["Prompt"] == p) & (sub["Quantization"] == q)][value_col]
                if len(v): data[i, j] = float(v.values[0])

        n_groups, n_bars = len(prompts), len(quants)
        centers, bar_x = _bar_positions(n_groups, n_bars)

        for j in range(n_bars):
            ax.bar(bar_x[j], data[:, j], width=0.22, label=quants[j])

        ax.set_xticks(centers)
        ax.set_xticklabels(pretty_prompts)
        for t in ax.get_xticklabels(): t.set_rotation(10); t.set_ha("center")
        ax.set_title(model)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.05)

        for j in range(n_bars):
            for x, y in zip(bar_x[j], data[:, j]):
                if not (isinstance(y, float) and math.isnan(y)):
                    ax.text(x, y, (f"{y:.0f}" if abs(y) >= 10 else f"{y:.2f}"),
                            ha="center", va="bottom", fontsize=8)

        ax.legend(title="Quantization", fontsize=8, title_fontsize=9)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(outfile, dpi=200)
    plt.close()

def plot_accuracy_heatmaps(df_avg: pd.DataFrame, outfile: str):
    models  = sorted(df_avg["Model"].unique())
    prompts = sorted(df_avg["Prompt"].unique())
    quants  = sorted(df_avg["Quantization"].unique(), key=_quant_sort_key)
    pretty_prompts = [_pretty(p) for p in prompts]

    fig, axes = plt.subplots(1, len(models), figsize=(max(4.8*len(models), 10), 4.4), constrained_layout=True)
    if len(models) == 1: axes = [axes]
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.08, hspace=0.15)

    cmap = plt.colormaps.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    im_last = None

    for ax, model in zip(axes, models):
        sub = df_avg[df_avg["Model"] == model]
        mat = np.full((len(quants), len(prompts)), np.nan)
        for r, q in enumerate(quants):
            for c, p in enumerate(prompts):
                s = sub[(sub["Quantization"] == q) & (sub["Prompt"] == p)]["Accuracy_pct"]
                if len(s): mat[r, c] = float(s.values[0])

        im_last = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", vmin=0, vmax=100, cmap=cmap)
        ax.set_xticks(np.arange(len(prompts))); ax.set_xticklabels(pretty_prompts, rotation=10, ha="center")
        ax.set_yticks(np.arange(len(quants)));  ax.set_yticklabels(quants)
        ax.set_xlabel("Prompt"); ax.set_ylabel("Quantization")
        ax.set_title(model, fontsize=11, pad=12)

        for r in range(len(quants)):
            for c in range(len(prompts)):
                v = mat[r, c]
                if not (isinstance(v, float) and np.isnan(v)):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im_last, ax=axes, location="right", shrink=0.9, pad=0.03)
    cbar.set_label("Accuracy (%)")
    fig.suptitle("Accuracy (%) by Quantization and Prompt", fontsize=13, y=1.02)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_accuracy_per_time_bar(df_avg: pd.DataFrame, model_name: str, by_prompt: bool, save_path: str):
    sub = df_avg[df_avg["Model"] == str(model_name)].copy()
    if sub.empty: raise ValueError(f"No rows found for model: {model_name}")

    sub["acc_per_time"] = np.where((sub["Time_avg"] > 0) & np.isfinite(sub["Time_avg"]),
                                   sub["Accuracy_pct"] / sub["Time_avg"], np.nan)

    prompts = sorted(sub["Prompt"].unique())
    quants  = sorted(sub["Quantization"].unique(), key=_quant_sort_key)

    if by_prompt:
        x_groups, series = prompts, quants
        x_label, legend_title = "Prompt", "Quantization"
        x_pretty = [_pretty(p) for p in x_groups]
    else:
        x_groups, series = quants, prompts
        x_label, legend_title = "Quantization", "Prompt"
        x_pretty = x_groups

    n_groups, n_series = len(x_groups), len(series)
    data = np.full((n_groups, n_series), np.nan)
    for i, g in enumerate(x_groups):
        for j, s in enumerate(series):
            row = (sub[(sub["Prompt"] == g) & (sub["Quantization"] == s)] if by_prompt
                   else sub[(sub["Quantization"] == g) & (sub["Prompt"] == s)])
            if len(row): data[i, j] = float(row["acc_per_time"].iloc[0])

    fig, ax = plt.subplots(figsize=(max(6.5, 1.6*n_groups+2.5), 3.8))
    centers, bar_x = _bar_positions(n_groups, n_series)
    for j in range(n_series):
        ax.bar(bar_x[j], data[:, j], width=0.22, label=str(series[j]))

    ax.set_xticks(centers); ax.set_xticklabels(x_pretty)
    for t in ax.get_xticklabels(): t.set_rotation(10); t.set_ha("center")
    ax.set_ylabel("Accuracy % per second (↑ better)")
    ax.set_title(f"{model_name} — Accuracy ÷ Time ({'by Prompt' if by_prompt else 'by Quantization'})")
    ax.grid(axis="y", alpha=0.3); ax.margins(x=0.05)

    for j in range(n_series):
        for x, y in zip(bar_x[j], data[:, j]):
            if np.isfinite(y): ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    ax.legend(title=legend_title, fontsize=8, title_fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_tokens_per_accuracy(df_avg: pd.DataFrame, save_path: str, y_max: float = 20.0, yticks_step: float = 5.0):
    required = {"Model","Prompt","Quantization","Ttotal_avg","Accuracy_pct"}
    missing = required - set(df_avg.columns)
    if missing: raise ValueError(f"df_avg missing columns: {missing}")

    g = (df_avg.groupby(["Model","Prompt"], dropna=False)
                .agg(Ttotal_avg=("Ttotal_avg","mean"), Accuracy_pct=("Accuracy_pct","mean"))
                .reset_index())

    eps = 1e-9
    g["tokens_per_accuracy"] = g["Ttotal_avg"] / (g["Accuracy_pct"] + eps)

    models = sorted(g["Model"].unique())
    present_prompts = [p for p in ["complete_prompt_text","concise_prompt_text","ultra_concise_prompt_text"]
                       if p in set(g["Prompt"])] or sorted(g["Prompt"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(max(8.0, 4.5*len(models)), 3.8),
                             constrained_layout=True, sharey=True)
    if len(models) == 1: axes = [axes]

    for ax, model in zip(axes, models):
        sub = g[g["Model"] == model]
        vals, labels = [], []
        for p in present_prompts:
            row = sub[sub["Prompt"] == p]
            if len(row): vals.append(float(row["tokens_per_accuracy"].iloc[0])); labels.append(_pretty(p))

        x = np.arange(len(vals))
        ax.bar(x, vals)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Tokens per 1% Accuracy (↓ better)")
        ax.set_title(model); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.0, y_max); ax.set_yticks(np.arange(0.0, y_max + 1e-9, yticks_step))
        for xi, h in zip(x, vals): ax.text(xi, min(h, y_max), f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Efficiency: Total Tokens / Accuracy (%)", fontsize=12)
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def compute_epoch_variances_from_norm(df_norm: pd.DataFrame) -> pd.DataFrame:
    grp = ["Model", "Prompt", "Quantization", "Compute"]
    var_df = (df_norm.groupby(grp, dropna=False)
                    .agg(n_obs=("T_out", "size"),
                         Tout_var=("T_out", lambda x: pd.to_numeric(x, errors="coerce").var(ddof=1)),
                         Time_var=("Inference_Time_s", lambda x: pd.to_numeric(x, errors="coerce").var(ddof=1)))
                    .reset_index())
    return var_df

def plot_model_variance_bars_from_var(var_df: pd.DataFrame, model_name: str, save_path: str,
                                      x_by_prompt: bool = True, annotate: bool = True, logy: bool = False):
    sub = var_df[var_df["Model"] == model_name].copy()
    if sub.empty: raise ValueError(f"No variance rows for model: {model_name}")

    prompts = [p for p in PROMPT_LABELS if p in set(sub["Prompt"])] or sorted(sub["Prompt"].unique())
    quants  = sorted(sub["Quantization"].unique(), key=_quant_sort_key)

    pv_tout = sub.pivot(index="Quantization", columns="Prompt", values="Tout_var").reindex(index=quants, columns=prompts)
    pv_time = sub.pivot(index="Quantization", columns="Prompt", values="Time_var").reindex(index=quants, columns=prompts)

    if x_by_prompt:
        x_labels = [_pretty(p) for p in pv_tout.columns]; series = list(pv_tout.index)
        data_tout = pv_tout.to_numpy().T; data_time = pv_time.to_numpy().T
        legend_title = "Quantization"; xlabel = "Prompt"
    else:
        x_labels = list(pv_tout.index); series = [_pretty(p) for p in pv_tout.columns]
        data_tout = pv_tout.to_numpy(); data_time = pv_time.to_numpy()
        legend_title = "Prompt"; xlabel = "Quantization"

    n_x, n_series = data_tout.shape
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 1.6*n_x + 3.0), 5.6), constrained_layout=True)

    def _grouped(ax, data, title, ylabel):
        width = 0.85 / max(1, n_series)
        xs = np.arange(n_x)
        for j in range(n_series):
            ax.bar(xs + (j - (n_series - 1)/2) * width, data[:, j], width=width, label=str(series[j]))
        ax.set_xticks(xs); ax.set_xticklabels(x_labels)
        for t in ax.get_xticklabels(): t.set_rotation(10); t.set_ha("center")
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if logy: ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3); ax.margins(x=0.02)
        if annotate:
            for j in range(n_series):
                for xi, y in enumerate(data[:, j]):
                    if np.isfinite(y): ax.text(xs[xi] + (j-(n_series-1)/2)*width, y, f"{y:.3g}",
                                               ha="center", va="bottom", fontsize=8)
        ax.legend(title=legend_title, fontsize=8, title_fontsize=9)

    _grouped(axes[0], data_tout, f"{model_name} — Var(T_out) across epochs", "Variance of T_out (tokens)")
    _grouped(axes[1], data_time, "Var(Inference Time) across epochs", "Variance of Time (s)")
    plt.savefig(save_path, dpi=200, bbox_inches="tight"); plt.close()

def plot_inference_time_compare_two_csvs(csv_a: str, csv_b: str, label_a: str, label_b: str, outfile: str):
    """
    Compare average inference time between two CSVs for the SAME (Model, Quantization, Prompt).
    Handles name mismatches:
      - Model: colon vs hyphen vs dots (all stripped to alphanumerics)
      - Quantization: fp16 <-> unquantized, q8_0 <-> 8bit, q4_0 <-> 4bit
      - Prompt: maps to Complete / Concise / Ultra
    Accepts either 'Inference_Time_s' or 'InferenceTime' as the time column.
    Produces grouped bars (A vs B) with short two-line tick labels (Quant on top, Prompt below).
    """
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -------- Canon helpers --------
    def _canon_model(s: str) -> str:
        return re.sub(r'[^a-z0-9]+', '', str(s).strip().lower())  # remove : - . etc.

    def _canon_soft(s: str) -> str:
        return re.sub(r'[\s\-_]+', '_', str(s).strip().lower())   # collapse space/hyphen/underscore

    QUANT_MAP = {
        "fp16": {"fp16", "f16", "float16", "unquantized", "full_precision", "fullprec"},
        "q8_0": {"q8_0", "q8", "8bit", "8_bit", "int8"},
        "q4_0": {"q4_0", "q4", "4bit", "4_bit", "int4"},
    }
    PROMPT_MAP = {
        "complete_prompt_text": {"complete", "complete_prompt_text"},
        "concise_prompt_text": {"concise", "concise_prompt_text"},
        "ultra_concise_prompt_text": {"ultra", "ultra_concise", "ultra_concise_prompt_text"},
    }
    Q_SHORT = {"fp16": "FP16", "q8_0": "Q8", "q4_0": "Q4", "8bit": "Q8", "4bit": "Q4", "unquantized": "FP16"}
    P_SHORT = {"complete_prompt_text": "Complete", "concise_prompt_text": "Concise", "ultra_concise_prompt_text": "Ultra"}

    def _map_from_sets(val: str, mapping: dict) -> str:
        v = _canon_soft(val)
        for k, vs in mapping.items():
            if v in vs:
                return k
        return v  # fallback to normalized value

    def _pick_time_col(df: pd.DataFrame) -> str:
        lower = {c.lower(): c for c in df.columns}
        if "inference_time_s" in lower: return lower["inference_time_s"]
        if "inferencetime"   in lower: return lower["inferencetime"]
        raise KeyError("Neither 'Inference_Time_s' nor 'InferenceTime' found in file.")

    def _load_agg(path: str, source_label: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        for col in ["Model", "Quantization", "Prompt"]:
            if col not in raw.columns:
                raise KeyError(f"{path}: missing column '{col}'")
        tcol = _pick_time_col(raw)
        agg = (raw.groupby(["Model", "Quantization", "Prompt"], dropna=False)[tcol]
                  .mean()
                  .reset_index()
                  .rename(columns={tcol: "Time_avg"}))

        # Canonical triplet for robust join
        agg["Model_key"]  = agg["Model"].map(_canon_model)
        agg["Quant_key"]  = agg["Quantization"].map(lambda x: _map_from_sets(x, QUANT_MAP))
        agg["Prompt_key"] = agg["Prompt"].map(lambda x: _map_from_sets(x, PROMPT_MAP))
        agg["source"] = source_label
        return agg

    A = _load_agg(csv_a, label_a)
    B = _load_agg(csv_b, label_b)

    # Join on canonical keys
    keys = ["Model_key", "Quant_key", "Prompt_key"]
    common = pd.merge(A[keys], B[keys], on=keys, how="inner").drop_duplicates()
    if common.empty:
        setA = set(map(tuple, A[keys].itertuples(index=False, name=None)))
        setB = set(map(tuple, B[keys].itertuples(index=False, name=None)))
        print("[compare] No overlap after normalization.")
        print("  Examples only in A:", list(setA - setB)[:12])
        print("  Examples only in B:", list(setB - setA)[:12])
        raise ValueError("No common (Model, Quantization, Prompt) entries between the two CSVs.")

    A_common = pd.merge(common, A, on=keys, how="left")
    B_common = pd.merge(common, B, on=keys, how="left")
    df_plot = pd.concat([A_common, B_common], ignore_index=True)

    # Friendly group labels (Quant on 1st line, Prompt on 2nd)
    def _two_line(q_raw: str, p_raw: str, q_key: str, p_key: str) -> str:
        q_disp = Q_SHORT.get(q_key, q_raw)
        p_disp = P_SHORT.get(p_key, p_raw)
        return f"{q_disp}\n{p_disp}"

    # One subplot per canonical model, with a readable title picked from A if possible
    model_title = (A_common.drop_duplicates(keys)[keys + ["Model"]].set_index(keys)["Model"]).to_dict()

    models = sorted(df_plot["Model_key"].unique())
    fig, axes = plt.subplots(
        1, len(models),
        figsize=(max(9.5, 6.5 * len(models)), 5.2),
        sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    # plotting in model loop
    for ax, mkey in zip(axes, models):
        sub = df_plot[df_plot["Model_key"] == mkey].copy()

        # stable ordering: fp16 → q8_0 → q4_0; Complete → Concise → Ultra
        q_order = {"fp16": 0, "q8_0": 1, "q4_0": 2}
        p_order = {"complete_prompt_text": 0, "concise_prompt_text": 1, "ultra_concise_prompt_text": 2}
        sub["q_ord"] = sub["Quant_key"].map(lambda x: q_order.get(x, 99))
        sub["p_ord"] = sub["Prompt_key"].map(lambda x: p_order.get(x, 99))
        sub = sub.sort_values(["q_ord", "p_ord", "source"])

        # build two-line group labels in the sorted order
        sub["group"] = [
            _two_line(q_raw, p_raw, q_key, p_key)
            for q_raw, p_raw, q_key, p_key in zip(
                sub["Quantization"].astype(str),
                sub["Prompt"].astype(str),
                sub["Quant_key"].astype(str),
                sub["Prompt_key"].astype(str),
            )
        ]
        groups = sub["group"].drop_duplicates().tolist()
        x = np.arange(len(groups))
        width = 0.38

        piv = (sub.pivot_table(index="group", columns="source", values="Time_avg", aggfunc="mean")
                  .reindex(groups))

        ax.bar(x - width/2, piv.get(label_a, pd.Series(index=groups, dtype=float)).values, width, label=label_a)
        ax.bar(x + width/2, piv.get(label_b, pd.Series(index=groups, dtype=float)).values, width, label=label_b)

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        for t in ax.get_xticklabels():
            t.set_ha("center")
        ax.margins(x=0.02)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Average Inference Time (s)")

        # pretty model title
        title = next((v for (mk, _, _), v in model_title.items() if mk == mkey), mkey)
        ax.set_title(title, fontsize=12, pad=18)  # push the axes title down a bit
        ax.tick_params(axis="x", labelsize=10)

        # legend above, centered; title above legend
    # Shared legend ABOVE the axes, outside the plotting area
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    # Big title even higher, so it doesn't collide with the legend
    fig.suptitle("Inference Time Comparison (Same Model • Quant • Prompt)",
                 y=1.10, fontsize=16)

    # Reserve vertical room at the top for legend + title and pack the rest
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))  # left, bottom, right, top
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# 4) Main
# -------------------------
def main():
    # Load primary CSV
    df_raw = pd.read_csv(CSV_PATH_A)
    print(f"Loaded {len(df_raw)} rows from {CSV_PATH_A}")

    df_norm = normalize_columns(df_raw)
    if "Compute" not in df_norm.columns:
        df_norm["Compute"] = "Local"

    df_avg = grouped_averages(df_norm)

    # Save aggregated data
    agg_path = os.path.join(OUT_DIR, "aggregated_means_by_model_quant_prompt.csv")
    df_avg.to_csv(agg_path, index=False)
    print(f"Saved aggregated means to: {agg_path}")
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
    plot_tokens_per_accuracy(df_avg, os.path.join(OUT_DIR, "eff_tokens_per_accuracy.png"),
                             y_max=20.0, yticks_step=5.0)

    plot_accuracy_heatmaps(df_avg, outfile=os.path.join(OUT_DIR, "heatmap_accuracy.png"))

    # Per-model: quantization time-only bars
    for model_name in pd.read_csv(CSV_PATH_A)["Model"].unique():
        safe = sanitize_filename(model_name)
        plot_path = os.path.join(OUT_DIR, f"quant_inference_time_{safe}.png")
        # reuse raw CSV for true epoch avg by Quant|Prompt then avg across prompts
        def _plot(csv_path, model, path):
            df = pd.read_csv(csv_path)
            cols = {c.lower(): c for c in df.columns}
            Model   = cols.get('model', 'Model')
            Quant   = cols.get('quantization', 'Quantization')
            Prompt  = cols.get('prompt', 'Prompt')
            TimeCol = cols.get('inference_time_s', 'Inference_Time_s')
            AccCol  = cols.get('accuracy', 'Accuracy')

            sub = df[df[Model].astype(str) == str(model)].copy()
            if sub.empty: return
            acc = sub[AccCol].astype(str).str.strip().str.lower()
            sub['acc_bin'] = np.where(acc.str.startswith('pass'), 1,
                                 np.where(acc.str.startswith('fail'), 0, np.nan))
            gp = (sub.groupby([Quant, Prompt], dropna=False)
                      .agg(time_mean=(TimeCol, 'mean'),
                           acc_pct =('acc_bin', lambda x: x.mean()*100.0))
                      .reset_index())
            gq = (gp.groupby(Quant, dropna=False)
                     .agg(time_mean=('time_mean', 'mean'),
                          time_std =('time_mean', 'std'),
                          acc_pct  =('acc_pct', 'mean'))
                     .reset_index())
            gq = gq.sort_values(by=Quant, key=lambda s: s.map(_quant_sort_key))

            fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
            x = np.arange(len(gq))
            ax.bar(x, gq['time_mean'], yerr=gq['time_std'], capsize=3)
            ax.set_xticks(x); ax.set_xticklabels([str(q) for q in gq[Quant]])
            ax.set_ylabel("Average Inference Time (s)")
            ax.set_title(f"{model}: Avg Time by Quantization")
            ax.grid(axis='y', alpha=0.3)
            for xi, h, accv in zip(x, gq['time_mean'], gq['acc_pct']):
                if np.isfinite(h): ax.text(xi, h, f"{accv:.0f}%", ha='center', va='bottom', fontsize=9)
            fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
        _plot(CSV_PATH_A, model_name, plot_path)

    # Variance plots
    var_df = compute_epoch_variances_from_norm(df_norm)
    for m in sorted(var_df["Model"].dropna().unique()):
        safe = sanitize_filename(m)
        plot_model_variance_bars_from_var(
            var_df, m,
            save_path=os.path.join(OUT_DIR, f"{safe}_variance_bars.png"),
            x_by_prompt=True, annotate=True, logy=False,
        )

    # Accuracy per time (two orientations)
    for model_name in pd.read_csv(CSV_PATH_A)["Model"].unique():
        safe = sanitize_filename(model_name)
        plot_accuracy_per_time_bar(
            df_avg, model_name=model_name, by_prompt=False,
            save_path=os.path.join(OUT_DIR, f"{safe}_acc_per_time_by_quant.png"),
        )
        plot_accuracy_per_time_bar(
            df_avg, model_name=model_name, by_prompt=True,
            save_path=os.path.join(OUT_DIR, f"{safe}_acc_per_time_by_prompt.png"),
        )

    # --- NEW: compare two CSVs if the second exists ---
    if CSV_PATH_B and os.path.exists(CSV_PATH_B):
        compare_out = os.path.join(OUT_DIR, "inference_time_compare_local_vs_cloud.png")
        plot_inference_time_compare_two_csvs(CSV_PATH_A, CSV_PATH_B, LABEL_A, LABEL_B, compare_out)
        print(f"Saved comparison figure to: {compare_out}")
    else:
        print("Second CSV not found or not set; skipping cross-CSV comparison.")

if __name__ == "__main__":
    main()
