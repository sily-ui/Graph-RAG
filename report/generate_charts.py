"""Generate academic-style PNG figures for the Graph-RAG assessment report.

Reads JSONL evaluation results from eval/reports_smd_full/ (primary, real SMD)
and eval/reports_full_v3/ (synthetic, comparison), aggregates by baseline and
hop count, and renders publication-quality matplotlib figures into
report/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
SMD_DIR = ROOT / "eval" / "reports_smd_full"
SYN_DIR = ROOT / "eval" / "reports_full_v3"
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Academic palette: distinct, color-blind safe (Wong 2011 palette + tweaks)
COLORS = {
    "B1_NaiveRAG": "#999999",        # neutral grey = no-graph baseline
    "B2_GraphitiDefault": "#E69F00", # orange = stock Graphiti
    "B3_NoTemporal": "#56B4E9",      # sky blue = ablation
    "B4_Full_GraphRAG": "#0072B2",   # deep blue = our full system
}
LABELS = {
    "B1_NaiveRAG": "B1 NaiveRAG",
    "B2_GraphitiDefault": "B2 Graphiti Default",
    "B3_NoTemporal": "B3 NoTemporal (ablation)",
    "B4_Full_GraphRAG": "B4 Full GraphRAG (ours)",
}
BASELINE_ORDER = ["B1_NaiveRAG", "B2_GraphitiDefault", "B3_NoTemporal", "B4_Full_GraphRAG"]

METRIC_FIELDS = [
    "path_error_rate", "hallucination_rate_overall", "hallucination_rate_per_hop",
    "recall", "precision", "temporal_accuracy", "provenance_completeness",
    "entity_recall", "entity_precision", "relation_recall", "pipeline_f1",
    "r_score", "ar_score", "em",
]

# Configure matplotlib for academic style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _recompute_hallucination(case: dict) -> tuple[float, float]:
    """Recompute hallucination rates from raw per-hop stats so that pre-fix
    and post-fix runs are directly comparable. The original `rate` field in
    each hop's stats only counted CONTRADICTED; the corrected rate counts
    (CONTRADICTED + UNSUPPORTED) / total, matching the v3 overall fix.
    Returns (overall, per_hop_mean). Falls back to stored values if stats missing.
    """
    stats = case.get("hallucination_stats")
    if not stats or not isinstance(stats, dict):
        v = case.get("hallucination_rate_overall", 0.0) or 0.0
        return float(v), float(case.get("hallucination_rate_per_hop", 0.0) or 0.0)
    total_claims = 0
    bad_claims = 0
    per_hop_rates = []
    for hop_key, hop_stat in stats.items():
        if not isinstance(hop_stat, dict):
            continue
        t = hop_stat.get("total", 0) or 0
        c = hop_stat.get("contradicted", 0) or 0
        u = hop_stat.get("unsupported", 0) or 0
        total_claims += t
        bad_claims += (c + u)
        if t > 0:
            per_hop_rates.append((c + u) / t)
    overall = (bad_claims / total_claims) if total_claims > 0 else 0.0
    per_hop = (sum(per_hop_rates) / len(per_hop_rates)) if per_hop_rates else 0.0
    return overall, per_hop


def aggregate(cases: List[dict]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return {baseline_name: {hop: {metric: mean}}} — but cases here are
    pre-grouped per file, so we just need {hop: {metric: mean}}.
    Actually we keep baseline label from filename externally.
    """
    by_hop: Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for c in cases:
        hop = c.get("hop_count")
        if hop is None:
            continue
        # recompute hallucination rates from raw stats so v2 (pre-fix) and v3
        # (post-fix) runs are directly comparable on the same metric definition
        hallu_overall, hallu_perhop = _recompute_hallucination(c)
        for m in METRIC_FIELDS:
            if m == "hallucination_rate_overall":
                by_hop[hop][m].append(hallu_overall)
            elif m == "hallucination_rate_per_hop":
                by_hop[hop][m].append(hallu_perhop)
            else:
                v = c.get(m)
                if isinstance(v, (int, float)):
                    by_hop[hop][m].append(float(v))
    # also aggregate all hops
    allhop: Dict[str, list] = defaultdict(list)
    for c in cases:
        hallu_overall, hallu_perhop = _recompute_hallucination(c)
        for m in METRIC_FIELDS:
            if m == "hallucination_rate_overall":
                allhop[m].append(hallu_overall)
            elif m == "hallucination_rate_per_hop":
                allhop[m].append(hallu_perhop)
            else:
                v = c.get(m)
                if isinstance(v, (int, float)):
                    allhop[m].append(float(v))
    out: Dict[int, Dict[str, float]] = {}
    for hop, mdict in by_hop.items():
        out[hop] = {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in mdict.items()}
    out["all"] = {m: (sum(vs) / len(vs) if vs else 0.0) for m, vs in allhop.items()}
    out["n_all"] = len(cases)  # type: ignore[assignment]
    return out


def load_dataset(dirpath: Path) -> Dict[str, Dict]:
    """{baseline: aggregate_dict}"""
    res = {}
    for bl in BASELINE_ORDER:
        f = dirpath / f"{bl}.jsonl"
        if not f.exists():
            continue
        res[bl] = aggregate(load_jsonl(f))
    return res


SMD = load_dataset(SMD_DIR)
SYN = load_dataset(SYN_DIR)


def _save(fig, name: str):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}")


# ---------- Figure 1: Radar — overall profile on SMD ----------
def fig_radar():
    metrics = ["recall", "precision", "temporal_accuracy",
               "provenance_completeness", "r_score", "em"]
    metric_labels = ["Recall", "Precision", "Temporal\nAcc.", "Provenance\nCompl.",
                     "R Score", "EM"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9.5, 8.5), subplot_kw=dict(polar=True))
    for bl in BASELINE_ORDER:
        if bl not in SMD:
            continue
        vals = [SMD[bl]["all"][m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=COLORS[bl], linewidth=2.0 if bl == "B4_Full_GraphRAG" else 1.4,
                label=LABELS[bl], marker="o", markersize=4)
        ax.fill(angles, vals, color=COLORS[bl], alpha=0.08 if bl == "B4_Full_GraphRAG" else 0.04)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10.5)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    # position the radar slightly down so title and legend have dedicated space
    ax.set_position([0.10, 0.08, 0.70, 0.74])
    ax.set_title("Figure 1. Overall Metric Profile on SMD (Real-World, N=150 per baseline)",
                 pad=30, fontsize=12, y=1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(0.82, 0.95), fontsize=9, frameon=True,
              edgecolor="#cccccc", framealpha=0.95)
    _save(fig, "fig1_radar_overall.png")


# ---------- Figure 2: Grouped bars — answer quality metrics ----------
def fig_answer_quality():
    metrics = [("recall", "Recall"), ("r_score", "R Score"),
               ("ar_score", "AR Score"), ("em", "EM")]
    baselines = [b for b in BASELINE_ORDER if b in SMD]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, bl in enumerate(baselines):
        vals = [SMD[bl]["all"][m] for m, _ in metrics]
        offset = (i - (len(baselines) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[bl], color=COLORS[bl],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if v > 0.02:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score (↑ better)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 2. Answer-Quality Metrics by Baseline (SMD, N=150 each)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    _save(fig, "fig2_answer_quality.png")


# ---------- Figure 3: Hop-count difficulty scaling ----------
def fig_hop_scaling():
    hops = [2, 3, 4]
    metrics = [("recall", "Recall"), ("r_score", "R Score"), ("em", "EM")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), sharey=True)
    for ax, (mkey, mname) in zip(axes, metrics):
        # Always put B3 labels ABOVE markers, B4 labels BELOW markers to avoid
        # collision when the two lines are close in value.
        for bl, sign in [("B3_NoTemporal", +1), ("B4_Full_GraphRAG", -1)]:
            if bl not in SMD:
                continue
            vals = [SMD[bl][h][mkey] for h in hops]
            ax.plot(hops, vals, marker="o", linewidth=2.0, markersize=8,
                    color=COLORS[bl], label=LABELS[bl])
            for h, v in zip(hops, vals):
                # vertical offset with arrow if very close to other line
                if sign > 0:
                    y_off = 0.045
                    va = "bottom"
                else:
                    y_off = -0.045
                    va = "top"
                ax.text(h, v + y_off, f"{v:.2f}", ha="center", va=va,
                        fontsize=8.5, color=COLORS[bl], fontweight="bold")
        ax.set_xticks(hops)
        ax.set_xticklabels([f"{h}-hop" for h in hops])
        ax.set_xlabel("Reasoning Depth (hops)")
        ax.set_title(mname, pad=10)
        ax.set_ylim(-0.10, 1.10)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Score (↑ better)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Figure 3. Performance Decay with Increasing Hop Count (SMD, N=50 per hop)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig3_hop_scaling.png")


# ---------- Figure 4: Hallucination rate v2 -> v3 fix ----------
def fig_hallu_fix():
    """Isolate the bug-fix effect on the SAME (synthetic v3) dataset by comparing
    the buggy stored `hallucination_rate_overall` (only CONTRADICTED counted at
    per-hop level, contaminating the overall via a separate code path used in
    v2 reports) against the recomputed correct rate (CONTRADICTED + UNSUPPORTED).

    The IMPROVEMENT_REPORT.md documents that the per-hop rate computation in
    `eval/metrics.py` ignored UNSUPPORTED claims, so baselines with no graph
    paths (B1, B2) had their per-hop hallucination rates artificially driven
    to 0. The overall rate was independently fixed, but the v2 stored values
    in eval/reports_full_v3/ were generated before that fix landed.
    """
    # Use synthetic v3 raw cases so the comparison is on a single dataset
    buggy = {}
    corrected = {}
    for bl in BASELINE_ORDER:
        f = SYN_DIR / f"{bl}.jsonl"
        if not f.exists():
            continue
        cases = load_jsonl(f)
        stored_vals = []
        corrected_vals = []
        for c in cases:
            stored_vals.append(float(c.get("hallucination_rate_overall", 0.0) or 0.0))
            ov, _ = _recompute_hallucination(c)
            corrected_vals.append(ov)
        buggy[bl] = sum(stored_vals) / len(stored_vals) if stored_vals else 0.0
        corrected[bl] = sum(corrected_vals) / len(corrected_vals) if corrected_vals else 0.0

    baselines = [b for b in BASELINE_ORDER if b in buggy]
    x = np.arange(len(baselines))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    v2 = [buggy[b] for b in baselines]
    v3 = [corrected[b] for b in baselines]
    b1 = ax.bar(x - width / 2, v2, width,
                label="v2 stored (buggy: per-hop ignored UNSUPPORTED)",
                color="#cccccc", edgecolor="white")
    b2 = ax.bar(x + width / 2, v3, width,
                label="recomputed (CONTRADICTED + UNSUPPORTED)",
                color="#c0392b", edgecolor="white")
    for bars, vals in [(b1, v2), (b2, v3)]:
        for b, v in zip(bars, vals):
            if v > 0.005:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                        "0.000", ha="center", va="bottom", fontsize=8, color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[b] for b in baselines], rotation=15, ha="right")
    ax.set_ylabel("Hallucination Rate (↓ better)")
    ax.set_ylim(0, 1.25)
    ax.set_title("Figure 4. Hallucination-Rate Bug Fix Impact (Synthetic v3 dataset, N=150 per baseline)",
                 pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    # place annotation above bars (not on top of values) and to the side
    ax.annotate("B1/B2 true hallucination\nrecovered from false 0",
                xy=(0.55, 0.95), xytext=(1.6, 1.10),
                fontsize=8.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    fig.subplots_adjust(bottom=0.22, top=0.92)
    _save(fig, "fig4_hallucination_fix.png")


# ---------- Figure 5: 4-hop empty path rate v2 -> v3 ----------
def fig_empty_path_fix():
    """Empty-path counts at 4-hop, v2 → v3, from IMPROVEMENT_REPORT.md.
    Empty = baseline returned no path (path_error_rate = 1.0 case-level).
    v2 numbers reflect pre-fallback Cypher; v3 numbers reflect post-fallback
    (4-hop UNION 3-hop) plus LLM multi-keyword extraction.
    """
    # v2: pre-fix empty counts at 4-hop (out of 50)
    # B1 = 50/50 (no graph at all); B2 = 0/50 (default search returns nodes,
    # not paths — counted as 0 empty but path_error_rate is high); B3 = 29/50;
    # B4 = 31/50. From IMPROVEMENT_REPORT.md.
    v2_empty = {"B1_NaiveRAG": 50, "B2_GraphitiDefault": 0,
                "B3_NoTemporal": 29, "B4_Full_GraphRAG": 31}
    # v3: post-fix empty counts at 4-hop (out of 50), from IMPROVEMENT_REPORT.md
    v3_empty = {"B1_NaiveRAG": 50, "B2_GraphitiDefault": 0,
                "B3_NoTemporal": 11, "B4_Full_GraphRAG": 17}
    baselines = [b for b in BASELINE_ORDER if b in v2_empty]
    x = np.arange(len(baselines))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    v2 = [v2_empty[b] for b in baselines]
    v3 = [v3_empty[b] for b in baselines]
    b1 = ax.bar(x - width / 2, v2, width, label="v2 (no fallback)",
                color="#cccccc", edgecolor="white")
    b2 = ax.bar(x + width / 2, v3, width,
                label="v3 (4-hop ∪ 3-hop fallback + multi-keyword Cypher)",
                color="#2c7a7b", edgecolor="white")
    for bars, vals in [(b1, v2), (b2, v3)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8,
                    f"{v}/50", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[b] for b in baselines], rotation=12, ha="right")
    ax.set_ylabel("# Empty-Path Cases at 4-hop (↓ better)")
    ax.set_ylim(0, 56)
    ax.set_title("Figure 5. 4-Hop Empty-Path Rate Fix (SMD, N=50 at 4-hop)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    _save(fig, "fig5_empty_path_fix.png")


# ---------- Figure 6: Pipeline 3-stage graph-construction quality ----------
def fig_pipeline_quality():
    metrics = [("entity_recall", "Entity Recall"),
               ("relation_recall", "Relation Recall"),
               ("pipeline_f1", "Pipeline F1")]
    baselines = [b for b in BASELINE_ORDER if b in SMD]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for i, bl in enumerate(baselines):
        vals = [SMD[bl]["all"][m] for m, _ in metrics]
        offset = (i - (len(baselines) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=LABELS[bl], color=COLORS[bl],
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            if v > 0.02:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylabel("Score (↑ better)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 6. Graph-Construction Pipeline Quality (GraphRAG-Bench §3.2, SMD)")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    _save(fig, "fig6_pipeline_quality.png")


# ---------- Figure 7: SMD vs Synthetic dataset comparison ----------
def fig_dataset_compare():
    metrics = [("recall", "Recall"), ("r_score", "R Score"),
               ("em", "EM"), ("hallucination_rate_overall", "Hallu.")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
    hops_all = "all"
    for ax, (mkey, mname) in zip(axes, metrics):
        better = mkey != "hallucination_rate_overall"
        ds_names = ["Synthetic", "SMD (real)"]
        ds_data = [SYN, SMD]
        width = 0.18
        x = np.arange(len(BASELINE_ORDER))
        for j, (ds_name, ds) in enumerate(zip(ds_names, ds_data)):
            vals = []
            for bl in BASELINE_ORDER:
                if bl in ds:
                    vals.append(ds[bl][hops_all].get(mkey, 0.0))
                else:
                    vals.append(0.0)
            offset = (j - 0.5) * width
            color = "#0072B2" if j == 1 else "#a0c4e0"
            bars = ax.bar(x + offset, vals, width, label=ds_name, color=color,
                          edgecolor="white", linewidth=0.5)
            for b, v in zip(bars, vals):
                if v > 0.02:
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(["B1", "B2", "B3", "B4"], fontsize=9)
        ax.set_title(mname, fontsize=11)
        ax.set_ylim(0, 1.1 if better else 1.15)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 7. Synthetic vs. Real-World SMD Dataset Comparison (overall, N=150 per cell)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig7_dataset_compare.png")


# ---------- Figure 8: System architecture pipeline ----------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(15.0, 6.0))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")  # critical: 1 data unit = 1 visual unit on both axes
    ax.axis("off")

    blocks = [
        # (x, y, w, h, label, color)  — uniform 1.6 high boxes for legibility
        (0.2, 2.0, 1.9, 1.6, "SMD + MicroSS\nHeterogeneous\nDual-Source Data", "#2c5282"),
        (2.4, 2.0, 1.9, 1.6, "Schema-Constrained\nGraphiti Writer", "#2c7a7b"),
        (4.6, 2.0, 1.9, 1.6, "Neo4j\nTemporal KG\n(valid_at / invalid_at)", "#2f855a"),
        (6.8, 3.4, 1.9, 1.6, "LLM Intent Parser\n(multi-keyword extract)", "#c05621"),
        (6.8, 0.6, 1.9, 1.6, "Cypher Generator\n(2/3/4-hop + fallback)", "#9c4221"),
        (9.0, 2.0, 1.9, 1.6, "Temporal Pruner\nLAG_MAX=3600s\nLAG_NEG=7200s", "#6b46c1"),
        (11.2, 2.0, 1.9, 1.6, "Claim Decomposer\n+ Hallucination\nVerifier", "#b83280"),
    ]
    centers = []
    for x, y, w, h, label, color in blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                             linewidth=1.2, edgecolor="white", facecolor=color)
        ax.add_patch(box)
        n_lines = label.count("\n") + 1
        # font sized so 3-line labels fit inside 1.6-high box with comfortable margin
        fs = 8.5 if n_lines == 2 else 7.8
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color="white", fontsize=fs, fontweight="bold", linespacing=1.05)
        centers.append((x + w / 2, y + h / 2))

    # arrows — sequential pipeline (route middle split above/below to avoid overlap)
    arrows = [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 6)]
    for a, b in arrows:
        x1, y1 = centers[a]
        x2, y2 = centers[b]
        if (a, b) == (2, 3):
            conn = "arc3,rad=-0.18"
        elif (a, b) == (2, 4):
            conn = "arc3,rad=0.18"
        elif (a, b) == (3, 5):
            conn = "arc3,rad=0.10"
        elif (a, b) == (4, 5):
            conn = "arc3,rad=-0.10"
        else:
            conn = "arc3,rad=0.0"
        arr = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle="->,head_length=8,head_width=5",
                              color="#444", linewidth=1.4,
                              connectionstyle=conn,
                              shrinkA=22, shrinkB=22)
        ax.add_patch(arr)

    # legend annotations for ablation cuts — place them on the SAME horizontal
    # line but with extra horizontal separation so labels do not overlap
    ax.text(8.0, 5.30, "↑ removed in B3_NoTemporal", ha="left",
            fontsize=9.5, color="#6b46c1", style="italic", fontweight="bold")
    ax.annotate("", xy=(9.95, 3.65), xytext=(8.0, 5.20),
                arrowprops=dict(arrowstyle="->", color="#6b46c1", lw=1, linestyle="--"))
    ax.text(13.2, 5.30, "↑ removed in B3_NoTemporal", ha="left",
            fontsize=9.5, color="#b83280", style="italic", fontweight="bold")
    ax.annotate("", xy=(12.15, 3.65), xytext=(13.2, 5.20),
                arrowprops=dict(arrowstyle="->", color="#b83280", lw=1, linestyle="--"))

    # baselines annotation
    ax.text(7.5, 0.10,
            "B1 NaiveRAG: skip graph (LLM direct)   •   B2 GraphitiDefault: Graphiti's built-in search()",
            ha="center", fontsize=9.5, color="#555", style="italic")
    ax.set_title("Figure 8. Graph-RAG Reasoning Pipeline & Baseline Ablation Map",
                 fontsize=13, fontweight="bold", pad=10)
    _save(fig, "fig8_architecture.png")


# ---------- Figure 9: B3 vs B4 ablation (temporal pruner) ----------
def fig_ablation_temporal():
    metrics = [("recall", "Recall", True), ("temporal_accuracy", "Temporal Acc.", True),
               ("ar_score", "AR Score", True),
               ("hallucination_rate_overall", "Hallucination", False)]
    hops = [2, 3, 4]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.0))
    for ax, (mkey, mname, higher_better) in zip(axes, metrics):
        for bl in ["B3_NoTemporal", "B4_Full_GraphRAG"]:
            vals = [SMD[bl][h][mkey] for h in hops]
            ax.plot(hops, vals, marker="o", linewidth=2.0, markersize=7,
                    color=COLORS[bl], label=LABELS[bl])
            for h, v in zip(hops, vals):
                ax.text(h, v + 0.025 if higher_better else v + 0.02,
                        f"{v:.2f}", ha="center", fontsize=8, color=COLORS[bl])
        ax.set_xticks(hops)
        ax.set_xticklabels([f"{h}-hop" for h in hops])
        ax.set_xlabel("Reasoning Depth")
        ax.set_title(f"{mname} {'(↑)' if higher_better else '(↓)'}", fontsize=10.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Figure 9. Ablation: Effect of Temporal Pruner + Claim Decomposer (B3 vs B4, SMD)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    _save(fig, "fig9_ablation_temporal.png")


if __name__ == "__main__":
    print("Generating figures into", OUT)
    fig_radar()
    fig_answer_quality()
    fig_hop_scaling()
    fig_hallu_fix()
    fig_empty_path_fix()
    fig_pipeline_quality()
    fig_dataset_compare()
    fig_architecture()
    fig_ablation_temporal()
    print("Done.")
