"""Conference-ready figures for neural-network and RNN experiments.

Run from the project root:
    python -m src.models.presentation_plots --project-root .

Requires trained results/runs/*/seed_*.npz. Explainability figures additionally
require dataset/ml_manifest.csv and checkpoints/<model>/seed_*.pt.
The example loop assumes the magnetic field and magnetization are input channels
0 and 1. Override with --example-h-channel / --example-m-channel if needed.
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

MODEL_NAMES = {
    "mlp": "Neural network",
    "m_data_check_MLP": "Neural network + Physics",
    "preisach_mlp": "Neural network + Preisach",
    "m_data_check_preisach_MLP": "Neural network + Physics + Preisach",
    "rnn_joint": "RNN",
    "m_data_check_RNN": "RNN + Physics",
    "preisach_rnn": "RNN + Preisach",
    "m_data_check_preisach_RNN": "RNN + Physics + Preisach",
}
METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("balanced_accuracy", "Balanced accuracy"),
]
PALETTE = {"Neural network": "#1976A3", "RNN": "#C26942"}
# Shade by the observed impact of the physics loss on held-out Macro F1.
# The paired counterpart has exactly the same family and input representation.
PHYSICS_PAIRS = {
    "m_data_check_MLP": "mlp",
    "m_data_check_RNN": "rnn_joint",
    "m_data_check_preisach_MLP": "preisach_mlp",
    "m_data_check_preisach_RNN": "preisach_rnn",
}
HELP_BG = "#E6F4EC"  # positive delta on matched base model
HURT_BG = "#FCEAE8"  # negative delta
NEUTRAL_BG = "#F2F5F8"  # difference smaller than threshold
MIN_DELTA_PP = 0.25  # practical display threshold, not statistical significance
DEFAULT_MODELS = [
    "mlp",
    "m_data_check_MLP",
    "preisach_mlp",
    "m_data_check_preisach_MLP",
    "rnn_joint",
    "m_data_check_RNN",
    "preisach_rnn",
    "m_data_check_preisach_RNN",
]


def physics_impact(runs):
    """Map physics model -> paired test Macro F1 difference, in percentage points."""
    impact = {}
    for physics_name, base in PHYSICS_PAIRS.items():
        if physics_name not in runs or base not in runs:
            continue
        # Match seeds rather than accidentally comparing different experiments.
        shared = sorted(set(runs[physics_name]) & set(runs[base]))
        if not shared:
            continue
        differences = [
            100
            * (
                scores(
                    runs[physics_name][sd]["test_probs"],
                    runs[physics_name][sd]["test_labels"],
                )["macro_f1"]
                - scores(runs[base][sd]["test_probs"], runs[base][sd]["test_labels"])[
                    "macro_f1"
                ]
            )
            for sd in shared
        ]
        impact[physics_name] = float(np.mean(differences))
    return impact


def highlight_rows(ax, names, impact):
    for i, name in enumerate(names):
        if name not in PHYSICS_PAIRS:
            continue
        delta = impact.get(name)
        color = (
            NEUTRAL_BG
            if delta is None or abs(delta) < MIN_DELTA_PP
            else (HELP_BG if delta > 0 else HURT_BG)
        )
        ax.axhspan(i - 0.44, i + 0.44, color=color, lw=0, zorder=0)


def impact_legend(fig, impact):
    from matplotlib.patches import Patch

    if not impact:
        return
    handles = [
        Patch(
            facecolor=HELP_BG,
            edgecolor="#8BBAA1",
            label="Macro F1 higher than family baseline",
        ),
        Patch(
            facecolor=HURT_BG,
            edgecolor="#DDA7A1",
            label="Macro F1 equal to or below family baseline",
        ),
        Patch(
            facecolor=NEUTRAL_BG,
            edgecolor="#C4CDD5",
            label="Difference < 0.25 pp / no matched baseline",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.065),
        ncol=3,
        frameon=False,
        fontsize=11,
    )


def family(m):
    return "RNN" if ("rnn" in m.lower()) else "Neural network"


def label(m):
    return MODEL_NAMES.get(
        m,
        m.replace("mlp", "neural network")
        .replace("MLP", "Neural network")
        .replace("_", " "),
    )


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


def save(fig, out, stem):
    fig.savefig(out / (stem + ".png"), dpi=240, bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out / (stem + ".pdf"), bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("Saved:", out / (stem + ".png"))


def load_runs(root, requested):
    result = {}
    for directory in sorted((root / "results" / "runs").glob("*")):
        if not directory.is_dir():
            continue
        name = directory.name
        if name not in MODEL_NAMES or (requested and name not in requested):
            continue
        runs = {}
        for path in sorted(directory.glob("seed_*.npz")):
            seed = int(re.search(r"seed_(\d+)", path.stem).group(1))
            with np.load(path) as z:
                runs[seed] = {k: z[k].copy() for k in z.files}
        if runs:
            result[name] = runs
    order = requested if requested else DEFAULT_MODELS
    result = {m: result[m] for m in order if m in result}
    if not result:
        print(
            "[warning] No seed NPZ predictions found in results/runs/. Trying evaluation CSVs for performance."
        )
    return result


def scores(p, y):
    pred = p.argmax(1)
    return dict(
        accuracy=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, pred),
    )


def ensemble(runs, split="test"):
    first = next(iter(runs.values()))
    y = first[f"{split}_labels"]
    for d in runs.values():
        if not np.array_equal(y, d[f"{split}_labels"]):
            raise ValueError("Seed label order mismatch")
    return np.mean([d[f"{split}_probs"] for d in runs.values()], axis=0), y


def performance_data(runs, root):
    """Combine NPZ predictions with saved evaluation tables (NPZ takes priority).

    This only supplies metrics for the performance chart. Confidence and
    attribution still use actual per-sample predictions/checkpoints.
    """
    data = {}
    for model, seeds in runs.items():
        arr = {
            key: np.asarray(
                [
                    scores(d["test_probs"], d["test_labels"])[key]
                    for d in seeds.values()
                ],
                dtype=float,
            )
            * 100
            for key, _ in METRICS
        }
        data[model] = {
            key: (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            for key, v in arr.items()
        }

    eval_dir = root / "results" / "eval"
    path = eval_dir / "per_seed.csv"
    if path.is_file():
        table = pd.read_csv(path)
        required = {"model", "accuracy", "macro_f1", "balanced_accuracy"}
        if required.issubset(table.columns):
            table["model"] = table["model"].astype(str).str.strip()
            for name, group in table.groupby("model"):
                if name not in MODEL_NAMES or name in data:
                    continue
                vals = {}
                for key, _ in METRICS:
                    v = (
                        pd.to_numeric(group[key], errors="coerce").dropna().to_numpy()
                        * 100
                    )
                    if not len(v):
                        break
                    vals[key] = (
                        float(v.mean()),
                        float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    )
                if len(vals) == len(METRICS):
                    data[name] = vals
                    print(f"[performance] {name}: recovered from {path}")

    path = eval_dir / "summary.csv"
    if path.is_file():
        table = pd.read_csv(path)
        if "model" in table.columns:
            table["model"] = table["model"].astype(str).str.strip()
            for _, row in table.iterrows():
                name = row["model"]
                if name not in MODEL_NAMES or name in data:
                    continue
                vals = {}
                for key, _ in METRICS:
                    mean = pd.to_numeric(row.get(f"{key}_mean"), errors="coerce")
                    std = pd.to_numeric(row.get(f"{key}_std"), errors="coerce")
                    if pd.isna(mean):
                        break
                    vals[key] = (
                        float(mean * 100),
                        float(std * 100) if pd.notna(std) else 0.0,
                    )
                if len(vals) == len(METRICS):
                    data[name] = vals
                    print(f"[performance] {name}: recovered from {path}")
    return data


def performance_physics_impact(data):
    result = {}
    for phys, baseline in PHYSICS_PAIRS.items():
        if phys in data and baseline in data:
            result[phys] = data[phys]["macro_f1"][0] - data[baseline]["macro_f1"][0]
    return result


def performance(
    runs,
    out,
    root,
    data_override=None,
    stem="01_objective_performance",
    title="Objective model performance",
    scope_note="",
):
    """Two presentation-sized vertical bar charts with four fixed variants."""
    from matplotlib.patches import Patch

    families = {
        "Neural network": [
            "mlp",
            "m_data_check_MLP",
            "preisach_mlp",
            "m_data_check_preisach_MLP",
        ],
        "RNN": [
            "rnn_joint",
            "m_data_check_RNN",
            "preisach_rnn",
            "m_data_check_preisach_RNN",
        ],
    }
    variant_labels = [
        "Baseline",
        "+ Physics",
        "+ Preisach",
        "+ Physics\n+ Preisach",
    ]
    metric_colors = {
        "accuracy": "#24688D",
        "macro_f1": "#548B63",
        "balanced_accuracy": "#C17D45",
    }
    data = data_override if data_override is not None else performance_data(runs, root)
    if not data:
        raise SystemExit(
            "No test metrics found. Expected results/runs/*/seed_*.npz or results/eval/per_seed.csv / summary.csv"
        )
    # Green/red compares EVERY variant with the plain baseline of its family.
    # The criterion is test Macro F1, matching the caption and legend.
    fig, axes = plt.subplots(2, 1, figsize=(19.2, 12.6), sharex=True, sharey=True)
    width = 0.22
    offsets = np.array([-width, 0.0, width])
    x = np.arange(4, dtype=float)

    for ax, (title, models) in zip(axes, families.items()):
        ax.set_xlim(-0.65, 3.65)
        ax.set_ylim(0, 108)  # Headroom for SD whiskers near 100%.
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel("Score (%)", fontsize=18, labelpad=14)
        ax.set_title(title, fontsize=22, weight="bold", pad=17)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E9EC", linewidth=1.0)

        baseline = models[0]
        baseline_f1 = data[baseline]["macro_f1"][0] if baseline in data else None
        for i, name in enumerate(models):
            if i > 0:
                if name not in data or baseline_f1 is None:
                    bg = NEUTRAL_BG
                else:
                    delta = data[name]["macro_f1"][0] - baseline_f1
                    # Strict increase = green; same or worse = red, as requested.
                    bg = HELP_BG if delta > 1e-9 else HURT_BG
                ax.axvspan(i - 0.46, i + 0.46, facecolor=bg, edgecolor="none", zorder=0)

            if name not in data:
                ax.text(
                    i,
                    51,
                    "Not\navailable",
                    ha="center",
                    va="center",
                    fontsize=15,
                    color="#7C8791",
                    style="italic",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="#D9E0E5",
                        boxstyle="round,pad=.5",
                        alpha=0.9,
                    ),
                    zorder=5,
                )
                continue

            for offset, (key, metric) in zip(offsets, METRICS):
                mean, std = data[name][key]
                ax.bar(
                    i + offset,
                    mean,
                    width=width * 0.92,
                    color=metric_colors[key],
                    zorder=2,
                )
                if std > 0:
                    ax.errorbar(
                        i + offset,
                        mean,
                        yerr=std,
                        fmt="none",
                        ecolor="#263440",
                        capsize=5,
                        elinewidth=1.6,
                        capthick=1.6,
                        zorder=4,
                    )

        ax.tick_params(axis="y", labelsize=16, length=0, pad=7)
        ax.tick_params(axis="x", labelsize=17, length=0, pad=12)
        ax.set_xticks(x, variant_labels)
        ax.tick_params(axis="x", labelbottom=True)
        ax.spines["left"].set_color("#B6C0C7")
        ax.spines["bottom"].set_color("#B6C0C7")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

    metric_handles = [
        Patch(facecolor=metric_colors[k], label=lab) for k, lab in METRICS
    ]
    physics_handles = [
        Patch(
            facecolor=HELP_BG,
            edgecolor="#AFD4C0",
            label="Higher Macro F1 than baseline",
        ),
        Patch(
            facecolor=HURT_BG,
            edgecolor="#E5B8B2",
            label="Same or lower Macro F1 than baseline",
        ),
        Patch(
            facecolor=NEUTRAL_BG, edgecolor="#D5DDE5", label="Comparison unavailable"
        ),
    ]
    fig.suptitle(title, fontsize=30, weight="bold", y=0.975)
    fig.legend(
        handles=metric_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.934),
        ncol=3,
        frameon=False,
        fontsize=17,
        columnspacing=3,
    )
    fig.legend(
        handles=physics_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.036),
        ncol=3,
        frameon=False,
        fontsize=13,
        columnspacing=2.3,
    )
    fig.text(
        0.5,
        0.082,
        scope_note
        or "Background: Macro F1 vs the plain baseline in each family. Bars: mean across seeds; whiskers: ±1 SD.",
        ha="center",
        fontsize=14,
        color="#53606B",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.165, hspace=0.41)
    save(fig, out, stem)


def calibration(p, y, bins=10):
    conf = p.max(1)
    correct = p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.minimum(np.digitize(conf, edges[1:-1]), bins - 1)
    err = 0.0
    for k in range(bins):
        take = idx == k
        if take.any():
            err += take.mean() * abs(conf[take].mean() - correct[take].mean())
    return conf, correct, err


def seed_disagreement(per_seed_probs):
    preds = np.stack([p.argmax(1) for p in per_seed_probs], axis=0)
    disagree = np.any(preds != preds[0:1], axis=0)
    return float(disagree.mean())


def confidence_std_across_seeds(per_seed_probs):
    conf = np.stack([p.max(1) for p in per_seed_probs], axis=0)
    return float(conf.std(axis=0, ddof=0).mean())


def reliability_data(runs, root):
    """Collect comparable calibration and seed-ensemble epistemic MI.

    Source priority: raw per-seed NPZ predictions > eval CSV results. The
    calibration.csv `raw` rows describe raw softmax, not temperature-scaled
    variants; uncertainty.csv `seed_ensemble` rows describe seed disagreement.
    """
    result = {}
    for name, per_seed in runs.items():
        ordered = [per_seed[k] for k in sorted(per_seed)]
        y = ordered[0]["test_labels"]
        pp = [d["test_probs"].astype(float) for d in ordered]
        if not all(np.array_equal(y, d["test_labels"]) for d in ordered):
            print(
                f"[reliability] Skipping {name}: inconsistent label order between seeds"
            )
            continue
        es = np.array([calibration(q, y)[2] for q in pp])
        ens = np.mean(pp, axis=0)
        result[name] = {
            "ece_mean": float(es.mean()),
            "ece_sd": float(es.std(ddof=1)) if len(es) > 1 else 0.0,
            "ece_ensemble": float(calibration(ens, y)[2]),
            "ece_seeds": es,
            "uncertainty": None,
            "source": "NPZ",
        }
        if len(pp) > 1:
            P = np.stack(pp, axis=0)
            H = lambda z: -(z * np.log(np.clip(z, 1e-12, 1))).sum(axis=-1)
            result[name]["uncertainty"] = float(
                np.maximum(H(P.mean(0)) - H(P).mean(0), 0).mean()
            )

    cal_path = root / "results" / "eval" / "calibration.csv"
    if cal_path.is_file():
        df = pd.read_csv(cal_path)
        if {"model", "variant", "ece_mean"}.issubset(df.columns):
            df["model"] = df["model"].astype(str).str.strip()
            for _, row in df.loc[
                df["variant"].astype(str).str.strip() == "raw"
            ].iterrows():
                name = row["model"]
                if name not in MODEL_NAMES or name in result:
                    continue
                val = pd.to_numeric(row["ece_mean"], errors="coerce")
                if not np.isfinite(val):
                    continue
                sd = pd.to_numeric(row.get("ece_std"), errors="coerce")
                en = pd.to_numeric(row.get("ece_ensemble"), errors="coerce")
                result[name] = {
                    "ece_mean": float(val),
                    "ece_sd": float(sd) if pd.notna(sd) else 0.0,
                    "ece_ensemble": float(en) if pd.notna(en) else None,
                    "ece_seeds": None,
                    "uncertainty": None,
                    "source": "calibration.csv",
                }
                print(f"[reliability] {name}: calibration recovered from {cal_path}")

    # Per-seed.csv contains raw per-seed ECE too; use when calibration.csv
    # was never generated, or as a source of individual points.
    seed_path = root / "results" / "eval" / "per_seed.csv"
    if seed_path.is_file():
        df = pd.read_csv(seed_path)
        if {"model", "ece"}.issubset(df.columns):
            df["model"] = df["model"].astype(str).str.strip()
            for name, group in df.groupby("model"):
                if name not in MODEL_NAMES:
                    continue
                v = pd.to_numeric(group["ece"], errors="coerce").dropna().to_numpy()
                if len(v) == 0:
                    continue
                if name in result:
                    if result[name]["ece_seeds"] is None:
                        result[name]["ece_seeds"] = v
                    continue
                result[name] = {
                    "ece_mean": float(v.mean()),
                    "ece_sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "ece_ensemble": None,
                    "ece_seeds": v,
                    "uncertainty": None,
                    "source": "per_seed.csv",
                }
                print(f"[reliability] {name}: calibration recovered from {seed_path}")

    unc_path = root / "results" / "eval" / "uncertainty.csv"
    if unc_path.is_file():
        df = pd.read_csv(unc_path)
        if {"model", "source", "epistemic_unc"}.issubset(df.columns):
            df["model"] = df["model"].astype(str).str.strip()
            for _, row in df.loc[
                df["source"].astype(str).str.strip() == "seed_ensemble"
            ].iterrows():
                name = row["model"]
                v = pd.to_numeric(row["epistemic_unc"], errors="coerce")
                if (
                    name in result
                    and result[name]["uncertainty"] is None
                    and pd.notna(v)
                ):
                    result[name]["uncertainty"] = max(float(v), 0.0)
                    print(
                        f"[reliability] {name}: epistemic uncertainty recovered from {unc_path}"
                    )

    missing_cal = [m for m in DEFAULT_MODELS if m not in result]
    missing_unc = [
        m for m in DEFAULT_MODELS if m in result and result[m]["uncertainty"] is None
    ]
    if missing_cal:
        print("[reliability] No raw ECE values for:", ", ".join(missing_cal))
    if missing_unc:
        print(
            "[reliability] No seed-ensemble epistemic MI for:", ", ".join(missing_unc)
        )
    return result


def panel_effect_vs_family_baseline(data, metric_key, threshold):
    """Return background color intent relative to the plain family baseline.

    For presentation clarity, every variant is compared against the simplest model
    in the same family:
      - Neural network family -> mlp
      - RNN family -> rnn_joint

    For ECE / uncertainty, lower is better, so a negative delta is an improvement.
    effect[name] = -1  -> improvement vs family baseline (green)
    effect[name] = +1  -> worse than family baseline (red)
    effect[name] = 0   -> nearly unchanged
    """
    base_for_family = {"Neural network": "mlp", "RNN": "rnn_joint"}
    effect = {}
    for name in data:
        fam = family(name)
        baseline = base_for_family.get(fam)
        if not baseline or baseline not in data or name == baseline:
            continue
        cur_val = data[name].get(metric_key)
        base_val = data[baseline].get(metric_key)
        if cur_val is None or base_val is None:
            continue
        delta = float(cur_val) - float(base_val)
        effect[name] = 0 if abs(delta) < threshold else (-1 if delta < 0 else 1)
    return effect


def confidence_plot(runs, out, root):
    """Two-by-two slide: calibration and epistemic uncertainty.

    Background shading is metric-specific here:
      - Each variant is compared against the plain baseline of the same family.
      - Green means the shown metric is better than that baseline; red means worse.
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    families = {
        "Neural network": [
            "mlp",
            "m_data_check_MLP",
            "preisach_mlp",
            "m_data_check_preisach_MLP",
        ],
        "RNN": [
            "rnn_joint",
            "m_data_check_RNN",
            "preisach_rnn",
            "m_data_check_preisach_RNN",
        ],
    }
    labels = ["Baseline", "+ Physics", "+ Preisach", "+ Physics\n+ Preisach"]
    data = reliability_data(runs, root)
    if not data:
        print(
            "[reliability] Figure skipped: no calibration metrics found in NPZ or eval CSVs."
        )
        return

    ece_effect = panel_effect_vs_family_baseline(
        data, "ece_mean", threshold=0.005
    )  # 0.5 pp ECE
    mi_effect = panel_effect_vs_family_baseline(
        data, "uncertainty", threshold=0.002
    )  # 0.002 nats MI

    fig, axes = plt.subplots(2, 2, figsize=(19.2, 12.8))
    xpos = np.arange(4)
    max_ece = max(
        (100 * (d["ece_mean"] + d["ece_sd"]) for d in data.values()), default=10
    )
    max_mi = max(
        (d["uncertainty"] for d in data.values() if d["uncertainty"] is not None),
        default=0.02,
    )
    cal_ylim = max(10, np.ceil((max_ece + 4) / 5) * 5)
    mi_ylim = max(0.025, max_mi * 1.25)

    for row, (fam, models) in enumerate(families.items()):
        left, right = axes[row]
        for ax in (left, right):
            ax.set_xlim(-0.62, 3.62)
            ax.set_xticks(xpos, labels)
            ax.tick_params(axis="both", labelsize=15, length=0, pad=8)
            ax.grid(axis="y", color="#E5E9EC", lw=1)
            ax.set_axisbelow(True)

        # metric-specific backgrounds
        for i, name in enumerate(models):
            if name in PHYSICS_PAIRS:
                sign = ece_effect.get(name)
                bg = (
                    NEUTRAL_BG
                    if sign is None or sign == 0
                    else (HELP_BG if sign < 0 else HURT_BG)
                )
                left.axvspan(i - 0.44, i + 0.44, color=bg, lw=0, zorder=0)
                sign = mi_effect.get(name)
                bg = (
                    NEUTRAL_BG
                    if sign is None or sign == 0
                    else (HELP_BG if sign < 0 else HURT_BG)
                )
                right.axvspan(i - 0.44, i + 0.44, color=bg, lw=0, zorder=0)

        left.set_title(
            f"{fam}  ·  calibration error", fontsize=20, weight="bold", pad=14
        )
        right.set_title(
            f"{fam}  ·  uncertainty across seeds", fontsize=20, weight="bold", pad=14
        )
        left.set_ylabel("Expected calibration error (pp)", fontsize=16)
        right.set_ylabel("Epistemic uncertainty (nats)", fontsize=16)
        left.set_ylim(0, cal_ylim)
        right.set_ylim(0, mi_ylim)

        for i, name in enumerate(models):
            d = data.get(name)
            if not d:
                left.text(
                    i,
                    cal_ylim * 0.44,
                    "No saved\nECE",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#667381",
                )
                right.text(
                    i,
                    mi_ylim * 0.44,
                    "No saved\nuncertainty",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#667381",
                )
                continue
            mean = d["ece_mean"] * 100
            sd = d["ece_sd"] * 100
            left.bar(i, mean, width=0.56, color="#35759B", zorder=2)
            if sd > 0:
                left.errorbar(
                    i,
                    mean,
                    yerr=sd,
                    fmt="none",
                    ecolor="#263440",
                    capsize=5,
                    lw=1.5,
                    zorder=4,
                )
            if d["ece_seeds"] is not None:
                v = np.asarray(d["ece_seeds"]) * 100
                jit = np.linspace(-0.10, 0.10, len(v)) if len(v) > 1 else np.zeros(1)
                left.scatter(
                    i + jit,
                    v,
                    s=24,
                    c="white",
                    edgecolors="#394F61",
                    linewidths=0.7,
                    zorder=5,
                )
            if d["ece_ensemble"] is not None:
                left.scatter(
                    i,
                    d["ece_ensemble"] * 100,
                    marker="D",
                    s=65,
                    color="#212936",
                    edgecolor="white",
                    linewidth=1,
                    zorder=6,
                )

            if d["uncertainty"] is None:
                right.text(
                    i,
                    mi_ylim * 0.38,
                    "No saved\nuncertainty",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#667381",
                )
            else:
                mi = d["uncertainty"]
                right.bar(i, mi, width=0.56, color="#8665A7", zorder=2)
                right.text(
                    i,
                    min(mi + mi_ylim * 0.025, mi_ylim * 0.95),
                    f"{mi:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#45515E",
                )

    legend = [
        Patch(facecolor="#35759B", label="Mean ECE (seeds)"),
        Line2D([0], [0], color="#263440", marker="_", markersize=12, label="±1 SD"),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            color="#212936",
            markersize=8,
            label="Seed ensemble ECE",
        ),
        Patch(facecolor="#8665A7", label="Seed-ensemble mutual information"),
    ]
    shade_legend = [
        Patch(
            facecolor=HELP_BG,
            edgecolor="#AFD4C0",
            label="Better than the plain baseline in this family",
        ),
        Patch(
            facecolor=HURT_BG,
            edgecolor="#E5B8B2",
            label="Worse than the plain baseline in this family",
        ),
        Patch(
            facecolor=NEUTRAL_BG,
            edgecolor="#D5DDE5",
            label="No clear change / comparison unavailable",
        ),
    ]
    fig.suptitle(
        "Calibration and predictive uncertainty", fontsize=28, weight="bold", y=0.98
    )
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
        fontsize=13,
    )
    fig.legend(
        handles=shade_legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=3,
        frameon=False,
        fontsize=12.0,
    )
    fig.text(
        0.5,
        0.118,
        "Lower ECE means better calibration. Lower mutual information means more similar predicted probabilities across training seeds.",
        ha="center",
        fontsize=12.6,
        color="#53606B",
    )
    fig.text(
        0.5,
        0.095,
        "Background color compares each variant with the plain baseline in the same family.",
        ha="center",
        fontsize=12.0,
        color="#53606B",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, top=0.83, bottom=0.21, hspace=0.48, wspace=0.20
    )
    save(fig, out, "02_confidence_and_calibration")


def explanation_models(root, requested=None):
    """Find trained models from checkpoints independently of the NPZ/CSV figures."""
    requested_set = set(requested) if requested is not None else None
    found = {}
    for name in DEFAULT_MODELS:
        if requested_set is not None and name not in requested_set:
            continue
        directory = root / "checkpoints" / name
        checkpoints = sorted(directory.glob("seed_*.pt")) if directory.exists() else []
        seeds = {}
        for path in checkpoints:
            match = re.fullmatch(r"seed_(\d+)", path.stem)
            if match:
                seeds[int(match.group(1))] = path
        if seeds:
            found[name] = seeds
    return found


def explanation_seed(model_checkpoints, requested_seed=None):
    if requested_seed is None:
        return min(model_checkpoints)
    if requested_seed not in model_checkpoints:
        raise FileNotFoundError(f"checkpoint for seed {requested_seed} not found")
    return requested_seed


def physics_importance(runs, root, out, repeats=5, seed_choice=None):
    import torch
    from torch.utils.data import DataLoader
    from src.models.registry import get_spec
    from src.models.experiment import MANIFEST_PATH as _unused
    from src import config

    n_temp = len(config.TEMPERATURES_K)
    names = []
    rows = []
    rng = np.random.default_rng(2109)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, all_seeds in runs.items():
        try:
            seeds = [explanation_seed(all_seeds, seed_choice)]
        except FileNotFoundError as exc:
            print(f"[explain] {name}: {exc}")
            continue
        for seed in seeds:
            checkpoint = root / "checkpoints" / name / f"seed_{seed}.pt"
            if not checkpoint.exists():
                print(f"Skipping physics analysis for {name}: missing {checkpoint}")
                continue
            try:
                spec = get_spec(name)
                ds = spec.dataset_cls(
                    root / "dataset" / "ml_manifest.csv", split="test"
                )
                loader = DataLoader(
                    ds, batch_size=64, shuffle=False, collate_fn=spec.collate_fn
                )
                ck = torch.load(checkpoint, map_location=device, weights_only=False)
                hp = ck["hparams"]
                model = spec.build(hp).to(device)
                model.load_state_dict(ck["model_state"])
                model.eval()
                batches = list(loader)

                def evaluate(temp=None, repeat=0):
                    predictions = []
                    labels = []
                    with torch.inference_mode():
                        for bi, batch in enumerate(batches):
                            if temp is not None:
                                batch = list(batch)
                                x = batch[0]
                                if (
                                    not isinstance(x, torch.Tensor)
                                    or x.ndim < 3
                                    or x.shape[1] != n_temp
                                ):
                                    raise ValueError(
                                        f'{name}: cannot identify temperature axis in first tensor: {getattr(x,"shape",None)}'
                                    )
                                if x.shape[0] > 1:
                                    perm = rng.permutation(x.shape[0])
                                    if np.all(perm == np.arange(len(perm))):
                                        perm = np.roll(perm, 1)
                                    x = x.clone()
                                    x[:, temp] = batch[0][perm, temp]
                                    batch[0] = x
                                    # RNN sequence lengths must follow swapped padded curves.
                                    if (
                                        len(batch) == 4
                                        and isinstance(batch[1], torch.Tensor)
                                        and batch[1].shape[:2] == (x.shape[0], n_temp)
                                    ):
                                        lengths = batch[1].clone()
                                        lengths[:, temp] = batch[1][perm, temp]
                                        batch[1] = lengths
                                batch = tuple(batch)
                            logits, y = spec.forward(model, batch, device)
                            predictions.append(logits.argmax(1).cpu().numpy())
                            labels.append(y.cpu().numpy())
                    pred = np.concatenate(predictions)
                    truth = np.concatenate(labels)
                    return f1_score(truth, pred, average="macro", zero_division=0)

                baseline = evaluate()
                imp = []
                for t in range(n_temp):
                    changed = np.mean([evaluate(t, r) for r in range(repeats)])
                    imp.append((baseline - changed) * 100)
                names.append(name)
                rows.append(imp)
                print(
                    f"{name}: baseline macro F1={baseline:.3f}; temperature importance (pp)={np.round(imp,2)}"
                )
            except Exception as exc:
                print(
                    f"[explain] Temperature sensitivity unavailable for {name}: {exc}"
                )
                continue
    if not rows:
        print("Physics figure not created: missing compatible checkpoints/dataset.")
        return
    values = np.asarray(rows, dtype=float)
    # Two clearly separated groups; one common color scale makes the values comparable.
    from matplotlib.colors import TwoSlopeNorm

    families = [
        (
            "Neural network",
            ["mlp", "m_data_check_MLP", "preisach_mlp", "m_data_check_preisach_MLP"],
        ),
        (
            "RNN",
            [
                "rnn_joint",
                "m_data_check_RNN",
                "preisach_rnn",
                "m_data_check_preisach_RNN",
            ],
        ),
    ]
    short = ["Baseline", "+ Physics", "+ Preisach", "+ Physics + Preisach"]
    row_map = dict(zip(names, values))
    scale = max(1.0, float(np.max(np.abs(values))))
    fig, axes = plt.subplots(2, 1, figsize=(17, 9.8), constrained_layout=False)
    for ax, (fam, models) in zip(axes, families):
        mat = np.stack([row_map.get(m, np.full(n_temp, np.nan)) for m in models])
        cm = plt.get_cmap("RdBu_r").copy()
        cm.set_bad("#F1F3F5")
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            aspect="auto",
            cmap=cm,
            norm=TwoSlopeNorm(vcenter=0, vmin=-scale, vmax=scale),
        )
        ax.set_yticks(np.arange(4), short, fontsize=16)
        ax.set_xticks(
            np.arange(n_temp), [f"{t:g} K" for t in config.TEMPERATURES_K], fontsize=15
        )
        ax.set_title(fam, fontsize=22, weight="bold", pad=12)
        ax.set_xticks(np.arange(-0.5, n_temp, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        for i in range(4):
            for j in range(n_temp):
                val = mat[i, j]
                text = f"{val:+.1f}" if np.isfinite(val) else "—"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=15,
                    weight="bold",
                    color=(
                        "white"
                        if np.isfinite(val) and abs(val) > scale * 0.5
                        else "#202A36"
                    ),
                )
    cb = fig.colorbar(im, ax=axes, location="right", shrink=0.79, pad=0.018)
    cb.set_label("Test Macro F1 drop (percentage points)", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    fig.suptitle(
        "Which measured temperatures affect performance?",
        fontsize=27,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.048,
        "Positive numbers: disrupting this temperature makes predictions worse.  Negative: performance improves.",
        ha="center",
        fontsize=14,
        color="#465563",
    )
    fig.text(
        0.5,
        0.022,
        "Permutation sensitivity is not proof that a model learned physical laws.  One checkpoint per variant.",
        ha="center",
        fontsize=12.5,
        color="#63717C",
    )
    fig.subplots_adjust(left=0.18, right=0.88, top=0.87, bottom=0.13, hspace=0.55)
    save(fig, out, "03_physical_signal_importance")


def find_difficult_test_example(
    runs, checkpoints, out, requested_seed=None, mode="auto"
):
    """Select an aligned held-out sample using real, per-seed NPZ predictions.

    Prefer the case where both RNNs are wrong and the two neural networks are
    correct; otherwise choose the sample with most incorrect predictions.
    Never infer individual predictions from aggregated evaluation CSVs.
    """
    names = ["mlp", "m_data_check_MLP", "rnn_joint", "m_data_check_RNN"]
    available = [name for name in names if name in runs and name in checkpoints]
    if len(available) < 2:
        print(
            "[example] Not enough per-sample NPZ predictions AND checkpoints to select a difficult case."
        )
        return None
    seeds = set.intersection(
        *(set(runs[name]) & set(checkpoints[name]) for name in available)
    )
    if requested_seed is not None:
        seeds &= {requested_seed}
    if not seeds:
        print(
            "[example] No common seed with prediction NPZ and checkpoints for the models."
        )
        return None
    seed = sorted(seeds)[0]
    y = runs[available[0]][seed]["test_labels"]
    for name in available[1:]:
        if not np.array_equal(y, runs[name][seed]["test_labels"]):
            print("[example] Cannot select sample: test label orders do not match.")
            return None
    pred = {name: runs[name][seed]["test_probs"].argmax(axis=1) for name in available}
    conf = {name: runs[name][seed]["test_probs"].max(axis=1) for name in available}
    rows = []
    for idx in range(len(y)):
        wrong = [name for name in available if int(pred[name][idx]) != int(y[idx])]
        record = {
            "test_index": idx,
            "seed": seed,
            "true_label_index": int(y[idx]),
            "wrong_models_count": len(wrong),
            "wrong_models": "; ".join(wrong),
        }
        for name in names:
            record[f"{name}_pred"] = int(pred[name][idx]) if name in pred else np.nan
            record[f"{name}_conf"] = float(conf[name][idx]) if name in conf else np.nan
            record[f"{name}_correct"] = (
                bool(name not in wrong) if name in pred else np.nan
            )
        rows.append(record)
    df = pd.DataFrame(rows)
    df.to_csv(out / "difficult_test_examples.csv", index=False)
    # Showcase mode: choose a DIFFERENT sample (not sample 0 and not the
    # first hard-case sample used by the former automatic selector). Prefer a
    # case with visibly different decisions between architectures. This does
    # not pretend to measure attribution quality from predictions alone.
    if mode == "showcase":
        candidates = [i for i in range(len(y)) if i != 0]
        if not candidates:
            print("[example] Only one test sample is available; using index 0.")
            candidates = [0]

        # Rank samples by differing predictions, RNN errors and non-saturated
        # confidence, so the slide has a useful story without cherry-picking
        # test scores as if they were estimates of global performance.
        def rank(i):
            p = [int(pred[n][i]) for n in available]
            n_predictions = len(set(p))
            rnn_errors = sum(
                int(pred[n][i] != y[i])
                for n in ("rnn_joint", "m_data_check_RNN")
                if n in pred
            )
            base_correct = sum(
                int(pred[n][i] == y[i])
                for n in ("mlp", "m_data_check_MLP")
                if n in pred
            )
            unsaturated = sum(float(1 - conf[n][i]) for n in available)
            return (n_predictions, rnn_errors, base_correct, unsaturated, -i)

        candidates = sorted(candidates, key=rank, reverse=True)
        # Skip the default difficult example, if present, to ensure that we
        # really show a different example from the previous version.
        auto_legacy = next(
            (
                i
                for i in range(len(y))
                if all(
                    name in available and pred[name][i] != y[i]
                    for name in ("rnn_joint", "m_data_check_RNN")
                )
            ),
            None,
        )
        fresh = [i for i in candidates if i != auto_legacy]
        idx = int((fresh if fresh else candidates)[0])
        why = "different test example with contrasting model predictions"
        print(f"[example] Selected test_index={idx}, seed={seed}: {why}.")
        for name in available:
            print(
                f"  {name}: true={int(y[idx])}, predicted={int(pred[name][idx])}, confidence={conf[name][idx]:.1%}"
            )
        print(
            "[example] Saved per-sample comparison:",
            out / "difficult_test_examples.csv",
        )
        return idx, seed

    all_wrong = df[df["wrong_models_count"] == len(available)]
    both_rnn_wrong = [
        i
        for i in range(len(y))
        if all(
            name in available and pred[name][i] != y[i]
            for name in ("rnn_joint", "m_data_check_RNN")
        )
    ]
    if not all_wrong.empty:
        idx = int(all_wrong.iloc[0]["test_index"])
        why = f"all {len(available)} compared models incorrect"
    elif both_rnn_wrong:
        # Prefer cases where both neural networks are correct, when present.
        def rnn_rank(i):
            nn_correct = sum(
                int(pred[n][i] == y[i])
                for n in ("mlp", "m_data_check_MLP")
                if n in pred
            )
            return (-nn_correct, -int(df.iloc[i]["wrong_models_count"]), i)

        idx = sorted(both_rnn_wrong, key=rnn_rank)[0]
        why = "both RNNs incorrect (and neural networks correct where possible)"
    else:
        idx = int(
            df.sort_values(
                ["wrong_models_count", "test_index"], ascending=[False, True]
            ).iloc[0]["test_index"]
        )
        why = "maximum number of incorrect models available"
    print(f"[example] Selected test_index={idx}, seed={seed}: {why}.")
    for name in available:
        print(
            f"  {name}: true={int(y[idx])}, predicted={int(pred[name][idx])}, confidence={conf[name][idx]:.1%}"
        )
    print("[example] Saved per-sample comparison:", out / "difficult_test_examples.csv")
    return idx, seed


def integrated_gradients_rnn_ensemble_attribution(
    spec_models,
    batch,
    reference_batch,
    n_segments=10,
    n_steps=48,
    target_class=None,
    m_channel=1,
):
    """Integrated Gradients of the ensemble log probability w.r.t. input M.

    A *single path* starts from measured training magnetizations interpolated
    onto the test sampling grid and ends at the test sample. Magnetic field,
    lengths, temperatures and every other input feature are held unchanged.
    Thus attributions sum (approximately) to the change in the ensemble log
    probability along this path. A different reference can change the result.

    Returns signed IG values for chronological bins at every temperature, plus
    per-point attributions. It is NOT TimeSHAP and does not use SHAP libraries.
    """
    import contextlib
    import torch

    if not spec_models:
        raise ValueError("No RNN ensemble models supplied for Integrated Gradients")
    if len(batch) != 4 or len(reference_batch) != 4:
        raise ValueError(
            "Expected four-tensor RNN batches (curves, lengths, temperatures, labels)"
        )
    if n_segments < 2 or n_steps < 4:
        raise ValueError("Use at least two segments and four IG integration steps")
    device = next(spec_models[0][1].parameters()).device
    original, lengths, temperatures, labels = [v.detach().to(device) for v in batch]
    ref_curves, ref_lengths = reference_batch[:2]
    ref_curves, ref_lengths = ref_curves.detach().cpu(), ref_lengths.detach().cpu()
    if original.shape[0] != 1:
        raise ValueError("Integrated Gradients requires one test sample")
    if original.ndim != 4 or not 0 <= m_channel < original.shape[-1]:
        raise ValueError("Invalid magnetization channel or RNN input shape")

    baseline = original.clone()
    n_temp = original.shape[1]
    valid_lengths = []
    for t in range(n_temp):
        L = int(lengths[0, t].item())
        R = int(ref_lengths[0, t].item())
        valid_lengths.append(L)
        if L < 2 or R < 2:
            continue
        reference_m = ref_curves[0, t, :R, m_channel].numpy().astype(np.float64)
        reference_m = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, R), reference_m)
        baseline[0, t, :L, m_channel] = torch.tensor(
            reference_m, dtype=original.dtype, device=device
        )
    difference = original - baseline
    # Only magnetization on observed (unpadded) points changes on the path.
    diff_m = difference[..., m_channel]

    def ensemble_log_prob(x, cls):
        scores = []
        for spec, net in spec_models:
            logits, _ = spec.forward(net, (x, lengths, temperatures, labels), device)
            scores.append(torch.log_softmax(logits, dim=1)[:, cls])
        return torch.logsumexp(torch.stack(scores, dim=0), dim=0) - np.log(len(scores))

    def raw_probs(x):
        with torch.no_grad():
            each = []
            for spec, net in spec_models:
                logits, _ = spec.forward(
                    net, (x, lengths, temperatures, labels), device
                )
                each.append(torch.softmax(logits, dim=1))
            return torch.stack(each, dim=0).mean(dim=0)[0].cpu().numpy()

    original_probs = raw_probs(original)
    predicted = int(np.argmax(original_probs))
    cls = predicted if target_class is None else int(target_class)
    if not 0 <= cls < len(original_probs):
        raise ValueError("Invalid explained class index")

    # cuDNN RNN backward is not available with eval-mode recurrent layers;
    # disable its fused kernels just for the differentiable forward pass.
    contexts = [torch.backends.cudnn.flags(enabled=False)]
    if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "flags"):
        contexts.append(torch.backends.mkldnn.flags(enabled=False))
    with contextlib.ExitStack() as stack:
        for cm in contexts:
            stack.enter_context(cm)
        with torch.no_grad():
            reference_score = float(ensemble_log_prob(baseline, cls).item())
            original_score = float(ensemble_log_prob(original, cls).item())
        accumulated = torch.zeros_like(original)
        # Trapezoid quadrature, including both endpoints.
        for step in range(n_steps + 1):
            alpha = step / n_steps
            x = (baseline + alpha * difference).detach().requires_grad_(True)
            scalar = ensemble_log_prob(x, cls).sum()
            (grad,) = torch.autograd.grad(
                scalar, x, retain_graph=False, create_graph=False
            )
            weight = 0.5 if step in (0, n_steps) else 1.0
            accumulated += weight * grad.detach()
    ig = (
        (difference * (accumulated / n_steps))[0, :, :, m_channel]
        .detach()
        .cpu()
        .numpy()
    )
    all_scores = {}
    for t, L in enumerate(valid_lengths):
        points = ig[t, :L].astype(float)
        edges = np.linspace(0, L, n_segments + 1, dtype=int)
        seg = np.array(
            [points[edges[k] : edges[k + 1]].sum() for k in range(n_segments)],
            dtype=float,
        )
        all_scores[t] = {
            "integrated_gradients": seg,
            "point_attributions": points,
            "reference_log_probability": reference_score,
            "original_log_probability": original_score,
        }
    actual_sum = float(
        sum(np.sum(v["integrated_gradients"]) for v in all_scores.values())
    )
    residual = abs(actual_sum - (original_score - reference_score))
    if residual > max(0.05, 0.08 * abs(original_score - reference_score)):
        print(
            f"[integrated-gradients] Completeness residual {residual:.4g}; increase --ig-steps or inspect model operations"
        )
    return {
        "ensemble_probs": original_probs,
        "predicted_class": predicted,
        "target_class": cls,
        "temperatures": all_scores,
        "most_relevant_temperature": max(
            all_scores,
            key=lambda t: np.sum(np.abs(all_scores[t]["integrated_gradients"])),
        ),
        "completeness_residual": residual,
        "baseline_log_probability": reference_score,
        "original_log_probability": original_score,
    }


def preisach_feature_ensemble_attribution(
    spec_models, batch, perturbation_pct=5.0, batch_size=24
):
    """Local sensitivity of each Preisach coefficient for a seed ensemble.

    Returns an (n_temperatures, n_features) matrix of mean absolute changes to
    the ensemble probability of the originally predicted class. Uses symmetric
    +/- feature perturbations. These are *feature-space* interventions, not
    measured-loop perturbations and not SHAP/causal attributions.

    Works with PreisachFeatureDataset (2 tensors) and PreisachCurveDataset
    (5 tensors), using the model registry's actual forward method unchanged.
    The feature ordering is preserved; we do not assume an alpha/beta grid.
    """
    import torch

    if not spec_models:
        raise ValueError("No Preisach checkpoint models were provided")
    if not 0 < float(perturbation_pct) <= 25:
        raise ValueError("perturbation_pct must be in (0, 25]")
    device = next(spec_models[0][1].parameters()).device
    batch = tuple(
        a.detach().to(device) if isinstance(a, torch.Tensor) else a for a in batch
    )
    x = batch[0]
    if not isinstance(x, torch.Tensor) or x.shape[0] != 1 or x.ndim < 3:
        raise ValueError(
            f'Expected Preisach input (1, temperatures, features...), got {getattr(x, "shape", None)}'
        )
    n_temp = int(x.shape[1])
    n_feat = int(np.prod(x.shape[2:]))
    flat = x.reshape(1, n_temp, n_feat)

    def evaluate(inputs):
        count = inputs.shape[0]
        expanded = [inputs]
        for field in batch[1:]:
            if isinstance(field, torch.Tensor) and field.ndim and field.shape[0] == 1:
                expanded.append(field.repeat((count,) + (1,) * (field.ndim - 1)))
            else:
                expanded.append(field)
        scores = []
        with torch.inference_mode():
            for spec, model in spec_models:
                logits, _labels = spec.forward(model, tuple(expanded), device)
                if logits.shape[0] != count:
                    raise ValueError(
                        "The Preisach forward pass returned the wrong batch size"
                    )
                scores.append(torch.softmax(logits, dim=-1))
        return torch.stack(scores).mean(dim=0)

    base_probs = evaluate(x)[0].detach().cpu().numpy()
    pred = int(np.argmax(base_probs))
    base_p = float(base_probs[pred])
    scores = np.zeros((n_temp, n_feat), dtype=float)
    # A single per-temperature perturbation scale avoids treating a zero
    # coefficient as inherently unimportant. It uses the observed feature
    # magnitude, not an arbitrary zero-masking operation.
    scales = flat.detach().abs().cpu().numpy()[0]
    reference_scale = np.maximum(np.quantile(scales, 0.90, axis=1), 1e-5)
    feature_scale = (float(perturbation_pct) / 100.0) * reference_scale

    # Vectorized evaluations: two counterfactuals for every feature.
    for start in range(0, n_temp * n_feat, max(1, int(batch_size // 2))):
        index = list(
            range(start, min(n_temp * n_feat, start + max(1, int(batch_size // 2))))
        )
        count = len(index)
        candidates = flat.repeat(2 * count, 1, 1)
        for i, number in enumerate(index):
            t, f = divmod(number, n_feat)
            step = float(feature_scale[t])
            candidates[2 * i, t, f] += step
            candidates[2 * i + 1, t, f] -= step
        changed = evaluate(candidates.reshape((2 * count,) + tuple(x.shape[1:])))[
            :, pred
        ]
        changed = changed.detach().cpu().numpy().reshape(count, 2)
        impacts = np.abs(changed - base_p).mean(axis=1)
        for number, value in zip(index, impacts):
            t, f = divmod(number, n_feat)
            scores[t, f] = float(value)
    temperature_importance = scores.sum(axis=1)
    return {
        "importance": scores,
        "feature_shape": tuple(int(i) for i in x.shape[2:]),
        "temperatures": temperature_importance,
        "most_relevant_temperature": int(np.argmax(temperature_importance)),
        "ensemble_probs": base_probs,
        "predicted_class": pred,
        "perturbation_pct": float(perturbation_pct),
        "reference_feature_scale": reference_scale,
    }


def preisach_switch_geometry(n_features):
    """Exact feature order and switching fields from src.models.preisach.

    The ordering is (alpha index i, beta index j) for j <= i. The
    dimensionless switching fields are in units of H / Hmax, matching the
    normalized magnetic-field channel of CurveSequenceDataset.
    """
    from src.models.preisach import PreisachConfig

    cfg = PreisachConfig()
    n = int(cfg.n_nodes)
    if n * (n + 1) // 2 != int(n_features):
        raise ValueError(
            f"Preisach geometry mismatch: {n_features} features but "
            f"{n} nodes imply {n*(n+1)//2}. Check PreisachConfig."
        )
    u = np.linspace(-1.0, 1.0, n)
    g = np.sinh(cfg.node_warp * u) / np.sinh(cfg.node_warp) if cfg.node_warp > 0 else u
    nodes = cfg.node_max_frac * g  # normalized H
    ij = np.array([(i, j) for i in range(n) for j in range(i + 1)], dtype=int)
    return {
        "nodes": nodes,
        "alpha": nodes[ij[:, 0]],
        "beta": nodes[ij[:, 1]],
        "cell_i": ij[:, 0],
        "cell_j": ij[:, 1],
        "n_nodes": n,
    }


def preisach_curve_importance(
    raw_curves, lengths, attribution, h_channel=0, temperature=None, width=None
):
    """Map coefficient sensitivity to *switching fields* on measured branches.

    A Preisach hysteron (alpha,beta) switches DOWN at beta on the descending
    branch and UP at alpha on the ascending branch. We spread its sensitivity
    over nearby H samples on that branch, not over arbitrary pixel positions.
    This is a physically grounded switching-field correspondence, NOT an
    exact causal attribution to individual raw points.
    """
    imp = np.asarray(attribution["importance"], dtype=float)
    geo = preisach_switch_geometry(imp.shape[1])
    t = int(
        attribution["most_relevant_temperature"] if temperature is None else temperature
    )
    L = int(lengths[t])
    curve = np.asarray(raw_curves[t, :L], dtype=float)
    h = curve[:, h_channel]
    turn = int(np.argmin(h))
    val = np.zeros(L, dtype=float)
    if L < 5 or turn < 2 or turn >= L - 2:
        return val
    nodes = geo["nodes"]
    spacing = np.diff(nodes)
    bandwidth = float(
        width if width is not None else max(0.018, np.median(spacing) * 0.6)
    )
    # Marginals are the observable contribution of Preisach weights to a
    # major loop: beta controls the descending branch, alpha the ascending.
    down = np.bincount(geo["cell_j"], weights=imp[t], minlength=len(nodes))
    up = np.bincount(geo["cell_i"], weights=imp[t], minlength=len(nodes))

    def distribute(indices, marginal):
        hh = h[indices]
        if not len(hh):
            return
        distance = (hh[:, None] - nodes[None, :]) / bandwidth
        contributions = np.exp(-0.5 * distance**2) * marginal[None, :]
        val[indices] = contributions.sum(axis=1)

    distribute(np.arange(turn + 1), down)
    distribute(np.arange(turn + 1, L), up)
    return val


def plot_preisach_curve_importance(
    ax,
    raw_curves,
    lengths,
    attribution,
    h_channel=0,
    m_channel=1,
    highlight_fraction=0.75,
):
    """Plot original loops and orange segments near influential switch fields."""
    from matplotlib.collections import LineCollection

    importance = np.asarray(attribution["importance"])
    focus = int(attribution["most_relevant_temperature"])
    n_temp = min(len(lengths), len(importance))
    for t in range(n_temp):
        L = int(lengths[t])
        if L < 2:
            continue
        h = raw_curves[t, :L, h_channel]
        m = raw_curves[t, :L, m_channel]
        ax.plot(h, m, color="#CDD7E1", lw=2.4, zorder=1)
    L = int(lengths[focus])
    if L < 2:
        return False
    h = np.asarray(raw_curves[focus, :L, h_channel], float)
    m = np.asarray(raw_curves[focus, :L, m_channel], float)
    ax.plot(h, m, color="#7D94A8", lw=3.4, zorder=2)
    values = preisach_curve_importance(
        raw_curves, lengths, attribution, h_channel, focus
    )
    peak = float(np.max(values)) if len(values) else 0.0
    if peak > 1e-15:
        seg = (values[:-1] + values[1:]) * 0.5
        # If all switching regions matter nearly equally, don't exaggerate
        # tiny numerical differences as a sharply localized explanation.
        spread = float(np.quantile(seg, 0.9) - np.quantile(seg, 0.1))
        contrasted = spread > 0.15 * peak
        keep = (
            (seg >= peak * highlight_fraction)
            if contrasted
            else np.zeros_like(seg, dtype=bool)
        )
        indices = np.where(keep)[0]
        if len(indices):
            pts = np.column_stack((h, m))
            segs = np.stack((pts[indices], pts[indices + 1]), axis=1)
            widths = 4.0 + 4.0 * seg[indices] / peak
            lc = LineCollection(segs, colors="#F07832", linewidths=widths, zorder=4)
            lc.set_capstyle("round")
            ax.add_collection(lc)
    ax.grid(alpha=0.11)
    return bool(
        peak > 1e-15
        and np.any((values[:-1] + values[1:]) * 0.5 >= peak * highlight_fraction)
    )


def plot_preisach_feature_map(ax, attribution, temperature_labels=None, max_columns=36):
    """True triangular Preisach plane for the most sensitive temperature.

    Horizontal: alpha (up-switch field); vertical: beta (down-switch field).
    Only alpha >= beta cells exist. Colors represent absolute prediction
    sensitivity to coefficient perturbation, not the trained input weights.
    """
    scores = np.asarray(attribution["importance"], float)
    geo = preisach_switch_geometry(scores.shape[1])
    focus = int(attribution["most_relevant_temperature"])
    n = geo["n_nodes"]
    plane = np.full((n, n), np.nan)
    plane[geo["cell_j"], geo["cell_i"]] = scores[focus]
    extent = (geo["nodes"][0], geo["nodes"][-1], geo["nodes"][0], geo["nodes"][-1])
    cm = plt.get_cmap("YlOrBr").copy()
    cm.set_bad("#F2F5F8")
    im = ax.imshow(
        np.ma.masked_invalid(plane),
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cm,
        vmin=0,
        vmax=max(float(np.nanmax(plane)), 1e-12),
    )
    ax.plot([extent[0], extent[1]], [extent[0], extent[1]], color="#7E8A96", lw=0.8)
    label = (
        f"{temperature_labels[focus]:g} K"
        if temperature_labels is not None
        else f"Temperature {focus}"
    )
    ax.set_title(f"Preisach plane · {label}", fontsize=15, weight="bold", pad=9)
    ax.set_xlabel("Up-switch field α (H/Hmax)", fontsize=12)
    ax.set_ylabel("Down-switch field β (H/Hmax)", fontsize=12)
    ax.tick_params(labelsize=10)
    return im


def preisach_models_explanation(
    checkpoints,
    root,
    out,
    sample_index=0,
    perturbation_pct=5.0,
    h_channel=0,
    m_channel=1,
    seed_choice=None,
):
    """Separate presentation figure for all four Preisach variants."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.models.registry import get_spec
    from src.dataset_ml.datasets_ml import CurveSequenceDataset, VALID_LABELS
    from src import config

    names = [
        "preisach_mlp",
        "m_data_check_preisach_MLP",
        "preisach_rnn",
        "m_data_check_preisach_RNN",
    ]
    manifest = root / "dataset" / "ml_manifest.csv"
    if not manifest.is_file():
        return
    raw_spec = get_spec("rnn_joint")
    raw_ds = CurveSequenceDataset(manifest, split="test")
    if not 0 <= sample_index < len(raw_ds):
        raise ValueError("Preisach example index outside the test split")
    raw_batch = next(
        iter(
            DataLoader(
                Subset(raw_ds, [sample_index]),
                batch_size=1,
                collate_fn=raw_spec.collate_fn,
            )
        )
    )
    raw, lengths = raw_batch[0][0].cpu().numpy(), raw_batch[1][0].cpu().numpy().astype(
        int
    )
    truth = int(raw_batch[-1].reshape(-1)[0].item())
    temps = [float(t) for t in config.TEMPERATURES_K]
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(17.5, 15),
        gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=False,
    )
    records = []
    ok = 0
    for r, name in enumerate(names):
        ax_curve, ax_map = axes[r]
        ax_curve.set_title(MODEL_NAMES[name], fontsize=17, weight="bold", pad=8)
        if name not in checkpoints:
            ax_curve.text(
                0.5,
                0.5,
                "Checkpoint unavailable",
                ha="center",
                va="center",
                transform=ax_curve.transAxes,
            )
            ax_map.axis("off")
            continue
        try:
            spec = get_spec(name)
            ds = spec.dataset_cls(manifest, split="test")
            batch = next(
                iter(
                    DataLoader(
                        Subset(ds, [sample_index]),
                        batch_size=1,
                        collate_fn=spec.collate_fn,
                    )
                )
            )
            ensemble = []
            for seed, path in sorted(checkpoints[name].items()):
                if seed_choice is not None and seed != seed_choice:
                    continue
                ck = torch.load(path, map_location="cpu", weights_only=False)
                net = spec.build(ck["hparams"])
                net.load_state_dict(ck["model_state"])
                net.eval()
                ensemble.append((spec, net))
            explained = preisach_feature_ensemble_attribution(
                ensemble, batch, perturbation_pct
            )
            focus = explained["most_relevant_temperature"]
            plot_preisach_curve_importance(
                ax_curve, raw, lengths, explained, h_channel, m_channel
            )
            pred = int(explained["predicted_class"])
            classes = list(VALID_LABELS)
            pred_name = str(classes[pred]) if pred < len(classes) else str(pred)
            ax_curve.text(
                0.5,
                -0.29,
                f"Prediction: {pred_name}   ·   Focus: {temps[focus]:g} K",
                transform=ax_curve.transAxes,
                fontsize=11,
                color="#364A5B",
                ha="center",
                va="top",
                clip_on=False,
            )
            ax_curve.set_xlabel("Magnetic field", fontsize=11)
            ax_curve.set_ylabel("Magnetization", fontsize=11)
            ax_curve.grid(alpha=0.12)
            im = plot_preisach_feature_map(ax_map, explained, temps)
            fig.colorbar(
                im,
                ax=ax_map,
                shrink=0.82,
                pad=0.025,
                label="Mean |change in class probability|",
            )
            geo = preisach_switch_geometry(explained["importance"].shape[1])
            for t in range(explained["importance"].shape[0]):
                for feature, value in enumerate(explained["importance"][t]):
                    records.append(
                        {
                            "test_index": sample_index,
                            "model": name,
                            "temperature_K": temps[t],
                            "feature_index": feature,
                            "alpha_normalized": float(geo["alpha"][feature]),
                            "beta_normalized": float(geo["beta"][feature]),
                            "mean_absolute_probability_change": float(value),
                            "perturbation_pct": perturbation_pct,
                        }
                    )
            ok += 1
        except Exception as exc:
            print(f"[preisach explanation] {name}: {type(exc).__name__}: {exc}")
            ax_curve.text(
                0.5,
                0.5,
                "Explanation unavailable (see console)",
                ha="center",
                va="center",
                transform=ax_curve.transAxes,
            )
            ax_map.axis("off")
    if not ok:
        plt.close(fig)
        return
    fig.suptitle(
        "Preisach models: which coefficients affect the prediction?",
        fontsize=25,
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        "Orange marks influential switching-field regions (beta on descending, alpha on ascending). The map shows the real Preisach plane.",
        ha="center",
        fontsize=12,
        color="#526477",
    )
    fig.subplots_adjust(
        left=0.09, right=0.95, top=0.945, bottom=0.075, hspace=0.63, wspace=0.26
    )
    save(fig, out, "07_preisach_feature_explainability")
    if records:
        pd.DataFrame(records).to_csv(
            out / "07_preisach_feature_importance.csv", index=False
        )
        print("Saved:", out / "07_preisach_feature_importance.csv")


def preisach_feature_ensemble_attribution(
    spec_models, batch, perturbation_pct=5.0, batch_size=24
):
    """Local sensitivity of each Preisach coefficient for a seed ensemble.

    Returns an (n_temperatures, n_features) matrix of mean absolute changes to
    the ensemble probability of the originally predicted class. Uses symmetric
    +/- feature perturbations. These are *feature-space* interventions, not
    measured-loop perturbations and not SHAP/causal attributions.

    Works with PreisachFeatureDataset (2 tensors) and PreisachCurveDataset
    (5 tensors), using the model registry's actual forward method unchanged.
    The feature ordering is preserved; we do not assume an alpha/beta grid.
    """
    import torch

    if not spec_models:
        raise ValueError("No Preisach checkpoint models were provided")
    if not 0 < float(perturbation_pct) <= 25:
        raise ValueError("perturbation_pct must be in (0, 25]")
    device = next(spec_models[0][1].parameters()).device
    batch = tuple(
        a.detach().to(device) if isinstance(a, torch.Tensor) else a for a in batch
    )
    x = batch[0]
    if not isinstance(x, torch.Tensor) or x.shape[0] != 1 or x.ndim < 3:
        raise ValueError(
            f'Expected Preisach input (1, temperatures, features...), got {getattr(x, "shape", None)}'
        )
    n_temp = int(x.shape[1])
    n_feat = int(np.prod(x.shape[2:]))
    flat = x.reshape(1, n_temp, n_feat)

    def evaluate(inputs):
        count = inputs.shape[0]
        expanded = [inputs]
        for field in batch[1:]:
            if isinstance(field, torch.Tensor) and field.ndim and field.shape[0] == 1:
                expanded.append(field.repeat((count,) + (1,) * (field.ndim - 1)))
            else:
                expanded.append(field)
        scores = []
        with torch.inference_mode():
            for spec, model in spec_models:
                logits, _labels = spec.forward(model, tuple(expanded), device)
                if logits.shape[0] != count:
                    raise ValueError(
                        "The Preisach forward pass returned the wrong batch size"
                    )
                scores.append(torch.softmax(logits, dim=-1))
        return torch.stack(scores).mean(dim=0)

    base_probs = evaluate(x)[0].detach().cpu().numpy()
    pred = int(np.argmax(base_probs))
    base_p = float(base_probs[pred])
    scores = np.zeros((n_temp, n_feat), dtype=float)
    # A single per-temperature perturbation scale avoids treating a zero
    # coefficient as inherently unimportant. It uses the observed feature
    # magnitude, not an arbitrary zero-masking operation.
    scales = flat.detach().abs().cpu().numpy()[0]
    reference_scale = np.maximum(np.quantile(scales, 0.90, axis=1), 1e-5)
    feature_scale = (float(perturbation_pct) / 100.0) * reference_scale

    # Vectorized evaluations: two counterfactuals for every feature.
    for start in range(0, n_temp * n_feat, max(1, int(batch_size // 2))):
        index = list(
            range(start, min(n_temp * n_feat, start + max(1, int(batch_size // 2))))
        )
        count = len(index)
        candidates = flat.repeat(2 * count, 1, 1)
        for i, number in enumerate(index):
            t, f = divmod(number, n_feat)
            step = float(feature_scale[t])
            candidates[2 * i, t, f] += step
            candidates[2 * i + 1, t, f] -= step
        changed = evaluate(candidates.reshape((2 * count,) + tuple(x.shape[1:])))[
            :, pred
        ]
        changed = changed.detach().cpu().numpy().reshape(count, 2)
        impacts = np.abs(changed - base_p).mean(axis=1)
        for number, value in zip(index, impacts):
            t, f = divmod(number, n_feat)
            scores[t, f] = float(value)
    temperature_importance = scores.sum(axis=1)
    return {
        "importance": scores,
        "feature_shape": tuple(int(i) for i in x.shape[2:]),
        "temperatures": temperature_importance,
        "most_relevant_temperature": int(np.argmax(temperature_importance)),
        "ensemble_probs": base_probs,
        "predicted_class": pred,
        "perturbation_pct": float(perturbation_pct),
        "reference_feature_scale": reference_scale,
    }


def preisach_switch_geometry(n_features):
    """Exact feature order and switching fields from src.models.preisach.

    The ordering is (alpha index i, beta index j) for j <= i. The
    dimensionless switching fields are in units of H / Hmax, matching the
    normalized magnetic-field channel of CurveSequenceDataset.
    """
    from src.models.preisach import PreisachConfig

    cfg = PreisachConfig()
    n = int(cfg.n_nodes)
    if n * (n + 1) // 2 != int(n_features):
        raise ValueError(
            f"Preisach geometry mismatch: {n_features} features but "
            f"{n} nodes imply {n*(n+1)//2}. Check PreisachConfig."
        )
    u = np.linspace(-1.0, 1.0, n)
    g = np.sinh(cfg.node_warp * u) / np.sinh(cfg.node_warp) if cfg.node_warp > 0 else u
    nodes = cfg.node_max_frac * g  # normalized H
    ij = np.array([(i, j) for i in range(n) for j in range(i + 1)], dtype=int)
    return {
        "nodes": nodes,
        "alpha": nodes[ij[:, 0]],
        "beta": nodes[ij[:, 1]],
        "cell_i": ij[:, 0],
        "cell_j": ij[:, 1],
        "n_nodes": n,
    }


def preisach_curve_importance(
    raw_curves, lengths, attribution, h_channel=0, temperature=None, width=None
):
    """Map coefficient sensitivity to *switching fields* on measured branches.

    A Preisach hysteron (alpha,beta) switches DOWN at beta on the descending
    branch and UP at alpha on the ascending branch. We spread its sensitivity
    over nearby H samples on that branch, not over arbitrary pixel positions.
    This is a physically grounded switching-field correspondence, NOT an
    exact causal attribution to individual raw points.
    """
    imp = np.asarray(attribution["importance"], dtype=float)
    geo = preisach_switch_geometry(imp.shape[1])
    t = int(
        attribution["most_relevant_temperature"] if temperature is None else temperature
    )
    L = int(lengths[t])
    curve = np.asarray(raw_curves[t, :L], dtype=float)
    h = curve[:, h_channel]
    turn = int(np.argmin(h))
    val = np.zeros(L, dtype=float)
    if L < 5 or turn < 2 or turn >= L - 2:
        return val
    nodes = geo["nodes"]
    spacing = np.diff(nodes)
    bandwidth = float(
        width if width is not None else max(0.018, np.median(spacing) * 0.6)
    )
    # Marginals are the observable contribution of Preisach weights to a
    # major loop: beta controls the descending branch, alpha the ascending.
    down = np.bincount(geo["cell_j"], weights=imp[t], minlength=len(nodes))
    up = np.bincount(geo["cell_i"], weights=imp[t], minlength=len(nodes))

    def distribute(indices, marginal):
        hh = h[indices]
        if not len(hh):
            return
        distance = (hh[:, None] - nodes[None, :]) / bandwidth
        contributions = np.exp(-0.5 * distance**2) * marginal[None, :]
        val[indices] = contributions.sum(axis=1)

    distribute(np.arange(turn + 1), down)
    distribute(np.arange(turn + 1, L), up)
    return val


def plot_preisach_curve_importance(
    ax,
    raw_curves,
    lengths,
    attribution,
    h_channel=0,
    m_channel=1,
    highlight_fraction=0.75,
):
    """Plot original loops and orange segments near influential switch fields."""
    from matplotlib.collections import LineCollection

    importance = np.asarray(attribution["importance"])
    focus = int(attribution["most_relevant_temperature"])
    n_temp = min(len(lengths), len(importance))
    for t in range(n_temp):
        L = int(lengths[t])
        if L < 2:
            continue
        h = raw_curves[t, :L, h_channel]
        m = raw_curves[t, :L, m_channel]
        ax.plot(h, m, color="#CDD7E1", lw=2.4, zorder=1)
    L = int(lengths[focus])
    if L < 2:
        return False
    h = np.asarray(raw_curves[focus, :L, h_channel], float)
    m = np.asarray(raw_curves[focus, :L, m_channel], float)
    ax.plot(h, m, color="#7D94A8", lw=3.4, zorder=2)
    values = preisach_curve_importance(
        raw_curves, lengths, attribution, h_channel, focus
    )
    peak = float(np.max(values)) if len(values) else 0.0
    if peak > 1e-15:
        seg = (values[:-1] + values[1:]) * 0.5
        # If all switching regions matter nearly equally, don't exaggerate
        # tiny numerical differences as a sharply localized explanation.
        spread = float(np.quantile(seg, 0.9) - np.quantile(seg, 0.1))
        contrasted = spread > 0.15 * peak
        keep = (
            (seg >= peak * highlight_fraction)
            if contrasted
            else np.zeros_like(seg, dtype=bool)
        )
        indices = np.where(keep)[0]
        if len(indices):
            pts = np.column_stack((h, m))
            segs = np.stack((pts[indices], pts[indices + 1]), axis=1)
            widths = 4.0 + 4.0 * seg[indices] / peak
            lc = LineCollection(segs, colors="#F07832", linewidths=widths, zorder=4)
            lc.set_capstyle("round")
            ax.add_collection(lc)
    ax.grid(alpha=0.11)
    return bool(
        peak > 1e-15
        and np.any((values[:-1] + values[1:]) * 0.5 >= peak * highlight_fraction)
    )


def plot_preisach_feature_map(ax, attribution, temperature_labels=None, max_columns=36):
    """True triangular Preisach plane for the most sensitive temperature.

    Horizontal: alpha (up-switch field); vertical: beta (down-switch field).
    Only alpha >= beta cells exist. Colors represent absolute prediction
    sensitivity to coefficient perturbation, not the trained input weights.
    """
    scores = np.asarray(attribution["importance"], float)
    geo = preisach_switch_geometry(scores.shape[1])
    focus = int(attribution["most_relevant_temperature"])
    n = geo["n_nodes"]
    plane = np.full((n, n), np.nan)
    plane[geo["cell_j"], geo["cell_i"]] = scores[focus]
    extent = (geo["nodes"][0], geo["nodes"][-1], geo["nodes"][0], geo["nodes"][-1])
    cm = plt.get_cmap("YlOrBr").copy()
    cm.set_bad("#F2F5F8")
    im = ax.imshow(
        np.ma.masked_invalid(plane),
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
        cmap=cm,
        vmin=0,
        vmax=max(float(np.nanmax(plane)), 1e-12),
    )
    ax.plot([extent[0], extent[1]], [extent[0], extent[1]], color="#7E8A96", lw=0.8)
    label = (
        f"{temperature_labels[focus]:g} K"
        if temperature_labels is not None
        else f"Temperature {focus}"
    )
    ax.set_title(f"Preisach plane · {label}", fontsize=15, weight="bold", pad=9)
    ax.set_xlabel("Up-switch field α (H/Hmax)", fontsize=12)
    ax.set_ylabel("Down-switch field β (H/Hmax)", fontsize=12)
    ax.tick_params(labelsize=10)
    return im


def preisach_models_explanation(
    checkpoints,
    root,
    out,
    sample_index=0,
    perturbation_pct=5.0,
    h_channel=0,
    m_channel=1,
    seed_choice=None,
):
    """Separate presentation figure for all four Preisach variants."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.models.registry import get_spec
    from src.dataset_ml.datasets_ml import CurveSequenceDataset, VALID_LABELS
    from src import config

    names = [
        "preisach_mlp",
        "m_data_check_preisach_MLP",
        "preisach_rnn",
        "m_data_check_preisach_RNN",
    ]
    manifest = root / "dataset" / "ml_manifest.csv"
    if not manifest.is_file():
        return
    raw_spec = get_spec("rnn_joint")
    raw_ds = CurveSequenceDataset(manifest, split="test")
    if not 0 <= sample_index < len(raw_ds):
        raise ValueError("Preisach example index outside the test split")
    raw_batch = next(
        iter(
            DataLoader(
                Subset(raw_ds, [sample_index]),
                batch_size=1,
                collate_fn=raw_spec.collate_fn,
            )
        )
    )
    raw, lengths = raw_batch[0][0].cpu().numpy(), raw_batch[1][0].cpu().numpy().astype(
        int
    )
    truth = int(raw_batch[-1].reshape(-1)[0].item())
    temps = [float(t) for t in config.TEMPERATURES_K]
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(17.5, 15),
        gridspec_kw={"width_ratios": [1.15, 1]},
        constrained_layout=False,
    )
    records = []
    ok = 0
    for r, name in enumerate(names):
        ax_curve, ax_map = axes[r]
        ax_curve.set_title(MODEL_NAMES[name], fontsize=17, weight="bold", pad=8)
        if name not in checkpoints:
            ax_curve.text(
                0.5,
                0.5,
                "Checkpoint unavailable",
                ha="center",
                va="center",
                transform=ax_curve.transAxes,
            )
            ax_map.axis("off")
            continue
        try:
            spec = get_spec(name)
            ds = spec.dataset_cls(manifest, split="test")
            batch = next(
                iter(
                    DataLoader(
                        Subset(ds, [sample_index]),
                        batch_size=1,
                        collate_fn=spec.collate_fn,
                    )
                )
            )
            ensemble = []
            for seed, path in sorted(checkpoints[name].items()):
                if seed_choice is not None and seed != seed_choice:
                    continue
                ck = torch.load(path, map_location="cpu", weights_only=False)
                net = spec.build(ck["hparams"])
                net.load_state_dict(ck["model_state"])
                net.eval()
                ensemble.append((spec, net))
            explained = preisach_feature_ensemble_attribution(
                ensemble, batch, perturbation_pct
            )
            focus = explained["most_relevant_temperature"]
            plot_preisach_curve_importance(
                ax_curve, raw, lengths, explained, h_channel, m_channel
            )
            pred = int(explained["predicted_class"])
            classes = list(VALID_LABELS)
            pred_name = str(classes[pred]) if pred < len(classes) else str(pred)
            ax_curve.text(
                0.5,
                -0.29,
                f"Prediction: {pred_name}   ·   Focus: {temps[focus]:g} K",
                transform=ax_curve.transAxes,
                fontsize=11,
                color="#364A5B",
                ha="center",
                va="top",
                clip_on=False,
            )
            ax_curve.set_xlabel("Magnetic field", fontsize=11)
            ax_curve.set_ylabel("Magnetization", fontsize=11)
            ax_curve.grid(alpha=0.12)
            im = plot_preisach_feature_map(ax_map, explained, temps)
            fig.colorbar(
                im,
                ax=ax_map,
                shrink=0.82,
                pad=0.025,
                label="Mean |change in class probability|",
            )
            geo = preisach_switch_geometry(explained["importance"].shape[1])
            for t in range(explained["importance"].shape[0]):
                for feature, value in enumerate(explained["importance"][t]):
                    records.append(
                        {
                            "test_index": sample_index,
                            "model": name,
                            "temperature_K": temps[t],
                            "feature_index": feature,
                            "alpha_normalized": float(geo["alpha"][feature]),
                            "beta_normalized": float(geo["beta"][feature]),
                            "mean_absolute_probability_change": float(value),
                            "perturbation_pct": perturbation_pct,
                        }
                    )
            ok += 1
        except Exception as exc:
            print(f"[preisach explanation] {name}: {type(exc).__name__}: {exc}")
            ax_curve.text(
                0.5,
                0.5,
                "Explanation unavailable (see console)",
                ha="center",
                va="center",
                transform=ax_curve.transAxes,
            )
            ax_map.axis("off")
    if not ok:
        plt.close(fig)
        return
    fig.suptitle(
        "Preisach models: which coefficients affect the prediction?",
        fontsize=25,
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.018,
        "Orange marks influential switching-field regions (beta on descending, alpha on ascending). The map shows the real Preisach plane.",
        ha="center",
        fontsize=12,
        color="#526477",
    )
    fig.subplots_adjust(
        left=0.09, right=0.95, top=0.945, bottom=0.075, hspace=0.63, wspace=0.26
    )
    save(fig, out, "07_preisach_feature_explainability")
    if records:
        pd.DataFrame(records).to_csv(
            out / "07_preisach_feature_importance.csv", index=False
        )
        print("Saved:", out / "07_preisach_feature_importance.csv")


def example_loop_explanations(
    runs,
    root,
    out,
    sample_index=0,
    temperature_index=0,
    seed_choice=None,
    h_channel=0,
    m_channel=1,
    perturbation_pct=5.0,
    highlight_threshold=0.85,
    ig_segments=10,
    ig_reference_index=0,
    ig_steps=48,
):
    """RNN ensemble Integrated Gradients and MLP smooth perturbation explanations.

    Integrated Gradients is computed across chronological M segments for the two raw-loop
    RNNs, using a real training loop as reference. MLPs retain the existing
    approximate raster deformation sensitivity.
    """
    import torch
    from torch.utils.data import DataLoader
    from src.dataset_ml.datasets_ml import CurveSequenceDataset, VALID_LABELS
    from src.models.registry import get_spec
    from src import config

    n_temp = len(config.TEMPERATURES_K)
    manifest = root / "dataset" / "ml_manifest.csv"
    if not manifest.exists():
        print("Example explanation skipped: missing dataset/ml_manifest.csv")
        return

    raw_ds = CurveSequenceDataset(manifest, split="test")
    if not 0 <= sample_index < len(raw_ds):
        raise ValueError(f"--example-index must be 0..{len(raw_ds)-1}")
    raw_spec = get_spec("rnn_joint")
    raw_loader = DataLoader(
        torch.utils.data.Subset(raw_ds, [sample_index]),
        batch_size=1,
        shuffle=False,
        collate_fn=raw_spec.collate_fn,
    )
    raw_batch = next(iter(raw_loader))
    raw_curves = raw_batch[0][0].detach().cpu().numpy()  # (T, L, F)
    raw_lengths = raw_batch[1][0].detach().cpu().numpy().astype(int)
    truth_idx = int(raw_batch[-1].reshape(-1)[0].item())
    truth_name = (
        str(VALID_LABELS[truth_idx])
        if 0 <= truth_idx < len(VALID_LABELS)
        else str(truth_idx)
    )
    if raw_curves.ndim != 3 or max(h_channel, m_channel) >= raw_curves.shape[2]:
        raise ValueError(
            "Raw curve channels are incompatible with requested H/M indices."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen_models = ["mlp", "m_data_check_MLP", "rnn_joint", "m_data_check_RNN"]
    prepared = {}
    diagnostic_rows = []
    if not (0 < float(perturbation_pct) <= 25):
        raise ValueError("--perturbation-pct must be greater than 0 and at most 25")
    if not 0.0 < float(highlight_threshold) <= 1.0:
        raise ValueError("--highlight-threshold must be between 0 and 1")

    def margin_value(logits, predicted):
        # Class margin remains informative when softmax is close to 100%.
        others = logits[0].clone()
        others[predicted] = -torch.inf
        return float((logits[0, predicted] - torch.logsumexp(others, dim=0)).item())

    def perturb_magnetization(x0, name, t, segment, sign, base_scale):
        """One localized, smooth, small M displacement, never a zero mask.

        segment is a center position between 0 and 1 along the sampled loop.
        The raster deformation for MLPs assumes H maps horizontally and M
        vertically; an exact mapping requires the original rasterizer.
        """
        import torch.nn.functional as F

        x = x0.clone()
        L = int(raw_lengths[t])
        if L < 3:
            return x
        pos = np.arange(L, dtype=float) / max(1, L - 1)
        sigma = 0.055  # Gaussian along acquisition order, about 13% FWHM.
        window = np.exp(-0.5 * ((pos - segment) / sigma) ** 2)
        magnitude = float(sign) * float(base_scale)
        if name in ("rnn_joint", "m_data_check_RNN"):
            local_len = min(int(x.shape[2]), L)
            displacement = torch.as_tensor(
                magnitude * window[:local_len], dtype=x.dtype, device=x.device
            )
            x[0, t, :local_len, m_channel] += displacement
            return x

        # MLP: input shape (B, temperature, height, width). Create a smooth
        # vertical raster warp around the *same* measured physical segment.
        import torch.nn.functional as F

        img = x[0, t]
        H, W = img.shape
        # Use global axis bounds, matching the approximate projection used
        # in previous figures, rather than per-temperature rescaling.
        good = [j for j in range(n_temp) if raw_lengths[j] > 0]
        all_h = np.concatenate(
            [raw_curves[j, : int(raw_lengths[j]), h_channel] for j in good]
        )
        all_m = np.concatenate(
            [raw_curves[j, : int(raw_lengths[j]), m_channel] for j in good]
        )
        h_span = max(float(np.ptp(all_h)), 1e-12)
        m_span = max(float(np.ptp(all_m)), 1e-12)
        hs = raw_curves[t, :L, h_channel]
        ms = raw_curves[t, :L, m_channel]
        sel = np.where(window > 0.08)[0][:: max(1, L // 40)]
        if len(sel) == 0:
            return x
        cc = (hs[sel] - float(all_h.min())) / h_span * (W - 1)
        rr = (float(all_m.max()) - ms[sel]) / m_span * (H - 1)
        win = window[sel]
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=img.dtype, device=img.device),
            torch.arange(W, dtype=img.dtype, device=img.device),
            indexing="ij",
        )
        # Smooth spatial support: nearby curve pixels get shifted together,
        # avoiding harsh boundaries/artifacts in the input raster.
        dist2 = (
            yy[..., None] - torch.as_tensor(rr, device=img.device, dtype=img.dtype)
        ) ** 2 + (
            xx[..., None] - torch.as_tensor(cc, device=img.device, dtype=img.dtype)
        ) ** 2
        width_px = max(2.0, min(H, W) * 0.055)
        weights = torch.exp(-0.5 * dist2 / (width_px**2)) * torch.as_tensor(
            win, device=img.device, dtype=img.dtype
        )
        weight = weights.max(dim=-1).values
        dy_px = magnitude / m_span * (H - 1) * weight
        # Pullback sampling: move the visible raster content up for M increase.
        gx = 2 * xx / max(W - 1, 1) - 1
        gy = 2 * (yy + dy_px) / max(H - 1, 1) - 1
        grid = torch.stack((gx, gy), dim=-1)[None]
        deformed = F.grid_sample(
            img[None, None],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[0, 0]
        x[0, t] = deformed
        return x

    for name in chosen_models:
        if name not in runs:
            continue
        try:
            chosen_seed = explanation_seed(runs[name], seed_choice)
            ck_path = runs[name][chosen_seed]
            spec = get_spec(name)
            ds = spec.dataset_cls(manifest, split="test")
            loader = DataLoader(
                torch.utils.data.Subset(ds, [sample_index]),
                batch_size=1,
                shuffle=False,
                collate_fn=spec.collate_fn,
            )
            batch = next(iter(loader))
            batch = tuple(x.to(device) if hasattr(x, "to") else x for x in batch)
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            net = spec.build(ck["hparams"]).to(device)
            net.load_state_dict(ck["model_state"])
            net.eval()

            with torch.inference_mode():
                logits, labels = spec.forward(net, batch, device)
                predicted = int(logits[0].argmax().item())
                confidence = float(torch.softmax(logits, dim=1)[0, predicted].item())
                baseline_margin = margin_value(logits, predicted)
            if int(labels.reshape(-1)[0].item()) != truth_idx:
                raise ValueError("Labels are not aligned across dataset types")

            if name in ("rnn_joint", "m_data_check_RNN"):
                # Integrated Gradients explains the ENSEMBLE log probability. Use all
                # available seeds unless --physics-seed explicitly requests one.
                seed_ids = (
                    [seed_choice] if seed_choice is not None else sorted(runs[name])
                )
                ensemble_models = [(spec, net)]
                for other_seed in seed_ids:
                    if other_seed == chosen_seed:
                        continue
                    other_ck = torch.load(
                        runs[name][other_seed], map_location=device, weights_only=False
                    )
                    other_model = spec.build(other_ck["hparams"]).to(device)
                    other_model.load_state_dict(other_ck["model_state"])
                    other_model.eval()
                    ensemble_models.append((spec, other_model))
                train_ds = spec.dataset_cls(manifest, split="train")
                if not 0 <= ig_reference_index < len(train_ds):
                    raise ValueError("IG reference index is outside train split")
                ref_batch = next(
                    iter(
                        DataLoader(
                            torch.utils.data.Subset(train_ds, [ig_reference_index]),
                            batch_size=1,
                            collate_fn=spec.collate_fn,
                        )
                    )
                )
                explained = integrated_gradients_rnn_ensemble_attribution(
                    ensemble_models,
                    batch,
                    ref_batch,
                    n_segments=ig_segments,
                    n_steps=ig_steps,
                    m_channel=m_channel,
                )
                top_temp = explained["most_relevant_temperature"]
                values_by_temp = explained["temperatures"]
                sv = values_by_temp[top_temp]["integrated_gradients"]
                L = int(raw_lengths[top_temp])
                progress = np.linspace(0, 1, L)
                centers_shap = (np.arange(len(sv)) + 0.5) / len(sv)
                # Positive IG supports the predicted class relative to the baseline; negative
                # IG opposes it. Plot both with different colors.
                values = np.interp(progress, centers_shap, sv, left=sv[0], right=sv[-1])
                probs = explained["ensemble_probs"]
                pred = int(np.argmax(probs))
                prepared[name] = {
                    "kind": "integrated_gradients",
                    "pred": pred,
                    "confidence": float(probs[pred]),
                    "top_temp": top_temp,
                    "temp_importance": np.array(
                        [
                            np.sum(np.abs(values_by_temp[t]["integrated_gradients"]))
                            for t in sorted(values_by_temp)
                        ]
                    ),
                    "h": raw_curves[top_temp, :L, h_channel].astype(float),
                    "m": raw_curves[top_temp, :L, m_channel].astype(float),
                    "values": values,
                }
                for t, score_info in values_by_temp.items():
                    for segment, val in enumerate(
                        score_info["integrated_gradients"], start=1
                    ):
                        diagnostic_rows.append(
                            {
                                "test_index": sample_index,
                                "model": name,
                                "temperature_K": float(config.TEMPERATURES_K[t]),
                                "segment": segment,
                                "method": "Integrated Gradients",
                                "ig_log_probability": float(val),
                                "reference_train_index": ig_reference_index,
                                "seed_count": len(ensemble_models),
                            }
                        )
                continue

            # Same physical segment centers, Gaussian width, perturbation
            # amplitude, and +/- repetitions for every architecture.
            x0 = batch[0].detach().clone()
            n_segments = 16
            centers = (np.arange(n_segments) + 0.5) / n_segments
            # Configurable displacement relative to this test sample's M range.
            m_ranges = [
                float(np.ptp(raw_curves[j, : int(raw_lengths[j]), m_channel]))
                for j in range(n_temp)
                if raw_lengths[j] > 1
            ]
            amplitude = (float(perturbation_pct) / 100.0) * max(
                max(m_ranges, default=0.0), 1e-12
            )
            scores = np.zeros((n_temp, n_segments), dtype=float)
            for t in range(n_temp):
                L = int(raw_lengths[t])
                if L < 3:
                    continue
                for k, center in enumerate(centers):
                    effects = []
                    for sign in (-1, +1):
                        changed = perturb_magnetization(
                            x0, name, t, center, sign, amplitude
                        )
                        with torch.inference_mode():
                            altered_logits, _ = spec.forward(
                                net, (changed, *batch[1:]), device
                            )
                        effects.append(
                            abs(
                                margin_value(altered_logits, predicted)
                                - baseline_margin
                            )
                        )
                    scores[t, k] = float(np.mean(effects))
            for t in range(n_temp):
                for k, center in enumerate(centers):
                    diagnostic_rows.append(
                        {
                            "test_index": sample_index,
                            "model": name,
                            "temperature_K": float(config.TEMPERATURES_K[t]),
                            "segment": k + 1,
                            "segment_center_fraction": float(center),
                            "perturbation_pct": float(perturbation_pct),
                            "mean_abs_margin_change": float(scores[t, k]),
                        }
                    )
            per_temp = np.mean(scores, axis=1)
            top_temp = int(np.argmax(per_temp))
            L = int(raw_lengths[top_temp])
            h = raw_curves[top_temp, :L, h_channel].astype(float)
            m = raw_curves[top_temp, :L, m_channel].astype(float)
            progress = np.linspace(0, 1, L)
            # Interpolate discrete segment sensitivities onto plotted points.
            values = np.interp(
                progress,
                centers,
                scores[top_temp],
                left=scores[top_temp, 0],
                right=scores[top_temp, -1],
            )
            # Report whether the final quarter truly dominates. These numbers
            # are measured before any per-panel visual normalization.
            final_q = float(np.mean(scores[top_temp, -4:]))
            first_q = float(np.mean(scores[top_temp, :4]))
            all_mean = float(np.mean(scores[top_temp]))
            print(
                f"[explain] {name}: selected {config.TEMPERATURES_K[top_temp]:g} K; "
                f"first-quarter={first_q:.5g}, final-quarter={final_q:.5g}, "
                f'final/first={(final_q / first_q if first_q > 1e-12 else float("nan")):.2f}, '
                f"overall mean={all_mean:.5g} (margin units)"
            )
            projected = name in ("mlp", "m_data_check_MLP")

            prepared[name] = {
                "kind": "curve",
                "pred": predicted,
                "confidence": confidence,
                "top_temp": top_temp,
                "temp_importance": np.array(per_temp),
                "h": h,
                "m": m,
                "values": values,
                "focus_idx": int(np.argmax(values)) if len(values) else 0,
                "projected": projected,
            }
        except Exception as exc:
            print(
                f"  Smooth-perturbation explanation unavailable for {name}: {type(exc).__name__}: {exc}"
            )

    if diagnostic_rows:
        diag_path = out / "04_segment_sensitivity.csv"
        pd.DataFrame(diagnostic_rows).to_csv(diag_path, index=False)
        print("[explain] Segment sensitivity saved:", diag_path)

    if not prepared:
        print("Example explanation skipped: no compatible checkpoints were available.")
        return

    fig = plt.figure(figsize=(18.8, 12.8), facecolor="white")
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.77, 1, 1],
        left=0.07,
        right=0.97,
        top=0.89,
        bottom=0.17,
        hspace=0.62,
        wspace=0.21,
    )

    # Top overview: all input loops, simple and uncluttered.
    ax_top = fig.add_subplot(outer[0, :])
    colors = ["#5B2A86", "#2A9D8F", "#E6B800", "#CC7722", "#4C78A8", "#D1495B"]
    for t in range(n_temp):
        L = int(raw_lengths[t])
        if L < 1:
            continue
        hh = raw_curves[t, :L, h_channel]
        mm = raw_curves[t, :L, m_channel]
        ax_top.plot(
            hh,
            mm,
            lw=2.4,
            color=colors[t % len(colors)],
            label=f"{config.TEMPERATURES_K[t]:g} K",
        )
    ax_top.set_title(
        f"All input loops   |   True class: {truth_name}",
        fontsize=20,
        weight="bold",
        pad=12,
    )
    ax_top.set_xlabel("Magnetic field", fontsize=15)
    ax_top.set_ylabel("Magnetization", fontsize=15)
    ax_top.legend(loc="upper left", ncol=min(6, n_temp), fontsize=11, frameon=False)
    ax_top.grid(alpha=0.12)
    ax_top.tick_params(labelsize=12)

    layout = [
        ("mlp", 0, 0),
        ("m_data_check_MLP", 0, 1),
        ("rnn_joint", 1, 0),
        ("m_data_check_RNN", 1, 1),
    ]
    for name, rr, cc in layout:
        cell = outer[rr + 1, cc].subgridspec(
            2, 1, height_ratios=[4.6, 1.55], hspace=0.20
        )
        ax = fig.add_subplot(cell[0])
        ax_caption = fig.add_subplot(cell[1])
        ax_caption.axis("off")

        ax.set_title(
            MODEL_NAMES.get(name, name), loc="left", fontsize=20, weight="bold", pad=10
        )
        if name not in prepared:
            ax.text(
                0.5,
                0.5,
                "Explanation unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
                color="#697783",
            )
            ax.axis("off")
            continue

        item = prepared[name]
        pred = item["pred"]
        pred_name = str(VALID_LABELS[pred]) if pred < len(VALID_LABELS) else str(pred)
        correct = pred == truth_idx
        top_temp = item["top_temp"]

        # Show all loops faintly, and then draw the most relevant loop itself
        # as a thicker colour-coded curve. This keeps the highlight *on* the
        # input trajectory rather than around it.
        from matplotlib.collections import LineCollection

        for t in range(n_temp):
            L = int(raw_lengths[t])
            if L < 1:
                continue
            hh = raw_curves[t, :L, h_channel]
            mm = raw_curves[t, :L, m_channel]
            ax.plot(hh, mm, color="#D5DCE3", lw=2.2, zorder=1)

        hh = item["h"]
        mm = item["m"]
        vals = np.asarray(item["values"], dtype=float)
        vmax = max(float(np.max(vals)), 1e-12)
        # Highlight only unambiguously strong peaks. A local maximum always
        # exists, even on a nearly flat profile, so concentration matters too.
        # Relative thresholds are per model: raw margin units are NOT directly
        # comparable between architectures.
        signed_shap = item["kind"] == "integrated_gradients"
        if signed_shap:
            vmax = max(float(np.max(np.abs(vals))), 1e-12)
        valid = (
            np.abs(vals[np.isfinite(vals)]) if signed_shap else vals[np.isfinite(vals)]
        )
        median = float(np.median(valid)) if valid.size else 0.0
        is_concentrated = vmax > 1e-9 and vmax >= max(1.8 * median, 1e-9)

        # Draw the whole measured loop in neutral grey. Never colour weak parts.
        ax.plot(hh, mm, color="#A8B8C9", lw=4.6, solid_capstyle="round", zorder=2)

        if (is_concentrated or (signed_shap and vmax > 1e-12)) and len(hh) >= 2:
            pts = np.column_stack([hh, mm]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_vals = 0.5 * (vals[:-1] + vals[1:])
            # Absolute significance isn't implied: this only identifies segments
            # above the chosen fraction of the model's strongest sensitivity.
            if signed_shap:
                # Show all signed temporal IG contributions proportionally;
                # previously, a hard cutoff hid every segment but the maximum.
                from matplotlib.colors import to_rgba

                weights = np.clip(np.abs(seg_vals) / vmax, 0, 1)
                rgba = []
                for value, weight in zip(seg_vals, weights):
                    r, g, b, _ = to_rgba("#E86A24" if value >= 0 else "#2985B8")
                    rgba.append((r, g, b, float(0.16 + 0.84 * weight)))
                lc = LineCollection(
                    segs, colors=rgba, linewidths=3.0 + 4.5 * weights, zorder=4
                )
                lc.set_capstyle("round")
                ax.add_collection(lc)
            else:
                strong = np.isfinite(seg_vals) & (
                    seg_vals >= highlight_threshold * vmax
                )
                if np.any(strong):
                    lc = LineCollection(
                        segs[strong], colors="#E86A24", linewidths=7.5, zorder=4
                    )
                    lc.set_capstyle("round")
                    ax.add_collection(lc)
                else:
                    is_concentrated = False
        else:
            is_concentrated = False

        if not is_concentrated:
            ax_caption.text(
                0.5,
                0.01,
                "No clearly dominant segment",
                ha="center",
                va="bottom",
                fontsize=10.5,
                color="#667381",
            )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.08)

        pred_line = (
            f'Predicted: {pred_name}  ·  {item["confidence"]:.0%} confident  ·  '
            + ("Correct" if correct else "Incorrect")
        )
        temp_line = f"Most affected loop: {config.TEMPERATURES_K[top_temp]:g} K"
        ax_caption.text(
            0.5,
            0.70,
            pred_line,
            ha="center",
            va="center",
            fontsize=11.5,
            color="#334E62",
        )
        ax_caption.text(
            0.5,
            0.20,
            temp_line,
            ha="center",
            va="center",
            fontsize=11.5,
            weight="bold",
            color="#4B6577",
        )

    fig.suptitle(
        "What part of the input does each model use?",
        fontsize=30,
        weight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.080,
        "RNNs: orange increases and blue decreases the ensemble log probability relative to a training reference.",
        ha="center",
        fontsize=13,
        color="#4C6070",
    )
    fig.text(
        0.5,
        0.055,
        f"RNN: Integrated Gradients  ·  Neural network: smooth perturbations (±{perturbation_pct:g}%)  ·  Not a causal explanation.",
        ha="center",
        fontsize=12.0,
        color="#63717C",
    )
    save(fig, out, "04_interpretability_baseline_vs_physics")


def evaluate_train_test_checkpoints(root, checkpoints, out, batch_size=64):
    """Evaluate each saved model on train + test, with retry and diagnostics.

    Evaluate exactly the registry's dataset and forward, including the special
    Preisach+Physics five-field batch; do not substitute test-only metrics.
    """
    import gc
    import traceback
    import torch
    from torch.utils.data import DataLoader
    from src.models.registry import get_spec
    from src.dataset_ml.datasets_ml import VALID_LABELS

    manifest = root / "dataset" / "ml_manifest.csv"
    if not manifest.is_file():
        print("[all-data] Missing dataset/ml_manifest.csv; skipping.")
        return {}, {}, []

    results, predictions, audit, diagnostics = {}, {}, [], []
    primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ordered_models = [m for m in DEFAULT_MODELS if m in checkpoints]
    for name in DEFAULT_MODELS:
        if name not in checkpoints:
            expected_dir = root / "checkpoints" / name
            msg = f"No seed_*.pt files found in {expected_dir}"
            print(f"[all-data] {name}: {msg}")
            diagnostics.append(
                dict(
                    model=name,
                    seed="",
                    status="missing_checkpoint",
                    details=msg,
                    attempted="",
                )
            )
    if not ordered_models:
        pd.DataFrame(diagnostics).to_csv(
            out / "05_train_test_diagnostics.csv", index=False
        )
        print("[all-data] No checkpoints found; skipping.")
        return {}, {}, list(VALID_LABELS)

    for name in ordered_models:
        try:
            spec = get_spec(name)
            datasets = {
                split: spec.dataset_cls(manifest, split=split)
                for split in ("train", "test")
            }
            for split, ds in datasets.items():
                if not len(ds):
                    raise ValueError(f"Empty {split} dataset")
        except Exception as exc:
            print(f"[all-data] {name}: could not prepare train/test: {exc}")
            diagnostics.append(
                dict(
                    model=name,
                    seed="",
                    status="dataset_error",
                    details=f"{type(exc).__name__}: {exc}",
                    attempted="",
                )
            )
            continue

        successful = {}
        # Fall back to smaller batches and CPU to accommodate model-dependent
        # memory usage and collation. Each retry starts with a fresh model.
        attempts = []
        for dev in (primary_device, torch.device("cpu")):
            for bs in (max(1, int(batch_size)), min(16, max(1, int(batch_size))), 1):
                key = (str(dev), bs)
                if key not in attempts:
                    attempts.append(key)

        for seed, checkpoint in sorted(checkpoints[name].items()):
            failures = []
            split_pred = None
            for device_str, bs in attempts:
                device = torch.device(device_str)
                net = None
                try:
                    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    net = spec.build(ck["hparams"]).to(device)
                    net.load_state_dict(ck["model_state"])
                    net.eval()
                    candidate = {}
                    with torch.inference_mode():
                        for split in ("train", "test"):
                            loader = DataLoader(
                                datasets[split],
                                batch_size=bs,
                                shuffle=False,
                                collate_fn=spec.collate_fn,
                            )
                            probs, labels = [], []
                            for batch in loader:
                                logits, y = spec.forward(net, batch, device)
                                probs.append(
                                    torch.softmax(logits, dim=1).detach().cpu().numpy()
                                )
                                labels.append(y.detach().cpu().numpy().reshape(-1))
                            candidate[split] = (
                                np.concatenate(probs),
                                np.concatenate(labels),
                            )
                    split_pred = candidate
                    if failures:
                        print(
                            f"[all-data] {name}, seed {seed}: recovered on {device_str}, batch={bs}"
                        )
                    diagnostics.append(
                        dict(
                            model=name,
                            seed=seed,
                            status="ok",
                            details="",
                            attempted=f"{device_str}, batch={bs}",
                        )
                    )
                    break
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"
                    failures.append(f"{device_str}/batch={bs}: {detail}")
                    if len(failures) == 1:
                        print(
                            f"[all-data] {name}, seed {seed}: first attempt failed: {detail}"
                        )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                finally:
                    if net is not None:
                        del net
                    gc.collect()
            if split_pred is None:
                detail = " | ".join(failures)
                diagnostics.append(
                    dict(
                        model=name,
                        seed=seed,
                        status="inference_error",
                        details=detail,
                        attempted="all attempts",
                    )
                )
                print(
                    f"[all-data] {name}, seed {seed}: ALL attempts failed. "
                    f"See results/presentation/05_train_test_diagnostics.csv"
                )
                continue

            joined_prob = np.concatenate(
                [split_pred[x][0] for x in ("train", "test")], axis=0
            )
            joined_truth = np.concatenate(
                [split_pred[x][1] for x in ("train", "test")], axis=0
            )
            if successful and not np.array_equal(
                joined_truth, next(iter(successful.values()))[1]
            ):
                diagnostics.append(
                    dict(
                        model=name,
                        seed=seed,
                        status="order_mismatch",
                        details="Train/test labels differ across seeds",
                        attempted="",
                    )
                )
                continue
            successful[seed] = (joined_prob, joined_truth)
            for split, (pr, yy) in split_pred.items():
                audit.append(
                    dict(
                        model=name,
                        seed=seed,
                        split=split,
                        n_samples=len(yy),
                        **scores(pr, yy),
                    )
                )
            audit.append(
                dict(
                    model=name,
                    seed=seed,
                    split="train+test",
                    n_samples=len(joined_truth),
                    **scores(joined_prob, joined_truth),
                )
            )

        if not successful:
            print(f"[all-data] {name}: no successful train+test predictions")
            continue
        summaries = {
            key: np.asarray(
                [scores(prob, truth)[key] * 100 for prob, truth in successful.values()]
            )
            for key, _ in METRICS
        }
        results[name] = {
            key: (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
            for key, vals in summaries.items()
        }
        truth = next(iter(successful.values()))[1]
        ensemble_prob = np.mean([pair[0] for pair in successful.values()], axis=0)
        predictions[name] = (truth, ensemble_prob.argmax(axis=1), len(successful))
        print(
            f"[all-data] {name}: {len(truth)} train+test samples, "
            f'{len(successful)} seeds, accuracy={results[name]["accuracy"][0]:.1f}%'
        )

    pd.DataFrame(
        diagnostics, columns=["model", "seed", "status", "details", "attempted"]
    ).to_csv(out / "05_train_test_diagnostics.csv", index=False)
    if audit:
        pd.DataFrame(audit).to_csv(
            out / "05_train_test_per_seed_metrics.csv", index=False
        )
    return results, predictions, list(VALID_LABELS)


def plot_train_test_confusion(predictions, class_names, out):
    """One 2×4 panel, seed-ensemble confusion, rows = true class."""
    if not predictions:
        return
    groups = [
        (
            "Neural network",
            ["mlp", "m_data_check_MLP", "preisach_mlp", "m_data_check_preisach_MLP"],
        ),
        (
            "RNN",
            [
                "rnn_joint",
                "m_data_check_RNN",
                "preisach_rnn",
                "m_data_check_preisach_RNN",
            ],
        ),
    ]
    ids = np.arange(len(class_names))
    labels = ["Baseline", "+ Physics", "+ Preisach", "+ Physics + Preisach"]
    fig, axes = plt.subplots(2, 4, figsize=(21, 11.3))
    csv_rows = []
    im = None
    for row, (fam, models) in enumerate(groups):
        for col, name in enumerate(models):
            ax = axes[row, col]
            ax.set_title(f"{fam}\n{labels[col]}", fontsize=16, weight="bold", pad=12)
            if name not in predictions:
                ax.text(
                    0.5,
                    0.5,
                    "Not available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            truth, pred, n_seeds = predictions[name]
            cm = confusion_matrix(truth, pred, labels=ids)
            denom = cm.sum(axis=1, keepdims=True)
            norm = np.divide(
                cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0
            )
            im = ax.imshow(norm * 100, cmap="Blues", vmin=0, vmax=100, aspect="equal")
            for i in range(len(ids)):
                for j in range(len(ids)):
                    value = int(cm[i, j])
                    ax.text(
                        j,
                        i,
                        f"{norm[i,j]*100:.0f}%\n({value})",
                        ha="center",
                        va="center",
                        fontsize=12,
                        color="white" if norm[i, j] >= 0.55 else "#1C3447",
                    )
                    csv_rows.append(
                        dict(
                            model=name,
                            true_class=str(class_names[i]),
                            predicted_class=str(class_names[j]),
                            count=value,
                            row_percent=100 * norm[i, j],
                            n_seeds=n_seeds,
                        )
                    )
            ax.set_xticks(ids, [str(x) for x in class_names], fontsize=11)
            ax.set_yticks(ids, [str(x) for x in class_names], fontsize=11)
            ax.set_xlabel("Predicted class" if row == 1 else "", fontsize=12)
            if col == 0:
                ax.set_ylabel("True class", fontsize=12)
            ax.tick_params(length=0)
    fig.suptitle(
        "Confusion matrices | Train + test", fontsize=27, weight="bold", y=0.98
    )
    if im is not None:
        cax = fig.add_axes([0.935, 0.26, 0.012, 0.48])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Per-class recall (%)", fontsize=13)
    fig.text(
        0.5,
        0.065,
        "Each sample counted once per model · Predictions from the mean probabilities across available seeds.",
        ha="center",
        fontsize=13,
        color="#53606B",
    )
    fig.text(
        0.5,
        0.039,
        "Cells: row-normalized percentage (number of samples) · Train + test, not held-out performance.",
        ha="center",
        fontsize=12.5,
        color="#53606B",
    )
    fig.subplots_adjust(
        left=0.07, right=0.91, top=0.87, bottom=0.15, wspace=0.34, hspace=0.38
    )
    save(fig, out, "06_train_test_confusion_matrices")
    pd.DataFrame(csv_rows).to_csv(
        out / "06_train_test_confusion_counts.csv", index=False
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Registry names to include, in the requested order",
    )
    p.add_argument("--skip-physics", action="store_true")
    p.add_argument(
        "--skip-all-data",
        action="store_true",
        help="Skip train+test performance and confusion matrices",
    )
    p.add_argument("--all-data-batch-size", type=int, default=64)
    p.add_argument("--physics-repeats", type=int, default=5)
    p.add_argument("--physics-seed", type=int, default=None)
    p.add_argument(
        "--example-index",
        default="showcase",
        help="Test index, showcase (different illustrative example), or auto (difficult case)",
    )
    p.add_argument(
        "--example-temperature-index",
        type=int,
        default=0,
        help="Temperature channel for the example loop and saliency",
    )
    p.add_argument("--example-h-channel", type=int, default=0)
    p.add_argument("--example-m-channel", type=int, default=1)
    p.add_argument(
        "--highlight-threshold",
        type=float,
        default=0.85,
        help="Only highlight sensitivity above this fraction of its peak (default: 0.85).",
    )
    p.add_argument(
        "--ig-segments",
        "--timeshap-segments",
        type=int,
        default=10,
        help="Chronological segments per RNN temperature (default: 10)",
    )
    p.add_argument(
        "--ig-reference-index",
        "--timeshap-reference-index",
        type=int,
        default=0,
        help="Training sample used as Integrated Gradients reference (default: 0)",
    )
    p.add_argument(
        "--ig-steps",
        type=int,
        default=48,
        help="Integrated Gradients integration steps for RNNs (default: 48)",
    )
    p.add_argument(
        "--perturbation-pct",
        type=float,
        default=5.0,
        help="Magnitude of smooth magnetization perturbations as percent of the sample M range (default: 5)",
    )
    args = p.parse_args()
    root = args.project_root.resolve()
    out = root / "results" / "presentation"
    out.mkdir(parents=True, exist_ok=True)
    style()
    runs = load_runs(root, args.models)
    print("Comparing:", ", ".join(runs))
    print(
        "Matched physics effects on test Macro F1 (percentage points):",
        performance_physics_impact(performance_data(runs, root)),
    )
    performance(runs, out, root)
    confidence_plot(runs, out, root)
    available_ckpts = explanation_models(root, args.models)
    if not args.skip_all_data:
        if not (root / "dataset" / "ml_manifest.csv").is_file():
            print("[all-data] Dataset manifest absent; cannot evaluate train+test.")
        else:
            sys.path.insert(0, str(root))
            all_data, all_predictions, class_names = evaluate_train_test_checkpoints(
                root, available_ckpts, out, max(1, args.all_data_batch_size)
            )
            if all_data:
                performance(
                    runs,
                    out,
                    root,
                    data_override=all_data,
                    stem="05_train_test_objective_performance",
                    title="Objective model performance | Train + test",
                    scope_note="Train + test (not held-out). Background: Macro F1 vs family baseline. Bars: seed mean ±1 SD.",
                )
                plot_train_test_confusion(all_predictions, class_names, out)
    if not args.skip_physics:
        if not (root / "dataset" / "ml_manifest.csv").is_file():
            print("Physics analysis skipped: dataset/ml_manifest.csv missing")
        else:
            sys.path.insert(0, str(root))
            try:
                physics_importance(
                    available_ckpts,
                    root,
                    out,
                    max(1, args.physics_repeats),
                    args.physics_seed,
                )
            except Exception as exc:
                print(
                    f"Global physics-importance analysis could not be completed: {exc}"
                )
            try:
                example_index = args.example_index
                example_seed = args.physics_seed
                if str(example_index).lower() in ("auto", "showcase"):
                    selected = find_difficult_test_example(
                        runs,
                        available_ckpts,
                        out,
                        example_seed,
                        mode=str(example_index).lower(),
                    )
                    if selected is None:
                        print(
                            "[example] Automatic selection unavailable; falling back to index 0."
                        )
                        example_index = 0
                    else:
                        example_index, selected_seed = selected
                        # Use the selected seed only to identify the example.
                        # The RNN Integrated Gradients explanation uses every seed
                        # unless the user explicitly passed --physics-seed.
                        example_seed = args.physics_seed
                else:
                    example_index = int(example_index)
                example_loop_explanations(
                    available_ckpts,
                    root,
                    out,
                    example_index,
                    args.example_temperature_index,
                    example_seed,
                    args.example_h_channel,
                    args.example_m_channel,
                    args.perturbation_pct,
                    args.highlight_threshold,
                    args.ig_segments,
                    args.ig_reference_index,
                    args.ig_steps,
                )
            except Exception as exc:
                print(f"Example-loop explanation could not be completed: {exc}")
            try:
                if "example_index" in locals() and isinstance(example_index, int):
                    preisach_models_explanation(
                        available_ckpts,
                        root,
                        out,
                        example_index,
                        args.perturbation_pct,
                        args.example_h_channel,
                        args.example_m_channel,
                        args.physics_seed,
                    )
            except Exception as exc:
                print(
                    f"Preisach feature explanation could not be completed: {type(exc).__name__}: {exc}"
                )


if __name__ == "__main__":
    main()
