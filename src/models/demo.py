"""Presentation-friendly Streamlit UI for the magnetic-loop classifiers.

From the PROJECT ROOT (where src/ and checkpoints/ live):
    streamlit run src/models/demo.py

This app runs real ensembles of all saved training seeds on test or uploaded measurements.
Select the project root containing src/, dataset/, and checkpoints/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]  # project root; demo lives in src/models/
# Streamlit can be launched from outside the project root. Keep project imports
# resolvable while preserving the user-selected project directory in the UI.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = {
    "mlp": "Neural network",
    "m_data_check_MLP": "Neural network + Physics",
    "preisach_mlp": "Neural network + Preisach",
    "m_data_check_preisach_MLP": "Neural network + Physics + Preisach",
    "rnn_joint": "RNN",
    "m_data_check_RNN": "RNN + Physics",
    "preisach_rnn": "RNN + Preisach",
    "m_data_check_preisach_RNN": "RNN + Physics + Preisach",
}
TEMPERATURE_COLORS = ["#7154A4", "#138A8A", "#E4AB22", "#DA7A52", "#5484BD"]
ACCENT = "#E97935"

st.set_page_config(
    page_title="Magnetic loop explorer",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
<style>
.block-container {max-width: 1450px; padding-top: 1.5rem; padding-bottom: 1.6rem;}
header[data-testid="stHeader"] {background: transparent;}
h1 {font-size: 3.0rem !important; letter-spacing: -.04em; margin: 0 !important;}
h2 {font-size: 2rem !important; letter-spacing: -.025em;}
[data-testid="stMarkdownContainer"] p {font-size: 1.15rem;}
[data-testid="stSelectbox"] label p, [data-testid="stNumberInput"] label p {font-size: 1.12rem !important; font-weight: 650;}
[data-testid="stSelectbox"] [data-baseweb="select"] {font-size: 1.12rem;}
[data-testid="stMetricValue"] {font-size: 3.3rem;}
div[data-testid="stAlert"] p {font-size: 1rem;}
.hero-subtitle {font-size: 1.25rem; color:#607187; margin-top: -.35rem; margin-bottom: 1.15rem;}
.pred-title {font-size: 1.22rem; font-weight:700; color:#526478; margin-bottom:.1rem;}
.pred-value {font-size:3.8rem; font-weight:800; color:#163C55; line-height:1.15;}
.conf-value {font-size:3.5rem; font-weight:800; color:#163C55; line-height:1.17;}
.pred-info {font-size:1.07rem; color:#637487; margin-top:.2rem;}
.model-caption {font-size:1.07rem; color:#637487;}
.note {font-size:1rem; color:#5E7182;}
.orange {color:#DA6A2C; font-weight:750;}
hr {margin:.8rem 0 !important;}
</style>
""",
    unsafe_allow_html=True,
)


def model_list(project_root: Path) -> dict[str, dict[int, Path]]:
    result = {}
    for name in MODELS:
        seeds = {}
        for p in sorted((project_root / "checkpoints" / name).glob("seed_*.pt")):
            try:
                seeds[int(p.stem.split("seed_")[1])] = p
            except (ValueError, IndexError):
                continue
        if seeds:
            result[name] = seeds
    return result


@st.cache_resource(show_spinner=False)
def load_model(
    project_root_str: str, model_name: str, ck_str: str, modified_time: float
):
    import torch

    project_root = Path(project_root_str)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.models.registry import get_spec

    spec = get_spec(model_name)
    checkpoint = torch.load(ck_str, map_location="cpu", weights_only=False)
    model = spec.build(checkpoint["hparams"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return spec, model


def read_sample(project_root: Path, model_name: str, idx: int):
    import torch
    from torch.utils.data import DataLoader, Subset

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset_ml.datasets_ml import CurveSequenceDataset, VALID_LABELS
    from src.models.registry import get_spec

    manifest = project_root / "dataset" / "ml_manifest.csv"
    raw_spec = get_spec("rnn_joint")
    raw_dataset = CurveSequenceDataset(manifest, split="test")
    if not 0 <= idx < len(raw_dataset):
        raise ValueError(f"Test example must be between 0 and {len(raw_dataset) - 1}.")
    raw_batch = next(
        iter(
            DataLoader(
                Subset(raw_dataset, [idx]), batch_size=1, collate_fn=raw_spec.collate_fn
            )
        )
    )
    raw = raw_batch[0][0].detach().cpu().numpy()  # temperatures, time, H/M
    lengths = raw_batch[1][0].detach().cpu().numpy().astype(int)
    y_idx = int(raw_batch[-1].reshape(-1)[0].item())
    spec = get_spec(model_name)
    model_dataset = spec.dataset_cls(manifest, split="test")
    batch = next(
        iter(
            DataLoader(
                Subset(model_dataset, [idx]), batch_size=1, collate_fn=spec.collate_fn
            )
        )
    )
    if len(model_dataset) != len(raw_dataset):
        raise RuntimeError(
            "Dataset lengths differ; sample alignment cannot be guaranteed."
        )
    if int(batch[-1].reshape(-1)[0].item()) != y_idx:
        raise RuntimeError(
            "Dataset labels are not aligned; refusing to compare samples."
        )
    return batch, raw, lengths, y_idx, [str(x) for x in VALID_LABELS]


def test_count(project_root: Path) -> int:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset_ml.datasets_ml import CurveSequenceDataset

    return len(
        CurveSequenceDataset(project_root / "dataset" / "ml_manifest.csv", split="test")
    )


def softmax_logits(spec, model, batch):
    import torch

    device = torch.device("cpu")
    with torch.inference_mode():
        logits, labels = spec.forward(model, batch, device)
    return logits, torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()


def margin(logits, target):
    import torch

    competitors = logits[0].clone()
    competitors[target] = -torch.inf
    return float((logits[0, target] - torch.logsumexp(competitors, dim=0)).item())


def perturb_raw_input(x0, name, raw, lengths, temp, center, sign, magnitude):
    """Localized smooth M perturbation; raster deformation for baseline MLPs is approximate."""
    import torch
    import torch.nn.functional as F

    x = x0.clone()
    L = int(lengths[temp])
    if L < 3:
        return x
    positions = np.arange(L, dtype=float) / max(L - 1, 1)
    weights = np.exp(-0.5 * ((positions - center) / 0.060) ** 2)
    delta = float(sign) * magnitude
    if name in ("rnn_joint", "m_data_check_RNN"):
        n = min(L, int(x.shape[2]))
        x[0, temp, :n, 1] += torch.as_tensor(delta * weights[:n], dtype=x.dtype)
        return x

    image = x[0, temp]
    H, W = image.shape
    valid = [t for t, length in enumerate(lengths) if length > 0]
    all_h = np.concatenate([raw[t, : lengths[t], 0] for t in valid])
    all_m = np.concatenate([raw[t, : lengths[t], 1] for t in valid])
    h_span = max(float(np.ptp(all_h)), 1e-12)
    m_span = max(float(np.ptp(all_m)), 1e-12)
    sel = np.where(weights > 0.08)[0][:: max(1, L // 40)]
    if not len(sel):
        return x
    cx = (raw[temp, sel, 0] - float(all_h.min())) / h_span * (W - 1)
    cy = (float(all_m.max()) - raw[temp, sel, 1]) / m_span * (H - 1)
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=image.dtype),
        torch.arange(W, dtype=image.dtype),
        indexing="ij",
    )
    dist2 = (yy[..., None] - torch.as_tensor(cy, dtype=image.dtype)) ** 2 + (
        xx[..., None] - torch.as_tensor(cx, dtype=image.dtype)
    ) ** 2
    kernel = torch.exp(-0.5 * dist2 / max(2.0, min(H, W) * 0.055) ** 2)
    coverage = (
        (kernel * torch.as_tensor(weights[sel], dtype=image.dtype)).max(-1).values
    )
    gy = 2 * (yy + (delta / m_span) * (H - 1) * coverage) / max(H - 1, 1) - 1
    gx = 2 * xx / max(W - 1, 1) - 1
    grid = torch.stack([gx, gy], dim=-1)[None]
    x[0, temp] = F.grid_sample(
        image[None, None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0]
    return x


def ensemble_predict(spec_models, batch):
    """Mean predictive probability and epistemic mutual information across seeds."""
    predictions = []
    for spec, model in spec_models:
        _, p = softmax_logits(spec, model, batch)
        predictions.append(p)
    seed_probs = np.stack(predictions, axis=0)
    ensemble_probs = seed_probs.mean(axis=0)
    eps = 1e-12
    entropy = lambda p: -np.sum(p * np.log(np.clip(p, eps, 1)), axis=-1)
    predictive_entropy = float(entropy(ensemble_probs))
    expected_entropy = float(np.mean(entropy(seed_probs)))
    epistemic_mi = float(max(0.0, predictive_entropy - expected_entropy))
    pred = int(np.argmax(ensemble_probs))
    votes = np.argmax(seed_probs, axis=1)
    disagreement = float(np.mean(votes != pred))
    return ensemble_probs, epistemic_mi, predictive_entropy, disagreement, seed_probs


def calculate_explanation(spec_models, batch, raw, lengths, name, pct=5.0):
    """Ensemble sensitivity: change in ensemble probability of its chosen class.

    Every perturbed input is evaluated by every ensemble member, so the orange
    highlights describe the ensemble rather than an arbitrary single seed.
    """
    ensemble_probs, _, _, _, _ = ensemble_predict(spec_models, batch)
    prediction = int(np.argmax(ensemble_probs))
    base_conf = float(ensemble_probs[prediction])
    T = len(lengths)
    x0 = batch[0].detach().clone()
    preisach = "preisach" in name.lower()
    n_segments = 10  # smooth perturbations; keep interactive latency reasonable
    centers = (np.arange(n_segments) + 0.5) / n_segments
    result = np.zeros((T, 1 if preisach else n_segments), dtype=float)
    spans = [float(np.ptp(raw[t, : lengths[t], 1])) for t in range(T) if lengths[t] > 1]
    magnitude = (pct / 100.0) * max(max(spans, default=0.0), 1e-12)

    for t in range(T):
        if lengths[t] < 3:
            continue
        for k in range(result.shape[1]):
            changes = []
            for sign in (-1, +1):
                if preisach:
                    # Temperature-level sensitivity in the *actual derived feature*
                    # space, not a claim about particular physical curve points.
                    new_input = x0.clone()
                    new_input[:, t] = x0[:, t] * (1 + sign * pct / 100.0)
                else:
                    new_input = perturb_raw_input(
                        x0, name, raw, lengths, t, float(centers[k]), sign, magnitude
                    )
                altered_batch = (new_input, *batch[1:])
                altered_probs, _, _, _, _ = ensemble_predict(spec_models, altered_batch)
                changes.append(abs(float(altered_probs[prediction]) - base_conf))
            result[t, k] = float(np.mean(changes))

    temp_impact = result.mean(axis=1)
    focus_temp = int(np.argmax(temp_impact))
    if preisach:
        return focus_temp, None, temp_impact
    L = int(lengths[focus_temp])
    progress = np.linspace(0, 1, L)
    values = np.interp(
        progress,
        centers,
        result[focus_temp],
        left=result[focus_temp, 0],
        right=result[focus_temp, -1],
    )
    return focus_temp, values, temp_impact


@st.cache_data(show_spinner=False, ttl=3600)
def cached_rnn_integrated_gradients(
    project_root_str: str,
    model_name: str,
    test_index: int,
    checkpoint_signatures: tuple,
    reference_index: int = 0,
    n_segments: int = 10,
    n_steps: int = 48,
):
    """Cached ensemble Integrated Gradients: shared implementation from presentation_plots.py.

    Cache is invalidated when the checkpoint paths or their modification times
    change, or when model, test sample, reference sample or settings change.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    project_root = Path(project_root_str)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.models.registry import get_spec

    try:
        from src.models.presentation_plots import (
            integrated_gradients_rnn_ensemble_attribution,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Make sure src/models/presentation_plots.py exists. "
            "Both files are provided in src/models/."
        ) from exc
    spec = get_spec(model_name)
    manifest = project_root / "dataset" / "ml_manifest.csv"
    test_ds = spec.dataset_cls(manifest, split="test")
    train_ds = spec.dataset_cls(manifest, split="train")
    if not 0 <= reference_index < len(train_ds):
        raise ValueError("Invalid IG reference training sample index")
    test_batch = next(
        iter(
            DataLoader(
                Subset(test_ds, [test_index]), batch_size=1, collate_fn=spec.collate_fn
            )
        )
    )
    ref_batch = next(
        iter(
            DataLoader(
                Subset(train_ds, [reference_index]),
                batch_size=1,
                collate_fn=spec.collate_fn,
            )
        )
    )
    models = [
        load_model(project_root_str, model_name, path, mtime)
        for path, mtime in checkpoint_signatures
    ]
    result = integrated_gradients_rnn_ensemble_attribution(
        models, test_batch, ref_batch, n_segments=n_segments, n_steps=n_steps
    )
    focus = int(result["most_relevant_temperature"])
    temp_info = result["temperatures"]
    vals = temp_info[focus]["integrated_gradients"]
    length = int(test_batch[1][0, focus].item())
    positions = np.linspace(0, 1, length)
    centers = (np.arange(len(vals)) + 0.5) / len(vals)
    points = np.interp(positions, centers, vals, left=vals[0], right=vals[-1])
    importance = np.asarray(
        [
            np.sum(np.abs(temp_info[t]["integrated_gradients"]))
            for t in sorted(temp_info)
        ],
        dtype=float,
    )
    return focus, points, importance, np.asarray(vals, dtype=float)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_uploaded_ig(
    project_root_str: str,
    model_name: str,
    upload_payload: tuple,
    actual_temperatures: tuple,
    moment_mode: str,
    manual_scale_emu: float,
    field_range_mode: str,
    checkpoint_signatures: tuple,
    n_segments: int = 10,
    n_steps: int = 48,
):
    """Integrated Gradients on real, uploaded loops. Cache includes all upload settings."""
    from torch.utils.data import DataLoader, Subset
    from src import config
    from src.models.registry import get_spec
    from src.models.presentation_plots import (
        integrated_gradients_rnn_ensemble_attribution,
    )
    from uploaded_measurements import load_measured_files, batch_for_uploaded
    from types import SimpleNamespace

    root = Path(project_root_str)
    uploaded = [
        SimpleNamespace(name=name, getvalue=lambda blob=blob: blob)
        for name, blob in upload_payload
    ]
    expected = tuple(map(float, config.TEMPERATURES_K))
    samples, _, _, _ = load_measured_files(
        uploaded,
        expected,
        temperature_override=actual_temperatures,
        allow_temperature_mismatch=True,
        hmax_oe=float(config.Hmax_Oe),
        moment_normalization=moment_mode,
        moment_scale_emu=manual_scale_emu,
        field_range_mode=field_range_mode,
    )
    template, _, _, _, _ = read_sample(root, model_name, 0)
    batch, _raw, lengths = batch_for_uploaded(
        root, model_name, samples, template, actual_temperatures, expected
    )
    spec = get_spec(model_name)
    train_ds = spec.dataset_cls(root / "dataset" / "ml_manifest.csv", split="train")
    ref = next(
        iter(
            DataLoader(Subset(train_ds, [0]), batch_size=1, collate_fn=spec.collate_fn)
        )
    )
    models = [
        load_model(project_root_str, model_name, path, mtime)
        for path, mtime in checkpoint_signatures
    ]
    result = integrated_gradients_rnn_ensemble_attribution(
        models, batch, ref, n_segments=n_segments, n_steps=n_steps
    )
    focus = int(result["most_relevant_temperature"])
    vals = np.asarray(
        result["temperatures"][focus]["integrated_gradients"], dtype=float
    )
    centers = (np.arange(len(vals)) + 0.5) / len(vals)
    points = np.interp(
        np.linspace(0, 1, int(lengths[focus])),
        centers,
        vals,
        left=vals[0],
        right=vals[-1],
    )
    importance = np.array(
        [
            sum(abs(result["temperatures"][t]["integrated_gradients"]))
            for t in sorted(result["temperatures"])
        ]
    )
    return focus, points, importance, vals


@st.cache_data(show_spinner=False, ttl=3600)
def cached_uploaded_preisach(
    project_root_str: str,
    model_name: str,
    upload_payload: tuple,
    actual_temperatures: tuple,
    moment_mode: str,
    manual_scale_emu: float,
    field_range_mode: str,
    checkpoint_signatures: tuple,
):
    from src import config
    from uploaded_measurements import load_measured_files, batch_for_uploaded
    from src.models.presentation_plots import preisach_feature_ensemble_attribution
    from types import SimpleNamespace

    root = Path(project_root_str)
    uploaded = [
        SimpleNamespace(name=name, getvalue=lambda blob=blob: blob)
        for name, blob in upload_payload
    ]
    expected = tuple(map(float, config.TEMPERATURES_K))
    samples, _, _, _ = load_measured_files(
        uploaded,
        expected,
        temperature_override=actual_temperatures,
        allow_temperature_mismatch=True,
        hmax_oe=float(config.Hmax_Oe),
        moment_normalization=moment_mode,
        moment_scale_emu=manual_scale_emu,
        field_range_mode=field_range_mode,
    )
    template, _, _, _, _ = read_sample(root, model_name, 0)
    batch, _, _ = batch_for_uploaded(
        root, model_name, samples, template, actual_temperatures, expected
    )
    members = [
        load_model(project_root_str, model_name, path, mtime)
        for path, mtime in checkpoint_signatures
    ]
    return preisach_feature_ensemble_attribution(members, batch, 5.0)


def plot_loops(raw, lengths, temperatures, emphasis=None):
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    fig.patch.set_facecolor("white")
    for t in range(len(lengths)):
        L = int(lengths[t])
        if not L:
            continue
        ax.plot(
            raw[t, :L, 0],
            raw[t, :L, 1],
            color=TEMPERATURE_COLORS[t % len(TEMPERATURE_COLORS)],
            lw=3.5 if emphasis == t else 2.9,
            label=f"{temperatures[t]:g} K",
            solid_capstyle="round",
        )
    ax.set(xlabel="Magnetic field", ylabel="Magnetization")
    ax.grid(alpha=0.12)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.legend(frameon=False, ncol=min(5, len(lengths)), loc="upper left", fontsize=13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=1.6)
    return fig


@st.cache_data(show_spinner=False, ttl=3600)
def cached_preisach_attribution(
    project_root_str: str,
    model_name: str,
    test_index: int,
    checkpoint_signatures: tuple,
    perturbation_pct: float = 5.0,
):
    """Ensemble feature-space perturbation sensitivity, cached for screenshots."""
    from src.models.presentation_plots import preisach_feature_ensemble_attribution

    project_root = Path(project_root_str)
    members = []
    for ck_path, timestamp in checkpoint_signatures:
        members.append(load_model(project_root_str, model_name, ck_path, timestamp))
    batch, _raw, _lengths, _truth, _labels = read_sample(
        project_root, model_name, test_index
    )
    return preisach_feature_ensemble_attribution(members, batch, perturbation_pct)


def plot_preisach_attribution(attribution, temperatures):
    """True Preisach (alpha,beta) plane for the dominant temperature."""
    from src.models.presentation_plots import plot_preisach_feature_map

    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor="white")
    im = plot_preisach_feature_map(ax, attribution, temperatures)
    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025, shrink=0.85)
    bar.set_label("Mean |change in predicted-class probability|", fontsize=10)
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.19, top=0.86)
    return fig


def plot_preisach_physical_explanation(raw, lengths, attribution, temperatures):
    """Highlight actual hysteron switch-field neighborhoods on measured loops."""
    from src.models.presentation_plots import plot_preisach_curve_importance

    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="white")
    clear = plot_preisach_curve_importance(ax, raw, lengths, attribution)
    ax.set(xlabel="Magnetic field", ylabel="Magnetization")
    ax.tick_params(labelsize=11)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=1.2)
    return fig, clear


def plot_explanation(
    raw,
    lengths,
    temp,
    values,
    impact,
    temperatures,
    threshold=0.85,
    signed=False,
    event_values=None,
):
    from matplotlib.collections import LineCollection

    if signed and values is not None:
        # No hard cutoff for IG: show the complete attribution profile.
        fig = plt.figure(figsize=(12.8, 5.0), facecolor="white")
        gs = fig.add_gridspec(
            2,
            1,
            height_ratios=[3.7, 1.0],
            hspace=0.48,
            left=0.095,
            right=0.985,
            top=0.95,
            bottom=0.17,
        )
        ax = fig.add_subplot(gs[0])
        ax_events = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(12.8, 3.8))
        ax_events = None
    fig.patch.set_facecolor("white")
    for t in range(len(lengths)):
        L = int(lengths[t])
        ax.plot(raw[t, :L, 0], raw[t, :L, 1], color="#D4DCE3", linewidth=2.4)
    L = int(lengths[temp])
    h, m = raw[temp, :L, 0], raw[temp, :L, 1]
    ax.plot(h, m, color="#8399AD", lw=5.5, solid_capstyle="round", zorder=3)
    strong = False
    if values is None:
        # Preisach group-level sensitivity: highlight the entire relevant loop
        # only if its temperature influence is unambiguously strongest.
        peak = float(np.max(impact))
        rest = float(np.median(impact))
        if peak > 1e-9 and peak >= max(1.8 * rest, 1e-9):
            ax.plot(h, m, lw=7.3, color=ACCENT, solid_capstyle="round", zorder=4)
            strong = True
    else:
        vals = np.asarray(values, dtype=float)
        magnitude = np.abs(vals) if signed else vals
        peak = float(np.nanmax(magnitude)) if len(magnitude) else 0
        median = float(np.nanmedian(magnitude)) if len(magnitude) else 0
        if signed and len(h) >= 2 and peak > 1e-12:
            from matplotlib.colors import to_rgba

            p = np.column_stack([h, m]).reshape(-1, 1, 2)
            edges = np.concatenate([p[:-1], p[1:]], axis=1)
            edge_scores = (vals[:-1] + vals[1:]) / 2
            magnitude_edges = np.abs(edge_scores) / peak
            # Draw all segments, with intensity proportional to their IG
            # magnitude. A single orange tail no longer hides other effects.
            rgba = []
            widths = []
            for val, relative in zip(edge_scores, magnitude_edges):
                base = "#E97935" if val >= 0 else "#2985B8"
                r, g, b, _ = to_rgba(base)
                rgba.append((r, g, b, float(0.16 + 0.84 * np.clip(relative, 0, 1))))
                widths.append(float(3.0 + 4.5 * np.clip(relative, 0, 1)))
            lc = LineCollection(edges, colors=rgba, linewidths=widths, zorder=5)
            lc.set_capstyle("round")
            ax.add_collection(lc)
            strong = True
        elif (
            not signed
            and len(h) >= 2
            and peak > 1e-9
            and peak >= max(1.8 * median, 1e-9)
        ):
            p = np.column_stack([h, m]).reshape(-1, 1, 2)
            edges = np.concatenate([p[:-1], p[1:]], axis=1)
            edge_scores = (vals[:-1] + vals[1:]) / 2
            selection = np.isfinite(edge_scores) & (edge_scores >= threshold * peak)
            if selection.any():
                lc = LineCollection(
                    edges[selection], colors=ACCENT, linewidths=8.3, zorder=5
                )
                lc.set_capstyle("round")
                ax.add_collection(lc)
                strong = True

        if ax_events is not None:
            # Chronological index, not H: two points of a hysteresis loop
            # can share H but occur at different acquisition times.
            event_vals = (
                np.asarray(event_values, dtype=float)
                if event_values is not None
                else np.interp(
                    (np.arange(10) + 0.5) / 10, np.linspace(0, 1, len(vals)), vals
                )
            )
            n = len(event_vals)
            ax_events.bar(
                np.arange(1, n + 1),
                event_vals,
                color=np.where(event_vals >= 0, "#E97935", "#2985B8"),
                width=0.76,
            )
            ax_events.axhline(0, color="#73808C", linewidth=1)
            ax_events.set_xlim(0.4, n + 0.6)
            ax_events.set_xticks([1, 3, 5, 7, 10])
            ax_events.set_xlabel(
                "Segment along measurement sequence (start → end)", fontsize=11
            )
            ax_events.set_ylabel("IG", fontsize=11)
            ax_events.tick_params(labelsize=10)
            ax_events.grid(axis="y", alpha=0.12)
            ax_events.spines["top"].set_visible(False)
            ax_events.spines["right"].set_visible(False)
    ax.set(xlabel="Magnetic field", ylabel="Magnetization")
    ax.grid(alpha=0.1)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if ax_events is None:
        fig.tight_layout(pad=1.4)
    return fig, strong


st.title("Magnetic loop explorer")
st.markdown(
    '<div class="hero-subtitle">From measured magnetic loops to an AI prediction — in one glance.</div>',
    unsafe_allow_html=True,
)


def valid_project_root(path: Path) -> bool:
    return (
        (path / "src" / "models" / "registry.py").is_file()
        and (path / "dataset" / "ml_manifest.csv").is_file()
        and (path / "checkpoints").is_dir()
    )


def discover_project_root() -> Path:
    """Find the project without silently switching to synthetic predictions."""
    import os

    candidates = []
    if os.environ.get("MAGNETIC_PROJECT_ROOT"):
        candidates.append(Path(os.environ["MAGNETIC_PROJECT_ROOT"]).expanduser())
    for base in (ROOT, Path.cwd()):
        candidates.extend([base, base.parent, base.parent.parent])
    for candidate in candidates:
        if valid_project_root(candidate):
            return candidate.resolve()
    return ROOT


if "project_root_input" not in st.session_state:
    st.session_state.project_root_input = str(discover_project_root())

project_root = Path(st.session_state.project_root_input).expanduser().resolve()
if not valid_project_root(project_root):
    st.error(
        "**Project not found.** This app only uses your real trained models; it will not display synthetic predictions."
    )
    st.markdown(
        "Choose the directory containing `src/`, `dataset/`, and `checkpoints/`."
    )
    if not __import__("os").environ.get("MAGNETIC_DEPLOYED_DEMO"):
        st.text_input(
            "Project folder (absolute path)",
            key="project_root_input",
            placeholder="/path/to/your/project",
        )
    st.code("streamlit run src/models/demo.py", language="bash")
    st.info(
        "The demo is in `src/models/`. Select the project root containing `src/`, `dataset/` and `checkpoints/`."
    )
    st.stop()

available = model_list(project_root)
if not available:
    st.error(
        f"No model checkpoints found under `{project_root / 'checkpoints'}`. Expected filenames like `checkpoints/mlp/seed_0.pt`."
    )
    if not __import__("os").environ.get("MAGNETIC_DEPLOYED_DEMO"):
        st.text_input("Project folder (absolute path)", key="project_root_input")
    st.stop()

# A small settings expander is hidden during screenshots. No preview switch:
# outputs always come from real saved checkpoints.
with st.expander("Project settings", expanded=False):
    if not __import__("os").environ.get("MAGNETIC_DEPLOYED_DEMO"):
        st.text_input("Project folder (absolute path)", key="project_root_input")
    st.caption(
        "This application uses real trained checkpoints and either test samples or your uploaded measured curves."
    )

import os
show_test_dataset = os.environ.get("MAGNETIC_SHOW_TEST_DATASET", "true").lower() in ("1", "true", "yes")
if show_test_dataset:
    source = st.radio(
        "Input source",
        ["Test dataset", "Upload my measurements"],
        horizontal=True,
        label_visibility="collapsed",
    )
else:
    source = "Upload my measurements"
    st.caption("Upload your own magnetic measurements for classification.")
controls = st.columns([2.5, 1.0], vertical_alignment="bottom", gap="large")
with controls[0]:
    name = st.selectbox(
        "Choose a model", list(available), format_func=lambda key: MODELS[key]
    )
with controls[1]:
    if source == "Test dataset":
        try:
            n_examples = test_count(project_root)
        except Exception as exc:
            st.error(f"Could not load the test dataset: {type(exc).__name__}: {exc}")
            st.stop()
        if not n_examples:
            st.error("The test split contains no examples.")
            st.stop()
        example_index = st.number_input(
            "Test example #", min_value=0, max_value=n_examples - 1, value=0, step=1
        )
    else:
        example_index = None
        st.markdown("**Your own measured loops**")

upload_payload = None
actual_temperatures = None
moment_mode = "measured_max"
manual_scale_emu = 1.0
field_range_mode = "strict"
if source == "Upload my measurements":
    if name in ("mlp", "m_data_check_MLP"):
        st.warning(
            "The two image-based Neural networks need the original rasterizer. "
            "Choose an RNN or a Preisach model for raw .dat files."
        )
        st.stop()
    st.caption(
        "Upload three MPMS3 .dat files, one at each temperature. "
        "The app reads DC Moment Fixed Ctr (emu), its uncertainty, "
        "Magnetic Field (Oe) and Temperature (K). A single combined CSV is also supported."
    )
    uploaded_files = st.file_uploader(
        "Measured loops (.dat or .csv)",
        type=["dat", "csv"],
        accept_multiple_files=True,
        help="e.g. 5 K, 10 K, and 300 K in three separate .dat files",
    )
    st.download_button(
        "Download CSV template",
        data=b"temperature_K,magnetic_field,magnetization,magnetization_std\n",
        file_name="magnetic_measurements_template.csv",
        mime="text/csv",
    )
    actual_temperatures = (5.0, 10.0, 300.0)
    st.warning(
        "**Experimental / out-of-distribution inference.** The trained model expects "
        "the temperatures from its training configuration, but you want 5, 10 and 300 K. "
        "The 10 K loop will enter the model's middle temperature slot (trained on 250 K). "
        "Confidence and uncertainty may not be reliable for this sample."
    )
    confirm_ood = st.checkbox(
        "I understand this is exploratory: temperatures differ from training",
        value=False,
    )
    moment_mode_label = st.selectbox(
        "Convert measured moment (emu) to training magnetization scale",
        [
            "Divide by maximum absolute moment in uploaded files (exploratory)",
            "Use a known emu-per-unit scale (recommended if calibrated)",
        ],
        index=0,
    )
    moment_mode = (
        "manual" if moment_mode_label.startswith("Use a known") else "measured_max"
    )
    if moment_mode == "manual":
        manual_scale_emu = st.number_input(
            "Moment in emu corresponding to one normalized magnetization unit",
            min_value=1e-12,
            value=0.004,
            format="%.8g",
        )
    st.caption(
        "Raw moments (emu) are not the training magnetization (mx_mean). "
        "Automatic rescaling is a heuristic and does not make these experimental loops equivalent to simulations."
    )
    field_mode_label = st.radio(
        "Experimental magnetic-field range",
        [
            "Keep the real field scale (recommended)",
            "Map my measured field range onto the training range (exploratory)",
        ],
        index=0,
        help="The model was trained around ±5,000 Oe, while the example MPMS3 .dat reaches ±10,000 Oe. "
        "Remapping compresses the entire field axis. It is NOT a unit conversion or a physical calibration.",
    )
    field_range_mode = (
        "map_to_training" if field_mode_label.startswith("Map") else "strict"
    )
    if field_range_mode == "map_to_training":
        st.error(
            "**Experimental field remapping enabled.** The ±10,000 Oe experimental sweep "
            "will be scaled to the model's training input range. Switching fields and "
            "coercivity will NOT retain their original physical scale. This is for demonstration only."
        )
    if not uploaded_files:
        st.info("Upload your 5 K, 10 K and 300 K .dat files to begin.")
        st.stop()
    if not confirm_ood:
        st.info("Confirm the out-of-distribution warning to run your trained ensemble.")
        st.stop()
    upload_payload = tuple((f.name, f.getvalue()) for f in uploaded_files)

seeds = sorted(available[name])
if len(seeds) < 2:
    st.warning(
        f"Only {len(seeds)} checkpoint found for this model. Ensemble uncertainty needs at least two independently trained seeds."
    )
try:
    with st.spinner(
        f"Evaluating ensemble of {len(seeds)} trained models and explaining its prediction…"
    ):
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src import config

        temperatures = [float(t) for t in config.TEMPERATURES_K]
        spec_models = []
        checkpoint_signatures = tuple(
            (str(available[name][seed]), available[name][seed].stat().st_mtime)
            for seed in seeds
        )
        for path, mtime in checkpoint_signatures:
            spec_models.append(load_model(str(project_root), name, path, mtime))
        if upload_payload is None:
            batch, raw, lengths, actual, class_names = read_sample(
                project_root, name, int(example_index)
            )
            display_temperatures = temperatures
        else:
            from uploaded_measurements import load_measured_files, batch_for_uploaded
            from types import SimpleNamespace

            physical_files = [
                SimpleNamespace(name=filename, getvalue=lambda payload=blob: payload)
                for filename, blob in upload_payload
            ]
            samples, display_temperatures, upload_warnings, used_scale = (
                load_measured_files(
                    physical_files,
                    temperatures,
                    temperature_override=actual_temperatures,
                    hmax_oe=float(config.Hmax_Oe),
                    moment_normalization=moment_mode,
                    moment_scale_emu=manual_scale_emu,
                    allow_temperature_mismatch=True,
                    field_range_mode=field_range_mode,
                )
            )
            template, _old_raw, _old_lengths, _old_actual, class_names = read_sample(
                project_root, name, 0
            )
            batch, raw, lengths = batch_for_uploaded(
                project_root,
                name,
                samples,
                template,
                actual_temperatures=display_temperatures,
                trained_temperatures=temperatures,
            )
            actual = None
        probs, epistemic_mi, predictive_entropy, disagreement, seed_probs = (
            ensemble_predict(spec_models, batch)
        )
        use_ig = name in ("rnn_joint", "m_data_check_RNN")
        preisach_attribution = None
        if "preisach" in name.lower():
            if upload_payload is None:
                preisach_attribution = cached_preisach_attribution(
                    str(project_root),
                    name,
                    int(example_index),
                    checkpoint_signatures,
                    perturbation_pct=5.0,
                )
            else:
                preisach_attribution = cached_uploaded_preisach(
                    str(project_root),
                    name,
                    upload_payload,
                    tuple(actual_temperatures),
                    moment_mode,
                    float(manual_scale_emu),
                    field_range_mode,
                    checkpoint_signatures,
                )
            focus = int(preisach_attribution["most_relevant_temperature"])
            impacts = np.asarray(preisach_attribution["temperatures"], dtype=float)
            values, ig_events = None, None
        elif use_ig:
            if upload_payload is None:
                focus, values, impacts, ig_events = cached_rnn_integrated_gradients(
                    str(project_root),
                    name,
                    int(example_index),
                    checkpoint_signatures,
                    reference_index=0,
                    n_segments=10,
                    n_steps=48,
                )
            else:
                focus, values, impacts, ig_events = cached_uploaded_ig(
                    str(project_root),
                    name,
                    upload_payload,
                    tuple(actual_temperatures),
                    moment_mode,
                    float(manual_scale_emu),
                    field_range_mode,
                    checkpoint_signatures,
                    n_segments=10,
                    n_steps=48,
                )
        else:
            ig_events = None
            focus, values, impacts = calculate_explanation(
                spec_models, batch, raw, lengths, name, pct=5.0
            )
except Exception as exc:
    st.error(f"Could not evaluate the trained ensemble: {type(exc).__name__}: {exc}")
    st.caption(f"Project: {project_root} · Model: {name} · Seeds: {seeds}")
    st.stop()

prediction = int(np.argmax(probs))
pred_label = (
    class_names[prediction] if prediction < len(class_names) else str(prediction)
)
confidence = float(probs[prediction])

st.caption(
    f"Real ensemble inference · {len(seeds)} independently trained seeds · "
    + (
        "held-out test example"
        if actual is not None
        else "uploaded experimental measurements (out of distribution)"
    )
)
if upload_payload is not None:
    for notice in upload_warnings:
        st.warning(notice)
    st.error(
        "**Out-of-distribution experimental input:** 5 / 10 / 300 K (trained at "
        + " / ".join(f"{T:g}" for T in temperatures)
        + " K). "
        "Reported probabilities and epistemic uncertainty are exploratory, not validated on this measurement domain."
    )
    st.caption(
        f"Moment conversion: 1 normalized magnetization unit = {used_scale:.6g} emu. "
        "The 10 K curve was assigned to the model's middle temperature slot."
    )

st.divider()
left, right = st.columns([1.7, 1.0], gap="large", vertical_alignment="top")
with left:
    st.subheader("Input · measured loops")
    fig = plot_loops(raw, lengths, display_temperatures)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
with right:
    st.subheader("AI output")
    with st.container(border=True):
        st.markdown('<div class="pred-title">PREDICTION</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pred-value">{pred_label}</div>', unsafe_allow_html=True
        )
        st.write("")
        st.markdown('<div class="pred-title">CONFIDENCE</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="conf-value">{confidence:.0%}</div>', unsafe_allow_html=True
        )
        st.progress(min(1.0, max(0.0, confidence)))
        if actual is not None:
            truth = class_names[actual] if actual < len(class_names) else str(actual)
            verdict = (
                "Correct prediction" if prediction == actual else "Incorrect prediction"
            )
            st.markdown(
                f'<div class="pred-info">True class: <b>{truth}</b> · {verdict}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-info">Your uploaded measurement · true class not supplied</div>',
                unsafe_allow_html=True,
            )
        st.write("")
        st.markdown(
            '<div class="pred-title">EPISTEMIC UNCERTAINTY</div>',
            unsafe_allow_html=True,
        )
        if len(seeds) > 1:
            st.markdown(
                f'<div class="conf-value">{epistemic_mi:.3f} <span style="font-size:1.25rem;color:#637487">nats</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="pred-info">Disagreement: <b>{disagreement:.0%}</b> of seeds predict another class</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="conf-value">N/A</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="pred-info">At least two trained seeds are needed.</div>',
                unsafe_allow_html=True,
            )
        st.caption(f"{MODELS[name]} · ensemble of {len(seeds)} seeds")

st.divider()
st.subheader("What matters most for this prediction?")
if preisach_attribution is not None:
    left_explain, right_explain = st.columns([1.15, 1.0], gap="large")
    with left_explain:
        st.markdown("**Most sensitive measured loop**")
        expl, clear = plot_preisach_physical_explanation(
            raw, lengths, preisach_attribution, display_temperatures
        )
        st.pyplot(expl, use_container_width=True)
        plt.close(expl)
    with right_explain:
        st.markdown("**Which Preisach coefficients matter?**")
        fmap = plot_preisach_attribution(preisach_attribution, display_temperatures)
        st.pyplot(fmap, use_container_width=True)
        plt.close(fmap)
else:
    expl, clear = plot_explanation(
        raw,
        lengths,
        focus,
        values,
        impacts,
        display_temperatures,
        threshold=0.85 if not use_ig else 0.65,
        signed=use_ig,
        event_values=ig_events,
    )
    st.pyplot(expl, use_container_width=True)
    plt.close(expl)

if "preisach" in name.lower():
    desc = (
        f"Most sensitive temperature: **{display_temperatures[focus]:g} K**. "
        "The triangular map shows Preisach coefficients in the switching-field plane. "
        "Orange curve segments mark nearby measured switching fields (β on the descending "
        "branch, α on the ascending branch), not verified causal input points."
    )
elif use_ig:
    desc = (
        f"Integrated Gradients · Main loop: **{display_temperatures[focus]:g} K**. "
        "Orange supports the prediction relative to the reference; blue opposes it. "
        "Colour intensity shows relative contribution. Reference: a real training loop."
    )
else:
    desc = f"Most sensitive loop: **{display_temperatures[focus]:g} K**. Orange marks the segments whose small changes affect the prediction most."
if not clear:
    desc += (
        " **No clearly dominant region** was detected, so no orange highlight is shown."
    )
st.markdown(desc)
st.caption(
    (
        "RNN explanation: Integrated Gradients on the ensemble log probability "
        "(orange = supports, blue = opposes, relative to a training reference). "
        if use_ig
        else "Other models: smooth 5% perturbation sensitivity (MLP raster deformations are approximate; "
        "Preisach coefficients are perturbed directly; the loop mapping is based on switching fields). "
    )
    + "Epistemic uncertainty is mutual information between training seeds, not prediction error."
)
