"""Import Quantum Design MPMS3 .dat or normalized CSV loops for exploratory ensemble inference.

The trained networks consume normalized magnetization (dimensionless), not raw emu.
Converting raw emu by saturation is a user-selected approximation, NOT a
calibrated match to the simulation training distribution.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

DAT_MOMENT = "DC Moment Fixed Ctr (emu)"
DAT_ERROR = "DC Moment Err Fixed Ctr (emu)"


def _number_column(df, name):
    if name not in df:
        raise ValueError(
            f"Column not found: {name}. Available: {list(df.columns)[:12]}"
        )
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def parse_mpms3_dat(data: bytes, filename: str = "") -> pd.DataFrame:
    """Return tidy measured data. Preserves the MPMS acquisition order."""
    contents = data.decode("utf-8-sig", errors="replace")
    lines = contents.splitlines()
    marker = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == "[data]"), None
    )
    if marker is None:
        raise ValueError(
            f'{filename or "The file"} is not an MPMS3 .dat: missing [Data] section.'
        )
    raw = pd.read_csv(
        io.StringIO("\n".join(lines[marker + 1 :])),
        low_memory=False,
        on_bad_lines="warn",
    )
    raw.columns = [str(x).strip() for x in raw.columns]
    for key in ("Temperature (K)", "Magnetic Field (Oe)", DAT_MOMENT):
        if key not in raw.columns:
            raise ValueError(f"{filename}: MPMS3 column missing: {key}")
    m = _number_column(raw, DAT_MOMENT)
    sd = (
        _number_column(raw, DAT_ERROR)
        if DAT_ERROR in raw.columns
        else np.full(len(raw), np.nan)
    )
    out = pd.DataFrame(
        {
            "temperature_k": _number_column(raw, "Temperature (K)"),
            "field_oe": _number_column(raw, "Magnetic Field (Oe)"),
            "moment_emu": m,
            "moment_error_emu": sd,
        }
    )
    out = out[
        np.isfinite(out["temperature_k"])
        & np.isfinite(out["field_oe"])
        & np.isfinite(out["moment_emu"])
    ].reset_index(drop=True)
    if len(out) < 8:
        raise ValueError(
            f"{filename}: fewer than eight valid field/moment measurements."
        )
    return out


def parse_measurement_csv(
    data: bytes, field_unit: str, moment_unit: str
) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    frame.columns = [re.sub(r"\s+", "_", str(k).strip().lower()) for k in frame.columns]

    def col(names, required=True):
        for name in names:
            if name in frame:
                return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        if required:
            raise ValueError(f"CSV must contain one of these columns: {names}")
        return np.full(len(frame), np.nan)

    temperature = col(["temperature_k", "temp_k", "temperature"])
    field = col(
        ["magnetic_field", "magnetic_field_oe", "field", "bx_mt", "bx_normalized", "h"]
    )
    moment = col(["magnetization", "magnetization_emu", "moment", "mx_mean", "mx"])
    std = col(
        ["magnetization_std", "moment_error_emu", "mx_std", "std"], required=False
    )
    h_oe = (
        field * {"Oe": 1.0, "mT": 10.0, "normalized": None}[field_unit]
        if field_unit != "normalized"
        else field
    )
    # normalized H stays in the first column; conversion deferred until hmax known.
    return pd.DataFrame(
        {
            "temperature_k": temperature,
            "field_oe": h_oe,
            "moment_emu": moment,
            "moment_error_emu": std,
        }
    )


def load_measured_files(
    uploaded_files,
    expected_temperatures,
    *,
    field_unit="Oe",
    temperature_override=None,
    hmax_oe=10000.0,
    moment_normalization="measured_max",
    moment_scale_emu=None,
    allow_temperature_mismatch=False,
    field_range_mode="strict",
):
    """Accept 1-3 MPMS3 files or one CSV and convert to model-shaped curve arrays.

    Returns curves in expected *slot* order, actual temperature list, and warnings.
    In mismatch mode, 10 K can occupy the slot of 250 K but is never renamed 250 K.
    """
    if not uploaded_files:
        raise ValueError(
            "Upload the three curves (e.g. .dat files at 5, 10, and 300 K)."
        )
    expected = [float(t) for t in expected_temperatures]
    parts = []
    for file in uploaded_files:
        name = str(getattr(file, "name", "uploaded.dat"))
        blob = file.getvalue() if hasattr(file, "getvalue") else file[1]
        if name.lower().endswith(".dat"):
            frame = parse_mpms3_dat(blob, name)
        elif name.lower().endswith(".csv"):
            frame = parse_measurement_csv(blob, field_unit, "emu")
        else:
            raise ValueError(f"Unsupported format: {name}")
        if len(frame) > 50000:
            raise ValueError(f"{name}: exceeds 50,000 rows.")
        parts.append(frame)
    data = pd.concat(parts, ignore_index=True)

    # Cluster by nominal temperature (temperature readouts fluctuate around set point).
    actual_targets = [
        float(t)
        for t in (
            temperature_override if temperature_override is not None else expected
        )
    ]
    if len(actual_targets) != len(expected):
        raise ValueError(
            f"Expected {len(expected)} temperatures; got {actual_targets}."
        )
    if len(set(actual_targets)) != len(actual_targets):
        raise ValueError("Temperature values must be distinct.")
    values = data["temperature_k"].to_numpy(dtype=float)
    target_array = np.asarray(actual_targets, dtype=float)
    nearest = np.argmin(abs(values[:, None] - target_array[None, :]), axis=1)
    deviation = abs(values - target_array[nearest])
    tolerance = 2.0  # K; actual temperature can drift modestly from nominal set point
    if (deviation > tolerance).any():
        examples = values[deviation > tolerance][:5]
        raise ValueError(
            f"File contains temperatures not matching {actual_targets} K (e.g. {examples}). Upload only the requested loops."
        )

    warnings = []
    if not np.allclose(actual_targets, expected, rtol=0, atol=0.5):
        if not allow_temperature_mismatch:
            raise ValueError(
                f"The model was trained at {expected} K but received {actual_targets} K. Explicitly enable exploratory out-of-distribution evaluation."
            )
        warnings.append(
            f"OUT-OF-DISTRIBUTION TEMPERATURES: trained at {expected} K; measured at {actual_targets} K. The {actual_targets[1]:g} K curve occupies the {expected[1]:g} K input slot. Confidence and uncertainty are NOT calibrated for these data."
        )
    if moment_normalization == "measured_max":
        moment_scale = float(np.nanmax(abs(data["moment_emu"].to_numpy(dtype=float))))
        warnings.append(
            "Moment normalized by the largest |emu| value across uploaded curves. This is only an exploratory scale and may not match training magnetization."
        )
    elif moment_normalization == "manual":
        moment_scale = float(moment_scale_emu) if moment_scale_emu is not None else 0.0
        warnings.append(
            "Moment divided by the manually supplied emu-per-unit normalization scale; ensure it matches the training magnetization definition."
        )
    else:
        raise ValueError("Unknown moment normalization mode.")
    if not np.isfinite(moment_scale) or moment_scale <= 0:
        raise ValueError(
            "The normalization scale must be a positive finite number of emu."
        )

    if not np.isfinite(hmax_oe) or hmax_oe <= 0:
        raise ValueError("Training Hmax must be a positive field in Oe.")
    measured_hmax = float(np.nanmax(np.abs(data["field_oe"].to_numpy(dtype=float))))
    if not np.isfinite(measured_hmax) or measured_hmax <= 0:
        raise ValueError("Uploaded file has no valid non-zero field values.")
    if field_range_mode not in ("strict", "map_to_training"):
        raise ValueError("field_range_mode must be strict or map_to_training.")
    # Keep the experimental Oe fields unchanged unless the user explicitly
    # chooses a *field-axis remapping* to the simulated training range.
    # Apply one scale to all temperatures so relative field values are preserved.
    field_divisor = float(hmax_oe)
    field_ratio = measured_hmax / field_divisor
    if field_ratio > 1.15:
        if field_range_mode == "strict":
            raise ValueError(
                f"Measured field reaches ±{measured_hmax:,.0f} Oe but the model was trained "
                f"with Hmax=±{hmax_oe:,.0f} Oe ({field_ratio:.2f}×). "
                "The .dat field units are already Oe. Enable explicit exploratory "
                "field-axis remapping if you want to map the measured range to ±1; "
                "this changes its physical field scale."
            )
        field_divisor = measured_hmax
        warnings.append(
            f"EXPLORATORY FIELD REMAPPING: experimental ±{measured_hmax:,.0f} Oe "
            f"mapped onto training ±{hmax_oe:,.0f} Oe (factor {field_ratio:.3g}). "
            "Switching/coercive fields are no longer in their original physical scale; "
            "confidence and attribution are not validated."
        )

    curves = []
    for i, T in enumerate(actual_targets):
        group = data.iloc[np.flatnonzero(nearest == i)].copy()
        if len(group) < 8:
            raise ValueError(f"Temperature {T:g} K has fewer than eight measurements.")
        h = group["field_oe"].to_numpy(dtype=float) / field_divisor
        m = group["moment_emu"].to_numpy(dtype=float) / moment_scale
        sd = group["moment_error_emu"].to_numpy(dtype=float) / moment_scale
        sd = np.where(np.isfinite(sd) & (sd >= 0), sd, 0.0)
        if not np.isfinite(h).all() or not np.isfinite(m).all():
            raise ValueError(f"Non-finite field/moment values at {T:g} K.")
        if np.max(np.abs(h)) > 1.15:
            raise ValueError(
                f"{T:g} K still exceeds the model field input range after preprocessing."
            )
        turn = int(np.argmin(h))
        if turn < 2 or turn > len(h) - 3:
            raise ValueError(
                f"At {T:g} K the field must run from +Hmax to -Hmax and back to +Hmax in acquisition order."
            )
        curves.append(np.stack([h, m, sd], axis=-1).astype(np.float32))
    return curves, actual_targets, warnings, moment_scale


def batch_for_uploaded(
    project_root: Path,
    model_name: str,
    curves,
    template_batch,
    actual_temperatures=None,
    trained_temperatures=None,
):
    """Replace model inputs with the uploaded curves and update actual normalized T values."""
    import torch
    from src import config

    n_t = len(curves)
    maxlen = max(len(x) for x in curves)
    padded = np.zeros((n_t, maxlen, 3), np.float32)
    lengths = np.zeros(n_t, dtype=np.int64)
    for t, c in enumerate(curves):
        padded[t, : len(c)] = c
        lengths[t] = len(c)
    x = torch.from_numpy(padded).unsqueeze(0)
    l = torch.from_numpy(lengths).unsqueeze(0)
    batch = list(template_batch)
    expected = list(map(float, trained_temperatures or config.TEMPERATURES_K))
    measured = list(map(float, actual_temperatures or expected))
    if len(measured) != n_t:
        raise ValueError("Temperature count differs from model input channels.")
    denominator = max(expected) - min(expected)
    tn = torch.tensor(
        [[(T - min(expected)) / denominator if denominator else 0.0 for T in measured]],
        dtype=torch.float32,
    )
    if model_name in ("rnn_joint", "m_data_check_RNN"):
        if len(batch) != 4:
            raise ValueError(f"Unexpected RNN batch of length {len(batch)}")
        batch[0], batch[1], batch[2] = x, l, tn
    elif "preisach" in model_name.lower():
        from src.models.preisach import PreisachDecomposer, PreisachConfig

        dec = PreisachDecomposer(float(config.Hmax_Oe) * 0.1, PreisachConfig())
        weights = []
        for t, c in enumerate(curves):
            fit = dec.fit_curve(c)
            if fit is None:
                raise ValueError(
                    f"Preisach decomposition failed for temperature {measured[t]:g} K."
                )
            weights.append(np.asarray(fit["x"], dtype=np.float32))
        w = torch.from_numpy(np.stack(weights)).unsqueeze(0)
        if w.shape != batch[0].shape:
            raise ValueError(
                f"Preisach feature mismatch: {w.shape} != {batch[0].shape}"
            )
        batch[0] = w
        if len(batch) == 5:
            batch[1], batch[2], batch[3] = x, l, tn
        elif len(batch) == 3:
            batch[1] = tn
        else:
            raise ValueError(f"Unexpected Preisach batch of length {len(batch)}")
    else:
        raise ValueError(
            "Image-based Neural networks require their original rasterizer; upload is currently supported for RNN and Preisach models."
        )
    return tuple(batch), padded, lengths
