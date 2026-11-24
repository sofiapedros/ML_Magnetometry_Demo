"""
Physics-informed M(H) consistency loss for the generic model pipeline.

This is a simplified version of the original M-data-check idea: it uses only
hysteresis M(H) curves (no ZFC/FC M(T) data).

For every sample it extracts six interpretable descriptors from the measured
loops:
    m_sat, mr, hc_mT, area, jump, rough

It then learns, from TRAINING data only, a typical [lo, hi] box for each class.
During training, predicted probability assigned to classes whose typical box is
incompatible with the measured descriptors is penalised:

    L = CE + lambda_phys * ramp * mean(gate * sum_c p_c * violation_c)

The gate downweights the physical prior when the input curves themselves fail
basic consistency checks (bounds, branch order, centro-symmetry, closure).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

FEATURES = ("m_sat", "mr", "hc_mT", "area", "jump", "rough")
N_FEATURES = len(FEATURES)
CLASSES = ("SPM", "SD", "MD")

# Natural scales used only to put descriptor violations on comparable units.
FEATURE_SCALE = torch.tensor([0.1, 0.1, 5.0, 0.05, 0.1, 0.02], dtype=torch.float32)

# Tolerances for the input-curve consistency gate.
TOL = {"bound": 0.02, "order": 0.05, "symmetry": 0.10, "closure": 0.10}


def _interp1d(x: torch.Tensor, y: torch.Tensor, x_new: torch.Tensor) -> torch.Tensor:
    """Linear interpolation for 1-D, monotonically increasing x."""
    idx = torch.searchsorted(x, x_new).clamp(1, x.numel() - 1)
    x0, x1 = x[idx - 1], x[idx]
    y0, y1 = y[idx - 1], y[idx]
    den = (x1 - x0).clamp_min(1e-8)
    w = (x_new - x0) / den
    return y0 + w * (y1 - y0)


def _at_zero(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), device=x.device, dtype=x.dtype)
    return _interp1d(x, y, zero)


def _zero_crossing(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """First zero crossing on an increasing field grid, linearly interpolated."""
    prod = y[:-1] * y[1:]
    cross = prod <= 0
    if bool(cross.any()):
        i = int(cross.float().argmax())
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]
        den = y0 - y1
        if float(den.abs()) < 1e-8:
            return x0
        return x0 + y0 / den * (x1 - x0)
    return x[y.abs().argmin()]


class MDataCheck(nn.Module):
    """
    Extract M(H) descriptors and a data-consistency gate from the existing
    CurveSequenceDataset batch format.

    Input:
        padded:  (B, n_T, max_len, 3), channels = [Bx_normalised, mx_mean, mx_std]
        lengths: (B, n_T)

    The field path is assumed to be the simulator path +Bmax -> -Bmax -> +Bmax.
    """

    def __init__(self, bx_max_mT: float, interp_points: int = 129):
        super().__init__()
        self.bx_max_mT = float(bx_max_mT)
        self.interp_points = int(interp_points)
        self.register_buffer("scale", FEATURE_SCALE.clone())

    def _one_curve(
        self, seq: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bx = seq[:, 0] * self.bx_max_mT
        m = seq[:, 1]

        # Turning point separates descending and ascending branches.
        # Handle empty or insufficiently sampled curves.
        if len(bx) < 5:
            z = torch.zeros(
                N_FEATURES,
                device=seq.device,
                dtype=seq.dtype,
            )
            bad = torch.ones(
                (),
                device=seq.device,
                dtype=seq.dtype,
            )
            return z, {k: bad for k in TOL}

        # Turning point separates descending and ascending branches.
        turn = int(bx.argmin())

        if turn < 2 or turn > len(bx) - 3:
            # Degenerate input: return finite placeholders and force gate down.
            z = torch.zeros(N_FEATURES, device=seq.device, dtype=seq.dtype)
            bad = torch.ones((), device=seq.device, dtype=seq.dtype)
            return z, {k: bad for k in TOL}

        b_desc = bx[: turn + 1].flip(0)  # increasing: -Bmax -> +Bmax
        m_desc = m[: turn + 1].flip(0)
        b_asc = bx[turn:]  # increasing: -Bmax -> +Bmax
        m_asc = m[turn:]

        # Common grid makes area/order/symmetry robust to slightly different
        # branch lengths or sampling.
        lo = torch.maximum(b_desc[0], b_asc[0])
        hi = torch.minimum(b_desc[-1], b_asc[-1])
        grid = torch.linspace(
            lo, hi, self.interp_points, device=seq.device, dtype=seq.dtype
        )
        d = _interp1d(b_desc, m_desc, grid)
        a = _interp1d(b_asc, m_asc, grid)

        m_sat = 0.5 * (m_desc[-1] + m_asc[-1])
        m_sat_safe = m_sat.abs().clamp_min(0.1)
        mr = 0.5 * (_at_zero(b_desc, m_desc) - _at_zero(b_asc, m_asc)) / m_sat_safe
        hc = 0.5 * (
            _zero_crossing(b_asc, m_asc) - _zero_crossing(b_desc, m_desc)
        ).clamp_min(0.0)
        area = torch.trapezoid(d - a, grid) / (2.0 * self.bx_max_mT) / m_sat_safe

        jump = torch.maximum(m_desc.diff().abs().amax(), m_asc.diff().abs().amax())
        rough = 0.5 * (m_desc.diff(n=2).abs().mean() + m_asc.diff(n=2).abs().mean())

        feats = torch.stack([m_sat, mr, hc, area, jump, rough])

        both = torch.cat([m_desc, m_asc])
        bound = (both.abs() - 1.0).clamp_min(0.0).mean()
        order = (a - d).clamp_min(0.0).mean()
        symmetry = (d + a.flip(0)).abs().mean()
        closure = 0.5 * ((m_desc[-1] - m_asc[-1]).abs() + (m_desc[0] - m_asc[0]).abs())

        return feats, {
            "bound": bound,
            "order": order,
            "symmetry": symmetry,
            "closure": closure,
        }

    @torch.no_grad()
    def forward(
        self, padded: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        B, n_T = padded.shape[:2]
        feats = torch.zeros(B, N_FEATURES, device=padded.device, dtype=padded.dtype)
        checks = {
            k: torch.zeros(B, device=padded.device, dtype=padded.dtype) for k in TOL
        }

        for b in range(B):
            per_t = []
            per_check = {k: [] for k in TOL}
            for t in range(n_T):
                n = int(lengths[b, t])
                f_t, c_t = self._one_curve(padded[b, t, :n])
                per_t.append(f_t)
                for k in TOL:
                    per_check[k].append(c_t[k])

            feats[b] = torch.stack(per_t).mean(0)
            for k in TOL:
                checks[k][b] = torch.stack(per_check[k]).mean()

        valid = torch.ones_like(feats)
        violation = sum(torch.relu(checks[k] - TOL[k]) / TOL[k] for k in TOL)
        gate = torch.exp(-violation)
        return feats, valid, gate, checks


class ClassPriors(nn.Module):
    """Training-data typical boxes [lo_c, hi_c] for every class/descriptor."""

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor):
        super().__init__()
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.register_buffer("scale", FEATURE_SCALE.clone())

    @classmethod
    def fit_from_data(
        cls,
        feats: torch.Tensor,
        labels: torch.Tensor,
        k: float = 1.5,
        min_sigma_frac: float = 0.25,
    ) -> "ClassPriors":
        C = len(CLASSES)
        lo = torch.full((C, N_FEATURES), float("-inf"), dtype=feats.dtype)
        hi = torch.full((C, N_FEATURES), float("inf"), dtype=feats.dtype)

        for c in range(C):
            x_c = feats[labels == c]
            if len(x_c) < 2:
                continue
            mean = x_c.mean(0)
            std = x_c.std(0)
            floor = min_sigma_frac * FEATURE_SCALE.to(feats)
            std = torch.maximum(std, floor)
            lo[c] = mean - k * std
            hi[c] = mean + k * std
        return cls(lo, hi)

    def violation(self, feats: torch.Tensor) -> torch.Tensor:
        """(B,F) -> (B,C), zero when all descriptors lie inside class box."""
        f = feats[:, None, :]
        d = (
            torch.relu(self.lo[None] - f).square()
            + torch.relu(f - self.hi[None]).square()
        )
        d = d / self.scale.square()
        return d.mean(-1)


def m_data_check_loss(
    probs: torch.Tensor,
    feats: torch.Tensor,
    gate: torch.Tensor,
    priors: ClassPriors,
) -> torch.Tensor:
    """Expected class-box violation under the network's predicted probabilities."""
    v = priors.violation(feats)
    return (gate * (probs * v).sum(-1)).mean()


def prepare_m_data_check(
    train_loader: DataLoader, device, *, bx_max_mT: float, prior_k: float
):
    """
    Fit priors from the current training loader only.

    This is called inside each CV fold as well as final training, so validation
    and test samples never contribute to the physical prior.
    """
    check = MDataCheck(bx_max_mT=bx_max_mT).to(device)

    # Fresh non-shuffled loader: do not consume the training loader RNG/order.
    loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        collate_fn=train_loader.collate_fn,
    )

    all_feats, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            # CurveSequenceDataset:      (padded, lengths, temperatures, labels)
            # ImageCurveDataset:         (images, padded, lengths, temperatures, labels)
            if len(batch) == 4:
                padded, lengths, _temperatures, labels = batch
            elif len(batch) == 5:
                _images, padded, lengths, _temperatures, labels = batch
            else:
                raise ValueError(
                    f"Unsupported batch format for M-data-check: {len(batch)} fields"
                )
            padded, lengths = padded.to(device), lengths.to(device)
            feats, _valid, _gate, _checks = check(padded, lengths)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())

    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    priors = ClassPriors.fit_from_data(feats, labels, k=prior_k).to(device)
    return {"check": check, "priors": priors}
