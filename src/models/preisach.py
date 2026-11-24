"""
Preisach input: replace every M(H) loop by a SUM OF SQUARE HYSTERONS.

The hysteron (the Preisach "relay")
-----------------------------------
A hysteron gamma_{alpha,beta} is a rectangular loop with output +1 / -1:

        +1 |        ______________          it switches UP   when H rises above alpha
           |       |      ^       |         it switches DOWN when H falls below beta
    -------+-------|------|-------|-----    (alpha >= beta)
           |  _____|______|       |
        -1 |
              beta        alpha

Equivalent parameters:  coercivity hc = (alpha - beta)/2,  bias (interaction) field
hu = (alpha + beta)/2.  Every square loop in the (alpha, beta) half plane
alpha >= beta is one hysteron: that half plane is the PREISACH PLANE.

The model:  M(H) = sum_k  w_k * gamma_{alpha_k, beta_k}[H],   w_k >= 0
(the Preisach distribution mu(alpha, beta) discretised on a grid of cells).

On the major loop (+Bmax -> -Bmax -> +Bmax) every hysteron starts at +1 and
    descending branch:  M_desc(H) = sum_k w_k * sign(H - beta_k )   (flips down at beta)
    ascending  branch:  M_asc (H) = sum_k w_k * sign(H - alpha_k)   (flips up   at alpha)

How the weights are obtained (exactly)
--------------------------------------
1. Resample the measured loop.  Split it at the turning point (-Bmax) and
   interpolate the descending and ascending branches onto ONE common field grid
   E (n_eval points, denser around H = 0).   y = [M_desc(E), M_asc(E)].
2. Build the dictionary.  Field nodes g_1 < ... < g_n (n_nodes, denser around 0)
   are the possible switching fields.  Every pair (alpha = g_i, beta = g_j) with
   j <= i is one square hysteron: n (n + 1) / 2 cells of the Preisach plane.
   (Cells on the diagonal i = j have hc = 0: reversible, no-loop hysterons; that
   is how superparamagnetic S-curves are represented.)
   Column c of the matrix A is that hysteron on the loop:
       [ sign(E - beta_c) ; sign(E - alpha_c) ]            (length 2 * n_eval)
3. Solve   min_{w >= 0}  || A w - y ||^2 + w' Q w
   with Q = lam_ridge I + lam_smooth (neighbouring cells get similar weight)
                        + lam_bias diag((hu/Bmax)^2)  (prefer weakly biased hysterons).
   Exactly, with NNLS after a Cholesky reformulation (R'R = A'A + Q).
4. The vector w IS the sum of hysterons: w_c = how much of the loop is switched by
   the square hysteron (alpha_c, beta_c).  It is the model input, one vector per
   temperature.  sum(w) is the saturation magnetisation (units of the loop).

Honest limits
  * A single major loop only fixes the two MARGINALS of the Preisach distribution
    (descending branch -> distribution of beta, ascending -> distribution of alpha).
    How the weight is split between hc and hu is NOT measured; Q is the explicit
    prior that picks one solution (smooth, and as little bias as possible).  The
    well-determined information is the pair of marginals and the total weight.
    Real FORC data would identify the 2-D distribution; it is not needed here.
  * The model is centro-symmetric about H = 0 only through that prior, not by
    construction, and fit_rmse tells you how well the loop is described at all.

Contents
  PreisachConfig, PreisachDecomposer           (numpy only)
  PreisachFeatureDataset                       (cached weights per sample)
  ResampledLoopDataset                         (same samples, resampled loop as input)
  PreisachCurveDataset                         (weights + the raw curves, for the M-data-check loss)
  PlaneImage, PreisachImageNet, PreisachRNN    (the original CNN / MLP / RNN fed with the plane)
  PreisachMLP, PreisachSetClassifier, preisach_forward, fit_feature_scaler
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear, nnls
from torch import nn
from torch.utils.data import Dataset

try:
    from src import config
except ModuleNotFoundError:
    import config

try:
    from src.dataset_ml.datasets_ml import CurveSequenceDataset, collate_curve_batches
    from src.models.basic_models import CurveRNNEncoder, RNNClassifier
except ModuleNotFoundError:
    from basic_models import CurveRNNEncoder, RNNClassifier

    from dataset_ml.datasets_ml import CurveSequenceDataset, collate_curve_batches


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PreisachConfig:
    n_nodes: int = 24  # switching-field nodes -> n(n+1)/2 hysteron cells
    node_max_frac: float = 0.98  # outermost node, fraction of Bmax
    node_warp: float = 3.0  # >0: nodes denser around H = 0
    n_eval: int = 129  # evaluation grid per branch
    warp: float = 3.0  # >0: evaluation grid denser around H = 0
    lam_ridge: float = 1e-3
    lam_smooth: float = 1e-2
    lam_bias: float = 1e-2


def n_features(cfg: PreisachConfig = PreisachConfig()) -> int:
    """Length D of the weight vector of one loop (number of hysteron cells)."""
    return cfg.n_nodes * (cfg.n_nodes + 1) // 2


class PreisachDecomposer:
    """Fits the square-hysteron dictionary to one loop. Build once per Bmax, reuse for all curves."""

    def __init__(self, bx_max_mT: float, cfg: PreisachConfig = PreisachConfig()):
        self.cfg = cfg
        self.bmax = float(bx_max_mT)
        c = cfg

        u = np.linspace(-1.0, 1.0, c.n_eval)
        self.E = self.bmax * (
            np.sinh(c.warp * u) / np.sinh(c.warp) if c.warp > 0 else u
        )

        n = c.n_nodes
        v = np.linspace(-1.0, 1.0, n)
        g = np.sinh(c.node_warp * v) / np.sinh(c.node_warp) if c.node_warp > 0 else v
        self.nodes = c.node_max_frac * self.bmax * g  # switching-field nodes (mT)

        # cell c <-> (i, j), j <= i : alpha = nodes[i] (switch up), beta = nodes[j] (switch down)
        self.cell_i, self.cell_j = np.array(
            [(i, j) for i in range(n) for j in range(i + 1)]
        ).T
        self.alpha = self.nodes[self.cell_i]
        self.beta = self.nodes[self.cell_j]
        self.hc = 0.5 * (self.alpha - self.beta)
        self.hu = 0.5 * (self.alpha + self.beta)
        self.n_atoms = len(self.alpha)

        E = self.E[:, None]
        self.A = np.vstack(
            [
                np.where(E > self.beta[None], 1.0, -1.0),  # descending branch
                np.where(E > self.alpha[None], 1.0, -1.0),
            ]
        )  # ascending branch

        # Regularisation: ridge + smoothness between neighbouring cells + weak-bias prior.
        index = {
            (int(i), int(j)): k
            for k, (i, j) in enumerate(zip(self.cell_i, self.cell_j))
        }
        rows = []
        for (i, j), k in index.items():
            for nb in ((i + 1, j), (i, j + 1)):
                if nb in index:
                    r = np.zeros(self.n_atoms)
                    r[k], r[index[nb]] = 1.0, -1.0
                    rows.append(r)
        Dm = np.array(rows)
        Q = (
            c.lam_ridge * np.eye(self.n_atoms)
            + c.lam_smooth * Dm.T @ Dm
            + c.lam_bias * np.diag((self.hu / self.bmax) ** 2)
        )
        # min ||A x - y||^2 + x'Qx  ==  min ||R x - R^-T A'y||^2 with R'R = A'A + Q
        self.R = np.linalg.cholesky(self.A.T @ self.A + Q).T

    # -- resampling -----------------------------------------------------
    def resample(self, seq: np.ndarray) -> np.ndarray | None:
        """
        seq: (n, 3) = [Bx_normalised, mx_mean, mx_std] along +Bmax -> -Bmax -> +Bmax.
        Returns y = [M_desc(E), M_asc(E)] or None if the loop is degenerate.
        """
        seq = np.asarray(seq, dtype=float)
        if seq.ndim != 2 or len(seq) < 8 or not np.isfinite(seq[:, :2]).all():
            return None
        bx, m = seq[:, 0] * self.bmax, seq[:, 1]
        turn = int(np.argmin(bx))
        if turn < 2 or turn > len(bx) - 3:
            return None
        b_d, m_d = bx[: turn + 1][::-1], m[: turn + 1][::-1]  # increasing field
        b_a, m_a = bx[turn:], m[turn:]
        od, oa = np.argsort(b_d, kind="stable"), np.argsort(b_a, kind="stable")
        return np.concatenate(
            [np.interp(self.E, b_d[od], m_d[od]), np.interp(self.E, b_a[oa], m_a[oa])]
        )

    # -- fit ------------------------------------------------------------
    def solve(self, y: np.ndarray) -> np.ndarray:
        z = solve_triangular(self.R, self.A.T @ y, trans="T", lower=False)
        try:
            x, _ = nnls(self.R, z, maxiter=50 * self.R.shape[1])
        except RuntimeError:
            x = lsq_linear(self.R, z, bounds=(0.0, np.inf), method="bvls").x
        return x

    def fit_curve(self, seq: np.ndarray) -> dict | None:
        """{"x": weights, "rmse": float, "y": resampled loop, "yhat": sum of hysterons} or None."""
        y = self.resample(seq)
        if y is None:
            return None
        x = self.solve(y)
        yhat = self.A @ x
        return {
            "x": x,
            "rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
            "y": y,
            "yhat": yhat,
        }

    # -- helpers for plots / inspection --------------------------------
    def plane(self, x: np.ndarray) -> np.ndarray:
        """Weights as a (n_nodes, n_nodes) matrix P[j, i] = weight of (alpha=nodes[i], beta=nodes[j]); NaN where alpha < beta."""
        n = self.cfg.n_nodes
        P = np.full((n, n), np.nan)
        P[self.cell_j, self.cell_i] = x
        return P

    def marginals(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(weight per alpha node, weight per beta node): the two things the loop actually measures."""
        n = self.cfg.n_nodes
        return np.bincount(self.cell_i, x, n), np.bincount(self.cell_j, x, n)

    def hysteron(
        self, c: int, w: float = 1.0, H: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Descending / ascending branch of w * hysteron c on the field array H (default: E)."""
        H = self.E if H is None else H
        return (
            w * np.where(H > self.beta[c], 1.0, -1.0),
            w * np.where(H > self.alpha[c], 1.0, -1.0),
        )

    def cell_label(self, c: int) -> str:
        return f"alpha={self.alpha[c]:.0f}, beta={self.beta[c]:.0f} mT (hc={self.hc[c]:.0f}, hu={self.hu[c]:.0f})"


# --------------------------------------------------------------------------
# Dataset (decompositions are computed once and cached)
# --------------------------------------------------------------------------
class PreisachFeatureDataset(Dataset):
    """
    Wraps CurveSequenceDataset: each sample becomes
        (weights (n_T, D) float32, temperatures (n_T,), label ()).
    The weights are the square-hysteron (Preisach cell) decomposition of every temperature's loop.
    Also keeps, for inspection or baselines on the same samples:
        .loops    (N, n_T, 2*n_eval)  the resampled loops that were fitted
        .fit_rmse (N, n_T)            reconstruction error (1.0 = degenerate loop)
    Cached in <manifest dir>/preisach_cache/, keyed by a hash of the curves and
    the PreisachConfig.  Also exposes `.rows` (used by tune.py for grouped CV).
    """

    def __init__(
        self,
        manifest_path,
        split: str,
        cfg: PreisachConfig | None = None,
        bx_max_mT: float | None = None,
        cache_dir: Path | None = None,
    ):
        cfg = cfg or PreisachConfig()
        bmax = float(bx_max_mT if bx_max_mT is not None else config.Hmax_Oe * 0.1)
        base = CurveSequenceDataset(manifest_path, split=split)
        self.base = base
        self.rows = getattr(base, "rows", None)
        self.cfg, self.bx_max_mT = cfg, bmax

        curves, lengths, temps, labels = [], [], [], []
        for i in range(len(base)):
            padded, lens, t, y = collate_curve_batches([base[i]])
            curves.append(padded[0].numpy())
            lengths.append(lens[0].numpy())
            temps.append(t[0].numpy())
            labels.append(int(y[0]))

        h = hashlib.sha1(
            json.dumps(
                {"cfg": asdict(cfg), "bmax": bmax, "v": 3}, sort_keys=True
            ).encode()
        )
        for c_, l_ in zip(curves, lengths):
            h.update(np.ascontiguousarray(c_).tobytes())
            h.update(np.ascontiguousarray(l_).tobytes())
        cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(manifest_path).parent / "preisach_cache"
        )
        cache_file = cache_dir / f"preisach_{split}_{h.hexdigest()[:16]}.npz"

        if cache_file.is_file():
            with np.load(cache_file) as z:
                weights, loops, rmse = z["weights"], z["loops"], z["rmse"]
        else:
            dec = PreisachDecomposer(bmax, cfg)
            n_T, D, L = len(curves[0]), n_features(cfg), 2 * cfg.n_eval
            weights = np.zeros((len(curves), n_T, D), np.float32)
            loops = np.zeros((len(curves), n_T, L), np.float32)
            rmse = np.ones((len(curves), n_T), np.float32)  # 1.0 flags degenerate loops
            for i, (c_, l_) in enumerate(zip(curves, lengths)):
                for t in range(n_T):
                    fit = dec.fit_curve(c_[t, : int(l_[t])])
                    if fit is not None:
                        weights[i, t], loops[i, t], rmse[i, t] = (
                            fit["x"],
                            fit["y"],
                            fit["rmse"],
                        )
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_file, weights=weights, loops=loops, rmse=rmse)

        n_bad = int((rmse >= 1.0).sum())
        if n_bad:
            print(f"[preisach] {split}: {n_bad} degenerate loops got all-zero weights.")
        self.weights = torch.from_numpy(weights)
        self.loops = loops
        self.fit_rmse = rmse
        self.temperatures = torch.from_numpy(np.stack(temps)).float()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return self.weights[i], self.temperatures[i], self.labels[i]


# --------------------------------------------------------------------------
# Models (same head design as the RNN pipeline, only the input changes)
# --------------------------------------------------------------------------
class _GlobalScale(nn.Module):
    """x / s with ONE scalar s (std of all training weights). A per-feature
    standardisation would blow up atoms that are almost always zero."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


class PreisachMLP(nn.Module):
    """Flat MLP over the hysteron weights of all temperatures."""

    def __init__(
        self,
        n_temperatures: int,
        n_features: int,
        num_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.scaler = _GlobalScale()
        layers: list[nn.Module] = [nn.Flatten()]
        dim = n_temperatures * n_features
        for _ in range(num_layers):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(
        self, weights: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        # weights: (B, n_T, D); temperatures unused (the position identifies T)
        return self.net(self.scaler(weights))


class PreisachSetClassifier(nn.Module):
    """
    Shared per-temperature encoder over the hysteron weights -> concatenated (+T)
    -> head.  Mirrors RNNClassifier with the GRU replaced by an MLP encoder.
    """

    def __init__(
        self,
        n_temperatures: int,
        n_features: int,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.scaler = _GlobalScale()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(n_temperatures * (hidden_dim + 1), 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, num_classes),
        )

    def forward(
        self, weights: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        z = self.encoder(self.scaler(weights))  # (B, n_T, hidden)
        z = torch.cat([z, temperatures.unsqueeze(-1)], dim=-1)  # + normalised T
        return self.head(z.flatten(1))


def preisach_forward(model, batch, device):
    weights, temperatures, labels = batch
    return model(weights.to(device), temperatures.to(device)), labels.to(device)


@torch.no_grad()
def fit_feature_scaler(model: nn.Module, train_loader, hp: dict, device) -> None:
    """`pretrain` hook: the scale comes from the TRAINING loader only (inside CV
    folds that is the fitting fold), so there is no val/test leakage."""
    x = torch.cat([w.reshape(-1) for w, *_ in train_loader])
    model.scaler.scale.fill_(float(x.std().clamp_min(1e-6)))


class ResampledLoopDataset(PreisachFeatureDataset):
    """Same samples, same cache, but the input is the resampled loop itself
    ((n_T, 2*n_eval) = [M_desc(E), M_asc(E)]) instead of its hysteron weights.
    Used for the 'only the representation changes' baselines."""

    def __getitem__(self, i):
        return (
            torch.from_numpy(np.ascontiguousarray(self.loops[i])),
            self.temperatures[i],
            self.labels[i],
        )


class PreisachCurveDataset(PreisachFeatureDataset):
    """Weights AND the raw curve of each sample, for models whose loss also needs the
    measured curves (M-data-check physics term).  Batch (see collate_preisach_curve_batches):
    (weights, padded, lengths, temperatures, labels)."""

    def __getitem__(self, i):
        return self.weights[i], self.base[i]


def collate_preisach_curve_batches(batch):
    weights = torch.stack([b[0] for b in batch])
    padded, lengths, temperatures, labels = collate_curve_batches([b[1] for b in batch])
    return weights, padded, lengths, temperatures, labels


def preisach_curve_forward(model, batch, device):
    weights, _padded, _lengths, temperatures, labels = batch
    return model(weights.to(device), temperatures.to(device)), labels.to(device)


# --------------------------------------------------------------------------
# The ORIGINAL architectures with the Preisach plane as input
# --------------------------------------------------------------------------
class PlaneImage(nn.Module):
    """(B, n_T, D) hysteron weights -> (B, n_T, n, n) Preisach-plane image
    (row = beta node, column = alpha node, zeros above the diagonal)."""

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n = n_nodes
        i, j = np.array([(i, j) for i in range(n_nodes) for j in range(i + 1)]).T
        self.register_buffer("flat_idx", torch.from_numpy(j * n_nodes + i).long())

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        B, T, _ = w.shape
        out = w.new_zeros(B, T, self.n * self.n)
        out[..., self.flat_idx] = w
        return out.view(B, T, self.n, self.n)


class PreisachImageNet(nn.Module):
    """CNNClassifier / MLPClassifier (basic_models.py, unchanged) over the Preisach
    plane: temperatures are the channels, exactly like the stacked hysteresis images."""

    def __init__(self, backbone: nn.Module, n_nodes: int):
        super().__init__()
        self.scaler = _GlobalScale()
        self.plane = PlaneImage(n_nodes)
        self.backbone = backbone

    def forward(
        self, weights: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        return self.backbone(self.plane(self.scaler(weights)))


class PreisachRNN(nn.Module):
    """RNNClassifier (basic_models.py, unchanged) over the plane: for every temperature
    the GRU reads the plane column by column (alpha = switch-up field, low -> high),
    each step being the weights over beta; same shared encoder + (+T) head."""

    def __init__(
        self,
        n_temperatures: int,
        n_nodes: int,
        num_classes: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.scaler = _GlobalScale()
        self.plane = PlaneImage(n_nodes)
        self.n = n_nodes
        self.rnn = RNNClassifier(
            CurveRNNEncoder(n_nodes, hidden_dim, num_layers),
            n_temperatures,
            num_classes,
        )

    def forward(
        self, weights: torch.Tensor, temperatures: torch.Tensor
    ) -> torch.Tensor:
        seq = self.plane(self.scaler(weights)).transpose(
            2, 3
        )  # (B, n_T, alpha steps, beta)
        lengths = torch.full(seq.shape[:2], self.n, dtype=torch.long, device=seq.device)
        return self.rnn(seq, lengths, temperatures)
