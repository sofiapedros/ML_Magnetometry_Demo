"""
Bayesian logistic regression as a regular registry model.

`BayesLogRegModel` is an nn.Module that holds the (closed-form) Laplace
posterior of bayes_logreg.BayesianLogReg in buffers, so it goes through the
same pipeline as every network:

    registry.py   ModelSpec(fit_fn=fit_bayes_logreg, mc_default=...)
    experiment    fit() calls spec.fit_fn instead of the gradient loop;
                  predict() -> posterior predictive; predict_mc() -> one posterior
                  draw per pass (module.resample_posterior)
    train_seeds   results/runs/<model>/seed_<s>.npz incl. val/test_mc_probs
    tune          hyperparameter `alpha` (prior precision) by grouped CV
    eval_models   picks the run up like any other; the posterior draws are its
                  "mc_samples" source -> epistemic uncertainty (mutual information)

The fit is deterministic (no epochs, no batches): the seed only changes which
posterior draws predict_mc samples.

The input size does not have to be known when the model is built: it is read
from the training data in `fit_fn` (e.g. raw curves of L points x n_T temperatures
x 3 channels), and restored from the state dict when a checkpoint is loaded.

Hyperparameters (hp):
    alpha_mode  "evidence" (default): prior precision by marginal likelihood on the
                training set;  "fixed": use hp["alpha"]
    alpha       prior precision, only used with alpha_mode == "fixed"
    n_post      number of fixed posterior draws averaged by the plain forward pass
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

try:
    from src.models.bayes_logreg import BayesianLogReg
except ModuleNotFoundError:
    from bayes_logreg import BayesianLogReg


class BayesLogRegModel(nn.Module):
    """
    in_features: inputs AFTER the optional pooling; None = take it from the training data.
    pool_size:   with it the model accepts (B, C, H, W) images of any size and average-pools
                 them to (C, pool, pool) first, exactly like MLPClassifier.
    Any other input (e.g. (B, n_T, L, 3) padded curves) is simply flattened.
    """

    def __init__(
        self,
        in_features: int | None = None,
        n_classes: int = 3,
        n_post: int = 200,
        pool_size: int | None = None,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)) if pool_size else None
        self.K, self.n_post = n_classes, n_post
        self.D = 0
        self.mc_mode = False
        self._alloc(in_features or 0, 0)

    def _alloc(self, D: int, r: int) -> None:
        """(Re)create the buffers for D inputs and an r-dimensional data subspace."""
        self.D, self.r = D, r
        K, P = self.K, r + 1
        # keep the device of the existing buffers (the model may already be on the GPU)
        dev = self.scale.device if hasattr(self, "scale") else None
        z = lambda *shape: torch.zeros(*shape, device=dev)
        for name, t in {
            "mu": z(D),
            "scale": torch.ones((), device=dev),
            "basis": z(D, r),  # V: data subspace
            "theta_sub": z(K, P),  # MAP in the subspace (+ bias)
            "chol": z(K * P, K * P),  # posterior covariance factor there
            "theta": z(K, D + 1),  # MAP, original input space
            "post_theta": z(self.n_post, K, D + 1),  # fixed posterior draws
            "mc_theta": z(K, D + 1),  # current MC draw
            "alpha": z(()),
            "log_evidence": z(()),
            "eff_params": z(()),
        }.items():
            if hasattr(self, name):
                setattr(self, name, t)
            else:
                self.register_buffer(name, t)

    def load_state_dict(self, state_dict, strict: bool = True, **kw):
        self._alloc(state_dict["mu"].numel(), state_dict["basis"].shape[1])
        return super().load_state_dict(state_dict, strict=strict, **kw)

    def flat(self, x: torch.Tensor) -> torch.Tensor:
        """Raw input -> (B, D): optional pooling, then flatten."""
        x = x.float()
        if self.pool is not None:
            x = self.pool(x)
        return x.flatten(1)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        f = self.flat(x)
        if f.shape[1] != self.D:
            raise ValueError(
                f"BayesLogRegModel expects {self.D} inputs per sample, got {f.shape[1]} "
                "(are all curves padded to the same length?)"
            )
        z = (f - self.mu) / self.scale
        return torch.cat([z, torch.ones(len(z), 1, device=z.device)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._phi(x)
        if self.mc_mode:  # one posterior draw
            return phi @ self.mc_theta.T
        # posterior predictive: mean of the softmax over the fixed draws, returned as
        # log-probabilities so softmax(logits) is exactly that mean.
        logits = torch.einsum("skp,np->nsk", self.post_theta, phi)
        p = torch.softmax(logits, dim=-1).mean(1)
        return torch.log(p.clamp_min(1e-12))

    @torch.no_grad()
    def resample_posterior(self) -> None:
        """Draw the weights from the posterior with the global torch RNG (seeded by set_seed)."""
        K, r, P = self.K, self.r, self.r + 1
        th = self.theta_sub + (
            self.chol @ torch.randn(K * P, device=self.chol.device)
        ).view(K, P)
        G = torch.randn(K, self.D, device=self.chol.device) / self.alpha.sqrt()
        G_perp = (
            G - (G @ self.basis) @ self.basis.T
        )  # prior draw outside the data subspace
        self.mc_theta.copy_(
            torch.cat([th[:, :r] @ self.basis.T + G_perp, th[:, r:]], dim=1)
        )

    @torch.no_grad()
    def load_posterior(
        self, bl: BayesianLogReg, n_post: int | None = None, seed: int = 0
    ) -> None:
        self.n_post = n_post or self.n_post
        self._alloc(bl.D, bl.V_.shape[1])
        f = lambda a: torch.from_numpy(a).float()
        self.mu.copy_(f(bl.mu_))
        self.scale.fill_(float(bl.s_))
        self.basis.copy_(f(bl.V_))
        self.theta_sub.copy_(f(bl.theta_).view(self.K, self.r + 1))
        self.chol.copy_(f(bl.L_))
        self.theta.copy_(f(bl.map_weights()))
        self.post_theta.copy_(f(bl.sample_weights(self.n_post, seed)))
        self.alpha.fill_(float(bl.alpha_))
        self.log_evidence.fill_(float(bl.log_evidence_))
        self.eff_params.fill_(float(bl.eff_params_))


def bayes_forward(model, batch, device):
    """(weights | loops, temperatures, labels) batches of the Preisach datasets."""
    x, _temperatures, labels = batch
    return model(x.to(device)), labels.to(device)


def bayes_image_forward(model, batch, device):
    """(images, labels) batches of ImageSampleDataset (same as cnn / mlp)."""
    images, labels = batch
    return model(images.to(device)), labels.to(device)


def bayes_curve_forward(model, batch, device):
    """(padded, lengths, temperatures, labels) batches of CurveSequenceDataset (same as the RNNs)."""
    padded, _lengths, _temperatures, labels = batch
    return model(padded.to(device)), labels.to(device)


def fit_bayes_logreg(model: BayesLogRegModel, train_loader, hp: dict, device) -> None:
    """`fit_fn` hook: closed-form Laplace fit on the TRAINING loader only
    (inside tune.py's CV that is the fitting fold, so the held-out fold never leaks)."""
    xs, ys = [], []
    for batch in train_loader:  # input first, label last, whatever the batch format
        xs.append(model.flat(batch[0]).numpy())
        ys.append(batch[-1].numpy())
    X, y = np.concatenate(xs), np.concatenate(ys)
    alpha = float(hp["alpha"]) if hp.get("alpha_mode", "evidence") == "fixed" else None
    bl = BayesianLogReg(model.K, alpha=alpha).fit(X, y)
    model.load_posterior(bl)
