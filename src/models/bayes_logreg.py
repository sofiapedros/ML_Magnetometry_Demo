"""
Bayesian multinomial logistic regression (numpy / scipy only).

Model
-----
    p(y = k | x, W) = softmax_k( W_k . phi(x) ),     phi(x) = [x_scaled, 1]
    prior           W_kj ~ N(0, 1/alpha)   (bias: N(0, 1/alpha_bias), weakly informative)

Inference
---------
1. MAP:       W* = argmax  log p(y | X, W) + log p(W | alpha)      (L-BFGS)
2. Laplace:   p(W | D) ~ N(W*, H^-1),  H = Hessian of the negative log posterior
              = sum_n (diag(p_n) - p_n p_n^T) (x) phi_n phi_n^T  + A
3. Prior strength alpha: never tuned on val/test. It is chosen by maximising the
              Laplace approximation of the marginal likelihood p(D | alpha)
              (MacKay's fixed point  alpha <- gamma / ||W||^2 ), i.e. from the
              training data alone.

              Known tendency: with N << D and many irrelevant inputs ONE isotropic alpha
              cannot tell useful from useless dimensions. Directions of input space that
              the training set never explores keep their prior variance, which blurs
              predictions for test points that have a component there, so the evidence
              picks a strong prior and the predictions come out UNDERCONFIDENT (the ranking
              and the epistemic uncertainty stay informative). Temperature scaling
              (eval_models.py --calibrate) fixes the confidence level afterwards.
4. Predict:   draw S weight vectors from the posterior and softmax them ->
              S probability vectors per sample (S, N, C). Their mean is the
              predictive distribution, and their spread gives the EPISTEMIC
              uncertainty (mutual information) used by eval_models.py.

Why this is useful here: with ~10^2 samples and 10^2-10^3 input dimensions the
posterior over W is wide in the directions the data do not constrain; a test
loop that uses features never seen in training gets a large predictive spread.

Inputs are centred with the training mean and divided by ONE global scalar
(the std of all training entries), so that sparse hysteron weights are not blown
up feature by feature.

Many inputs, few samples (exact reduction)
------------------------------------------
With D inputs and N < D training samples the likelihood only "sees" the r <= N
directions V (D x r) spanned by the centred training inputs. With the isotropic
prior W_k ~ N(0, 1/alpha) the weights split as W_k = V a_k + W_perp,k and

  * (a, bias) get the Laplace posterior, computed in r+1 dimensions instead of D+1
    (same MAP, same evidence, same alpha updates: the D - r extra directions
    contribute identical terms to log|H| and to the prior normalisation, so they cancel);
  * W_perp keeps its prior N(0, 1/alpha) (no data constrains it), and is sampled
    in full space when predicting.

This is why raw curves with thousands of points per sample are affordable: the
Hessian is (K (r+1))^2, not (K (D+1))^2, and nothing is approximated by it.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


class BayesianLogReg:
    def __init__(
        self,
        n_classes: int = 3,
        alpha: (
            float | None
        ) = None,  # fixed prior precision; None -> evidence maximisation
        alpha_init: float = 1.0,
        bias_precision: float = 1e-2,
        max_outer: int = 25,
        tol: float = 1e-3,
        max_map_iter: int = 500,
    ):
        self.K = n_classes
        self.alpha_fixed = alpha
        self.alpha_init = alpha_init
        self.alpha_b = bias_precision
        self.max_outer, self.tol, self.max_map_iter = max_outer, tol, max_map_iter

    # ------------------------------------------------------------------
    def _z(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mu_) / self.s_

    def _phi(self, X: np.ndarray) -> np.ndarray:
        """Features in the data subspace: [Z V, 1]  (N x (r+1))."""
        Z = self._z(X)
        return np.concatenate([Z @ self.V_, np.ones((len(Z), 1))], axis=1)

    def _prior_diag(self, alpha: float) -> np.ndarray:
        a = np.full((self.K, self.P), alpha)
        a[:, -1] = self.alpha_b
        return a.ravel()

    def _neg_log_post(self, theta, Phi, Y, a_diag):
        W = theta.reshape(self.K, self.P)
        logits = Phi @ W.T
        lse = logsumexp(logits, axis=1)
        nll = float((lse - (logits * Y).sum(1)).sum())
        grad = ((softmax(logits, axis=1) - Y).T @ Phi).ravel() + a_diag * theta
        return nll + 0.5 * float((a_diag * theta * theta).sum()), grad

    def _map(self, Phi, Y, a_diag, theta0):
        res = minimize(
            self._neg_log_post,
            theta0,
            args=(Phi, Y, a_diag),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_map_iter, "maxfun": 4 * self.max_map_iter},
        )
        return res.x

    def _hessian(self, theta, Phi, a_diag):
        W = theta.reshape(self.K, self.P)
        p = softmax(Phi @ W.T, axis=1)  # (N, K)
        S = np.einsum("nk,kl->nkl", p, np.eye(self.K)) - np.einsum("nk,nl->nkl", p, p)
        H = np.einsum("nkl,ni,nj->kilj", S, Phi, Phi, optimize=True)
        d = self.K * self.P
        H = H.reshape(d, d)
        H[np.diag_indices(d)] += a_diag
        return 0.5 * (H + H.T)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogReg":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.mu_ = X.mean(0)
        self.s_ = max(float((X - self.mu_).std()), 1e-8)
        Z = self._z(X)
        self.D = Z.shape[1]
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        self.V_ = Vt[S > 1e-10 * max(float(S[0]), 1e-300)].T  # (D, r) data subspace
        Phi = self._phi(X)
        N, self.P = Phi.shape  # P = r + 1
        Y = np.eye(self.K)[y]
        d = self.K * self.P
        n_w = self.K * (self.P - 1)  # number of weight (non-bias) parameters
        w_mask = np.ones((self.K, self.P), bool)
        w_mask[:, -1] = False
        w_mask = w_mask.ravel()

        single_pass = self.alpha_fixed is not None
        alpha = self.alpha_fixed if single_pass else self.alpha_init
        theta = np.zeros(d)
        self.alpha_history_ = []
        n_outer = 1 if single_pass else self.max_outer

        for it in range(n_outer):
            a_diag = self._prior_diag(alpha)
            theta = self._map(Phi, Y, a_diag, theta)
            H = self._hessian(theta, Phi, a_diag)
            c = cho_factor(H, lower=True)
            Sigma = cho_solve(c, np.eye(d))
            self.alpha_history_.append(alpha)
            if single_pass:
                break
            gamma = n_w - alpha * float(np.diag(Sigma)[w_mask].sum())
            gamma = float(np.clip(gamma, 1e-6, n_w))
            new_alpha = float(
                np.clip(
                    gamma / max(float(theta[w_mask] @ theta[w_mask]), 1e-12), 1e-6, 1e6
                )
            )
            done = abs(np.log(new_alpha) - np.log(alpha)) < self.tol
            alpha = new_alpha
            if done:
                break

        # final posterior at the selected alpha
        a_diag = self._prior_diag(alpha)
        theta = self._map(Phi, Y, a_diag, theta)
        H = self._hessian(theta, Phi, a_diag)
        c = cho_factor(H, lower=True)
        Sigma = cho_solve(c, np.eye(d))
        Sigma = 0.5 * (Sigma + Sigma.T)

        ll = float(
            -(self._neg_log_post(theta, Phi, Y, np.zeros(d))[0])
        )  # log-likelihood at MAP
        logdet_H = 2.0 * float(np.log(np.diag(c[0])).sum())
        self.log_evidence_ = (
            ll
            - 0.5 * float((a_diag * theta * theta).sum())
            + 0.5 * float(np.log(a_diag).sum())
            - 0.5 * logdet_H
        )
        self.alpha_, self.theta_, self.Sigma_ = alpha, theta, Sigma
        self.eff_params_ = float(n_w - alpha * np.diag(Sigma)[w_mask].sum())
        try:
            self.L_ = np.linalg.cholesky(Sigma + 1e-12 * np.eye(d))
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Sigma)
            self.L_ = V * np.sqrt(np.clip(w, 0, None))
        return self

    # ------------------------------------------------------------------
    def predict_proba_map(self, X: np.ndarray) -> np.ndarray:
        """Plug-in prediction with the MAP weights (no posterior uncertainty)."""
        return softmax(self._phi(X) @ self.theta_.reshape(self.K, self.P).T, axis=1)

    def map_weights(self) -> np.ndarray:
        """MAP weights in the original input space, (K, D+1) = [W, bias]."""
        r = self.V_.shape[1]
        th = self.theta_.reshape(self.K, self.P)
        return np.concatenate([th[:, :r] @ self.V_.T, th[:, r:]], axis=1)

    def sample_weights(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """(S, K, D+1) posterior draws of the full weights [W, bias]."""
        rng = np.random.default_rng(seed)
        r = self.V_.shape[1]
        z = rng.standard_normal((n_samples, len(self.theta_)))
        th = (self.theta_[None] + z @ self.L_.T).reshape(n_samples, self.K, self.P)
        G = rng.standard_normal((n_samples, self.K, self.D)) / np.sqrt(self.alpha_)
        G_perp = G - (G @ self.V_) @ self.V_.T  # prior draw outside the data subspace
        W = th[..., :r] @ self.V_.T + G_perp
        return np.concatenate([W, th[..., r:]], axis=-1)

    def predict_samples(
        self, X: np.ndarray, n_samples: int = 200, seed: int = 0
    ) -> np.ndarray:
        """(S, N, K) class probabilities, one row per posterior draw of the weights."""
        W = self.sample_weights(n_samples, seed)
        Zf = np.concatenate([self._z(X), np.ones((len(X), 1))], axis=1)
        return softmax(np.einsum("skp,np->snk", W, Zf), axis=-1)

    def predict_proba(
        self, X: np.ndarray, n_samples: int = 200, seed: int = 0
    ) -> np.ndarray:
        return self.predict_samples(X, n_samples, seed).mean(0)
