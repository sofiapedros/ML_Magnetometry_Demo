"""
Generic, model-agnostic training / prediction utilities used by
train_seeds.py, tune.py and (indirectly) eval_models.py.
"""

from __future__ import annotations

import copy
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from src.models.registry import ModelSpec
except ModuleNotFoundError:
    from registry import ModelSpec

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "ml_manifest.csv"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
HPARAMS_DIR = RESULTS_DIR / "hparams"
CHECKPOINT_DIR = ROOT / "checkpoints"

METRICS = ("accuracy", "macro_f1", "balanced_accuracy")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds, spec: ModelSpec, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=spec.collate_fn,
        generator=generator if shuffle else None,
    )


def load_hparams(spec: ModelSpec, hparams_dir: Path = HPARAMS_DIR) -> dict:
    """Defaults, overridden by the tuned values if tune.py has been run."""
    hp = dict(spec.default_hparams)
    path = hparams_dir / f"{spec.name}.json"
    if path.is_file():
        hp.update(json.loads(path.read_text())["best_params"])
    return hp


@torch.no_grad()
def predict(spec: ModelSpec, model: nn.Module, loader, device) -> tuple:
    """Returns (probs (N, C) numpy, labels (N,) numpy)."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logits, labels = spec.forward(model, batch, device)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)


def _enable_dropout(model: nn.Module) -> int:
    """
    Put ONLY the dropout layers in train mode (BatchNorm keeps its running
    statistics). Returns how many dropout layers with p > 0 are active.
    """
    active = 0
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            m.train()
            active += int(m.p > 0)
    return active


@torch.no_grad()
def predict_mc(
    spec: ModelSpec, model: nn.Module, loader, device, n_samples: int = 30
) -> tuple:
    """
    Monte-Carlo dropout (n_samples stochastic forward passes with dropout on), or, for
    Bayesian models, n_samples forward passes each with one posterior draw of the weights.
    Returns (probs (S, N, C) numpy, labels (N,) numpy).
    """
    model.eval()
    # Bayesian models: one posterior draw per pass. Otherwise: MC dropout.
    posterior = hasattr(model, "resample_posterior")
    if posterior:
        model.mc_mode = True
    elif _enable_dropout(model) == 0:
        warnings.warn(
            "predict_mc: the model has no active dropout (p > 0); "
            "all MC samples will be identical."
        )
    batches = list(loader)  # load the data once, reuse it for every MC pass
    samples, labels_np = [], None
    try:
        for _ in range(n_samples):
            if posterior:
                model.resample_posterior()
            probs, labs = [], []
            for batch in batches:
                logits, labels = spec.forward(model, batch, device)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                labs.append(labels.cpu().numpy())
            samples.append(np.concatenate(probs))
            if labels_np is None:
                labels_np = np.concatenate(labs)
    finally:
        if posterior:
            model.mc_mode = False
        model.eval()
    return np.stack(samples), labels_np


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


def fit(
    spec: ModelSpec,
    hp: dict,
    train_loader: DataLoader,
    device,
    *,
    epochs: int,
    val_loader: DataLoader | None = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Train one model.
    - With val_loader: the epoch with best val accuracy (ties: lower val
      loss) is restored at the end (same criterion as train_cnn/train_rnn).
    - Without val_loader: the last epoch is returned (used inside CV so
      the held-out fold is never used for model selection).
    """
    model = spec.build(hp).to(device)
    if spec.pretrain is not None:
        spec.pretrain(model, train_loader, hp, device)

    # Closed-form models (e.g. Bayesian logistic regression): no gradient loop.
    if spec.fit_fn is not None:
        spec.fit_fn(model, train_loader, hp, device)
        model.eval()
        return model

    # Optional auxiliary loss (e.g. the physics-informed M-data-check term).
    # prepare_loss only sees the training loader, so val / test / held-out
    # CV folds never leak into it.
    aux = None
    if spec.prepare_loss is not None:
        aux = spec.prepare_loss(train_loader, hp, device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.0)
    )
    criterion = nn.CrossEntropyLoss()

    best_state, best_key = None, (-1.0, float("inf"))

    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits, labels = spec.forward(model, batch, device)
            if spec.loss_fn is not None:
                loss = spec.loss_fn(logits, labels, batch, device, hp, aux, epoch)
            else:
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(labels)
            n += len(labels)

        if val_loader is not None:
            probs, labels_np = predict(spec, model, val_loader, device)
            val_acc = float((probs.argmax(1) == labels_np).mean())
            val_loss = float(
                -np.log(probs[np.arange(len(labels_np)), labels_np] + 1e-12).mean()
            )
            if val_acc > best_key[0] or (
                val_acc == best_key[0] and val_loss < best_key[1]
            ):
                best_key = (val_acc, val_loss)
                best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(
                    f"  epoch {epoch:3d}/{epochs} | train loss {running / n:.4f} "
                    f"| val loss {val_loss:.4f} acc {val_acc:.3f}"
                )
        elif verbose:
            print(f"  epoch {epoch:3d}/{epochs} | train loss {running / n:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model
