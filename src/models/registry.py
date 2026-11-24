"""
Model registry: the single place where a model is described.

Every script that trains / tunes / evaluates models (train_seeds.py,
tune.py, eval_models.py) only talks to this registry, so adding a new
model means adding ONE `register(ModelSpec(...))` call at the bottom of
this file (or in your own module that imports `register`).

A ModelSpec tells the generic pipeline:
  - which Dataset (+ collate_fn) feeds the model,
  - how to build the nn.Module from a hyperparameter dict,
  - how to run a forward pass on ONE batch -> (logits, labels),
  - the default hyperparameters,
  - the Optuna search space (`suggest`),
  - an optional `pretrain` hook (e.g. unsupervised RNN pretraining that
    also freezes part of the model; frozen params are skipped by the
    optimizer automatically).

Example of adding a new model (e.g. a transformer over the curves):

    register(ModelSpec(
        name="transformer",
        dataset_cls=CurveSequenceDataset,
        collate_fn=collate_curve_batches,
        build=lambda hp: MyTransformer(**hp),
        forward=rnn_forward,               # same batch format as the RNN
        default_hparams={"lr": 1e-3, "batch_size": 8, "d_model": 64},
        suggest=lambda trial: {...},
    ))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

try:
    from src import config
except ModuleNotFoundError:
    import config

try:
    from src.dataset_ml.datasets_ml import (
        VALID_LABELS,
        CurveSequenceDataset,
        ImageCurveDataset,
        ImageSampleDataset,
        collate_curve_batches,
        collate_image_curve_batches,
    )
    from src.models.basic_models import (
        CNNClassifier,
        CurveRNNEncoder,
        MLPClassifier,
        RNNClassifier,
    )
    from src.models.bayes_model import (
        BayesLogRegModel,
        bayes_curve_forward,
        bayes_forward,
        bayes_image_forward,
        fit_bayes_logreg,
    )
    from src.models.m_data_check import m_data_check_loss, prepare_m_data_check
    from src.models.preisach import (
        PreisachConfig,
        PreisachCurveDataset,
        PreisachFeatureDataset,
        PreisachImageNet,
        PreisachMLP,
        PreisachRNN,
        PreisachSetClassifier,
        collate_preisach_curve_batches,
        fit_feature_scaler,
    )
    from src.models.preisach import n_features as n_preisach_features
    from src.models.preisach import preisach_curve_forward, preisach_forward
    from src.models.train_rnn import pretrain_autoencoder
except ModuleNotFoundError:
    from basic_models import (
        CNNClassifier,
        CurveRNNEncoder,
        MLPClassifier,
        RNNClassifier,
    )
    from bayes_model import (
        BayesLogRegModel,
        bayes_curve_forward,
        bayes_forward,
        bayes_image_forward,
        fit_bayes_logreg,
    )
    from m_data_check import m_data_check_loss, prepare_m_data_check
    from preisach import (
        PreisachConfig,
        PreisachCurveDataset,
        PreisachFeatureDataset,
        PreisachImageNet,
        PreisachMLP,
        PreisachRNN,
        PreisachSetClassifier,
        collate_preisach_curve_batches,
        fit_feature_scaler,
    )
    from preisach import n_features as n_preisach_features
    from preisach import preisach_curve_forward, preisach_forward
    from train_rnn import pretrain_autoencoder

    from dataset_ml.datasets_ml import (
        VALID_LABELS,
        CurveSequenceDataset,
        ImageCurveDataset,
        ImageSampleDataset,
        collate_curve_batches,
        collate_image_curve_batches,
    )


@dataclass
class ModelSpec:
    name: str
    dataset_cls: Callable
    build: Callable[[dict], nn.Module]
    forward: Callable  # (model, batch, device) -> (logits, labels)
    default_hparams: dict
    suggest: Callable  # (optuna.Trial) -> dict of hyperparameters
    collate_fn: Optional[Callable] = None
    pretrain: Optional[Callable] = None  # (model, train_loader, hp, device) -> None
    prepare_loss: Optional[Callable] = None  # (train_loader, hp, device) -> aux
    loss_fn: Optional[Callable] = (
        None  # (logits, labels, batch, device, hp, aux, epoch) -> loss
    )
    # Closed-form models: replaces the gradient loop. (model, train_loader, hp, device) -> None
    fit_fn: Optional[Callable] = None
    # Posterior draws stored by train_seeds.py when --mc-samples is not given (0 = none)
    mc_default: int = 0


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"Model {spec.name!r} already registered.")
    REGISTRY[spec.name] = spec
    return spec


def get_spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


# --------------------------------------------------------------------------
# Forward functions (one per batch format)
# --------------------------------------------------------------------------
def image_forward(model, batch, device):
    images, labels = batch
    return model(images.to(device)), labels.to(device)


def image_curve_forward(model, batch, device):
    images, _padded, _lengths, _temperatures, labels = batch
    return model(images.to(device)), labels.to(device)


def curve_forward(model, batch, device):
    padded, lengths, temperatures, labels = batch
    logits = model(padded.to(device), lengths.to(device), temperatures.to(device))
    return logits, labels.to(device)


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
def _build_cnn(hp: dict) -> nn.Module:
    return CNNClassifier(
        in_channels=len(config.TEMPERATURES_K),
        num_classes=len(VALID_LABELS),
        base_channels=hp["base_channels"],
    )


def _suggest_cnn(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "base_channels": trial.suggest_categorical("base_channels", [8, 16, 32]),
    }


register(
    ModelSpec(
        name="cnn",
        dataset_cls=ImageSampleDataset,
        build=_build_cnn,
        forward=image_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "base_channels": 16,
        },
        suggest=_suggest_cnn,
    )
)


# --------------------------------------------------------------------------
# MLP baseline over the same stacked images as the CNN
# --------------------------------------------------------------------------
def _build_mlp(hp: dict) -> nn.Module:
    return MLPClassifier(
        in_channels=len(config.TEMPERATURES_K),
        num_classes=len(VALID_LABELS),
        pool_size=hp["pool_size"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    )


def _suggest_mlp(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "pool_size": trial.suggest_categorical("pool_size", [8, 16, 32]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
    }


register(
    ModelSpec(
        name="mlp",
        dataset_cls=ImageSampleDataset,
        build=_build_mlp,
        forward=image_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "pool_size": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
        suggest=_suggest_mlp,
    )
)


# --------------------------------------------------------------------------
# RNN (joint and separate = unsupervised pretraining + frozen encoder)
# --------------------------------------------------------------------------
def _build_rnn(hp: dict) -> nn.Module:
    encoder = CurveRNNEncoder(
        input_dim=3, hidden_dim=hp["hidden_dim"], num_layers=hp["num_layers"]
    )
    return RNNClassifier(
        encoder,
        n_temperatures=len(config.TEMPERATURES_K),
        num_classes=len(VALID_LABELS),
    )


def _suggest_rnn(trial, separate: bool) -> dict:
    hp = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
    }
    if separate:
        hp["pretrain_epochs"] = trial.suggest_categorical(
            "pretrain_epochs", [10, 20, 40]
        )
        hp["pretrain_lr"] = trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True)
    return hp


def _pretrain_and_freeze(model: RNNClassifier, loader, hp: dict, device) -> None:
    pretrain_autoencoder(
        model.encoder,
        loader,
        epochs=hp["pretrain_epochs"],
        lr=hp["pretrain_lr"],
        device=device,
    )
    for p in model.encoder.parameters():
        p.requires_grad_(False)


_RNN_DEFAULTS = {
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 8,
    "hidden_dim": 64,
    "num_layers": 1,
}

register(
    ModelSpec(
        name="rnn_joint",
        dataset_cls=CurveSequenceDataset,
        collate_fn=collate_curve_batches,
        build=_build_rnn,
        forward=curve_forward,
        default_hparams=dict(_RNN_DEFAULTS),
        suggest=lambda trial: _suggest_rnn(trial, separate=False),
    )
)

register(
    ModelSpec(
        name="rnn_separate",
        dataset_cls=CurveSequenceDataset,
        collate_fn=collate_curve_batches,
        build=_build_rnn,
        forward=curve_forward,
        default_hparams={**_RNN_DEFAULTS, "pretrain_epochs": 20, "pretrain_lr": 1e-3},
        suggest=lambda trial: _suggest_rnn(trial, separate=True),
        pretrain=_pretrain_and_freeze,
    )
)


# --------------------------------------------------------------------------
# RNN + M-data-check physics-informed loss
# --------------------------------------------------------------------------
def _prepare_m_data_check(train_loader, hp: dict, device):
    return prepare_m_data_check(
        train_loader,
        device,
        bx_max_mT=float(config.Hmax_Oe) * 0.1,
        prior_k=hp["prior_k"],
    )


def _m_data_check_rnn_loss(logits, labels, batch, device, hp: dict, aux, epoch: int):
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, labels)
    padded, lengths, _temperatures, _labels = batch
    feats, _valid, gate, _checks = aux["check"](padded.to(device), lengths.to(device))
    probs = torch.softmax(logits, dim=-1)
    phys = m_data_check_loss(probs, feats, gate, aux["priors"])

    warmup = max(1, int(hp.get("phys_warmup_epochs", 5)))
    ramp = min(1.0, epoch / warmup)
    return ce + hp["lambda_phys"] * ramp * phys


def _suggest_m_data_check_rnn(trial) -> dict:
    hp = _suggest_rnn(trial, separate=False)
    hp.update(
        {
            "lambda_phys": trial.suggest_float("lambda_phys", 0.02, 2.0, log=True),
            "prior_k": trial.suggest_float("prior_k", 1.0, 2.5, step=0.25),
        }
    )
    return hp


register(
    ModelSpec(
        name="m_data_check_RNN",
        dataset_cls=CurveSequenceDataset,
        collate_fn=collate_curve_batches,
        build=_build_rnn,
        forward=curve_forward,
        default_hparams={
            **_RNN_DEFAULTS,
            "lambda_phys": 0.5,
            "prior_k": 1.5,
            "phys_warmup_epochs": 5,
        },
        suggest=_suggest_m_data_check_rnn,
        prepare_loss=_prepare_m_data_check,
        loss_fn=_m_data_check_rnn_loss,
    )
)


# --------------------------------------------------------------------------
# MLP + M-data-check physics-informed loss
# --------------------------------------------------------------------------
def _m_data_check_mlp_loss(logits, labels, batch, device, hp: dict, aux, epoch: int):
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, labels)
    _images, padded, lengths, _temperatures, _labels = batch
    feats, _valid, gate, _checks = aux["check"](padded.to(device), lengths.to(device))
    probs = torch.softmax(logits, dim=-1)
    phys = m_data_check_loss(probs, feats, gate, aux["priors"])

    warmup = max(1, int(hp.get("phys_warmup_epochs", 5)))
    ramp = min(1.0, epoch / warmup)
    return ce + hp["lambda_phys"] * ramp * phys


def _suggest_m_data_check_mlp(trial) -> dict:
    hp = _suggest_mlp(trial)
    hp.update(
        {
            "lambda_phys": trial.suggest_float("lambda_phys", 0.02, 2.0, log=True),
            "prior_k": trial.suggest_float("prior_k", 1.0, 2.5, step=0.25),
        }
    )
    return hp


register(
    ModelSpec(
        name="m_data_check_MLP",
        dataset_cls=ImageCurveDataset,
        collate_fn=collate_image_curve_batches,
        build=_build_mlp,
        forward=image_curve_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "pool_size": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "lambda_phys": 0.5,
            "prior_k": 1.5,
            "phys_warmup_epochs": 5,
        },
        suggest=_suggest_m_data_check_mlp,
        prepare_loss=_prepare_m_data_check,
        loss_fn=_m_data_check_mlp_loss,
    )
)


# --------------------------------------------------------------------------
# Preisach input: every M(H) loop is replaced by a SUM OF HYSTERONS (see
# preisach.py) and the classifier sees the hysteron weights instead of the curve.
#   preisach_mlp : flat MLP over the weights of all temperatures
#   preisach_set : shared per-temperature encoder (+T) -> head, the analogue of
#                  the RNN pipeline with an MLP encoder on the weights
# The input scaling is fitted on the training loader only (pretrain hook), so it
# is also leakage-free inside the CV folds of tune.py.
# --------------------------------------------------------------------------
def _build_preisach_mlp(hp: dict) -> nn.Module:
    return PreisachMLP(
        n_temperatures=len(config.TEMPERATURES_K),
        n_features=n_preisach_features(),
        num_classes=len(VALID_LABELS),
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    )


def _build_preisach_set(hp: dict) -> nn.Module:
    return PreisachSetClassifier(
        n_temperatures=len(config.TEMPERATURES_K),
        n_features=n_preisach_features(),
        num_classes=len(VALID_LABELS),
        hidden_dim=hp["hidden_dim"],
        dropout=hp["dropout"],
    )


def _suggest_preisach_mlp(trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
    }


def _suggest_preisach_set(trial) -> dict:
    # same space as the MLP without num_layers (it must not be suggested at all,
    # otherwise Optuna still records it as a parameter of the trial)
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
    }


register(
    ModelSpec(
        name="preisach_mlp",
        dataset_cls=PreisachFeatureDataset,
        build=_build_preisach_mlp,
        forward=preisach_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
        },
        suggest=_suggest_preisach_mlp,
        pretrain=fit_feature_scaler,
    )
)

register(
    ModelSpec(
        name="preisach_set",
        dataset_cls=PreisachFeatureDataset,
        build=_build_preisach_set,
        forward=preisach_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "hidden_dim": 64,
            "dropout": 0.2,
        },
        suggest=_suggest_preisach_set,
        pretrain=fit_feature_scaler,
    )
)


# --------------------------------------------------------------------------
# The ORIGINAL architectures with the Preisach plane as input (same classes from
# basic_models.py, only the input changes; temperatures are the channels):
#   preisach_cnn            <- cnn            CNNClassifier over the plane image
#   preisach_rnn            <- rnn_joint      GRU over the plane (column by column) + (+T) head
#   m_data_check_preisach_MLP <- m_data_check_MLP   preisach_mlp + physics loss on the raw curves
#   m_data_check_preisach_RNN <- m_data_check_RNN   preisach_rnn + physics loss on the raw curves
# (rnn_separate has no counterpart: its autoencoder pretraining reconstructs the raw
#  M(H) sequence, which does not exist for the plane.)
# --------------------------------------------------------------------------
_N_NODES = PreisachConfig().n_nodes


def _build_preisach_cnn(hp: dict) -> nn.Module:
    return PreisachImageNet(
        CNNClassifier(
            in_channels=len(config.TEMPERATURES_K),
            num_classes=len(VALID_LABELS),
            base_channels=hp["base_channels"],
        ),
        _N_NODES,
    )


def _build_preisach_rnn(hp: dict) -> nn.Module:
    return PreisachRNN(
        n_temperatures=len(config.TEMPERATURES_K),
        n_nodes=_N_NODES,
        num_classes=len(VALID_LABELS),
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
    )


register(
    ModelSpec(
        name="preisach_cnn",
        dataset_cls=PreisachFeatureDataset,
        build=_build_preisach_cnn,
        forward=preisach_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 8,
            "base_channels": 16,
        },
        suggest=_suggest_cnn,
        pretrain=fit_feature_scaler,
    )
)

register(
    ModelSpec(
        name="preisach_rnn",
        dataset_cls=PreisachFeatureDataset,
        build=_build_preisach_rnn,
        forward=preisach_forward,
        default_hparams=dict(_RNN_DEFAULTS),
        suggest=lambda trial: _suggest_rnn(trial, separate=False),
        pretrain=fit_feature_scaler,
    )
)

_PHYS_DEFAULTS = {"lambda_phys": 0.5, "prior_k": 1.5, "phys_warmup_epochs": 5}


def _suggest_phys(hp: dict, trial) -> dict:
    hp.update(
        {
            "lambda_phys": trial.suggest_float("lambda_phys", 0.02, 2.0, log=True),
            "prior_k": trial.suggest_float("prior_k", 1.0, 2.5, step=0.25),
        }
    )
    return hp


register(
    ModelSpec(
        name="m_data_check_preisach_MLP",
        dataset_cls=PreisachCurveDataset,
        collate_fn=collate_preisach_curve_batches,
        build=_build_preisach_mlp,
        forward=preisach_curve_forward,
        default_hparams={
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            **_PHYS_DEFAULTS,
        },
        suggest=lambda trial: _suggest_phys(_suggest_preisach_mlp(trial), trial),
        pretrain=fit_feature_scaler,
        prepare_loss=_prepare_m_data_check,
        loss_fn=_m_data_check_mlp_loss,  # same 5-field batch: (weights, padded, lengths, T, labels)
    )
)

register(
    ModelSpec(
        name="m_data_check_preisach_RNN",
        dataset_cls=PreisachCurveDataset,
        collate_fn=collate_preisach_curve_batches,
        build=_build_preisach_rnn,
        forward=preisach_curve_forward,
        default_hparams={**_RNN_DEFAULTS, **_PHYS_DEFAULTS},
        suggest=lambda trial: _suggest_phys(_suggest_rnn(trial, separate=False), trial),
        pretrain=fit_feature_scaler,
        prepare_loss=_prepare_m_data_check,
        loss_fn=_m_data_check_mlp_loss,
    )
)


# --------------------------------------------------------------------------
# Bayesian logistic regression (bayes_logreg.py / bayes_model.py): purely
# probabilistic baselines. The posterior over the weights is computed in closed
# form (Laplace) in `fit_fn`; predictions are posterior draws, so every
# evaluation gets epistemic uncertainty. Same samples, only the input changes:
#   bayes_mlp      : exactly the input of `mlp` (stacked hysteresis images, same pooling), no network
#   bayes_preisach : hysteron weights of all temperatures
#   bayes_curve    : the raw curves of rnn_joint / rnn_separate (same dataset), flattened
# alpha (prior precision) is chosen by marginal likelihood on the training set
# unless tune.py finds that a fixed value cross-validates better.
# --------------------------------------------------------------------------
def _build_bayes_preisach(hp: dict) -> nn.Module:
    return BayesLogRegModel(
        in_features=len(config.TEMPERATURES_K) * n_preisach_features(),
        n_classes=len(VALID_LABELS),
        n_post=hp["n_post"],
    )


def _build_bayes_mlp(hp: dict) -> nn.Module:
    return BayesLogRegModel(
        in_features=len(config.TEMPERATURES_K) * hp["pool_size"] ** 2,
        n_classes=len(VALID_LABELS),
        n_post=hp["n_post"],
        pool_size=hp["pool_size"],
    )


def _suggest_bayes_mlp(trial) -> dict:
    return {
        "pool_size": trial.suggest_categorical("pool_size", [8, 16, 32]),
        **_suggest_bayes(trial),
    }


def _suggest_bayes(trial) -> dict:
    mode = trial.suggest_categorical("alpha_mode", ["evidence", "fixed"])
    hp = {"alpha_mode": mode}
    if mode == "fixed":
        hp["alpha"] = trial.suggest_float("alpha", 1e-2, 1e3, log=True)
    return hp


_BAYES_HP = {"batch_size": 64, "alpha_mode": "evidence", "alpha": 1.0, "n_post": 200}

register(
    ModelSpec(
        name="bayes_preisach",
        dataset_cls=PreisachFeatureDataset,
        build=_build_bayes_preisach,
        forward=bayes_forward,
        default_hparams=dict(_BAYES_HP),
        suggest=_suggest_bayes,
        fit_fn=fit_bayes_logreg,
        mc_default=200,
    )
)

# same input as `mlp` (stacked hysteresis images, average-pooled like MLPClassifier)
register(
    ModelSpec(
        name="bayes_mlp",
        dataset_cls=ImageSampleDataset,
        build=_build_bayes_mlp,
        forward=bayes_image_forward,
        default_hparams={**_BAYES_HP, "pool_size": 16},
        suggest=_suggest_bayes_mlp,
        fit_fn=fit_bayes_logreg,
        mc_default=200,
    )
)

# same input as the RNNs: the raw padded curves (B, n_T, L, 3), flattened (all curves must
# have the same number of points; the input size is read from the training data)
register(
    ModelSpec(
        name="bayes_curve",
        dataset_cls=CurveSequenceDataset,
        collate_fn=collate_curve_batches,
        build=lambda hp: BayesLogRegModel(
            n_classes=len(VALID_LABELS), n_post=hp["n_post"]
        ),
        forward=bayes_curve_forward,
        default_hparams=dict(_BAYES_HP),
        suggest=_suggest_bayes,
        fit_fn=fit_bayes_logreg,
        mc_default=200,
    )
)
