"""
Train a RNN + classifier: Extract features from each curve with an
RNN, then classify from the concatenated feature vectors with a small NN.

Two training modes

  --mode joint     End-to-end: gradients from the classification loss
                   flow all the way back through the shared RNN encoder.

  --mode separate  Two stages:
                     1. Pretrain the RNN UNSUPERVISED as a curve
                        autoencoder (reconstruct mx_mean from Bx_mT),
                        so it learns general loop-shape features without
                        seeing any label.
                     2. Freeze the RNN and train only the NN head on the
                        resulting fixed feature vectors.

Prerequisite:
    python -m src.dataset_ml.build_ml_dataset     (make_hysteresis_images.py is NOT
                                        needed for this branch -- it
                                        consumes the per-temperature CSVs
                                        directly, not images)

Usage:
    python -m src.models.train_rnn--mode joint
    python -m src.models.train_rnn --mode separate --pretrain-epochs 20 --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from src import config
except ModuleNotFoundError:
    import config

try:
    from src.dataset_ml.datasets_ml import (
        IDX_TO_LABEL,
        VALID_LABELS,
        CurveSequenceDataset,
        collate_curve_batches,
    )
    from src.models.basic_models import CurveAutoencoder, CurveRNNEncoder, RNNClassifier
except ModuleNotFoundError:
    from dataset_ml.datasets_ml import (
        IDX_TO_LABEL,
        VALID_LABELS,
        CurveSequenceDataset,
        collate_curve_batches,
    )
    from src.models.basic_models import CurveAutoencoder, CurveRNNEncoder, RNNClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ROOT / "dataset"
MANIFEST_PATH = DATASET_DIR / "ml_manifest.csv"
CHECKPOINT_PATH = ROOT / "checkpoints" / "idea1_rnn_nn.pt"


def pretrain_autoencoder(
    encoder: CurveRNNEncoder,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Stage 1 of --mode separate: unsupervised, ignores labels entirely."""
    autoencoder = CurveAutoencoder(encoder).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        running_loss, running_n = 0.0, 0

        for padded_curves, lengths, _temperatures, _labels in loader:
            padded_curves, lengths = padded_curves.to(device), lengths.to(device)
            batch_size, n_temperatures, max_len, input_dim = padded_curves.shape

            flat_curves = padded_curves.reshape(
                batch_size * n_temperatures, max_len, input_dim
            )
            flat_lengths = lengths.reshape(batch_size * n_temperatures)

            target = flat_curves[..., 1]  # mx_mean channel
            mask = (
                torch.arange(max_len, device=device).unsqueeze(0)
                < flat_lengths.unsqueeze(1)
            ).float()

            optimizer.zero_grad()
            reconstruction = autoencoder(flat_curves, flat_lengths)
            loss = (criterion(reconstruction, target) * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_size
            running_n += batch_size

        print(
            f"  [pretrain autoencoder] epoch {epoch:3d}/{epochs} | reconstruction MSE {running_loss / running_n:.5f}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    confusion = torch.zeros(len(VALID_LABELS), len(VALID_LABELS), dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    for padded_curves, lengths, temperatures, labels in loader:
        padded_curves, lengths = padded_curves.to(device), lengths.to(device)
        temperatures, labels = temperatures.to(device), labels.to(device)

        logits = model(padded_curves, lengths, temperatures)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_n += len(labels)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t, p] += 1

    return total_loss / total_n, total_correct / total_n, confusion


def train_classifier(
    model: RNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    device: torch.device,
    trainable_params,
) -> float:
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0

        for padded_curves, lengths, temperatures, labels in train_loader:
            padded_curves, lengths = padded_curves.to(device), lengths.to(device)
            temperatures, labels = temperatures.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(padded_curves, lengths, temperatures)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * len(labels)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_n += len(labels)

        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        val_loss, val_acc, _ = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:3d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "val_acc": val_acc}, CHECKPOINT_PATH
            )

    return best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", choices=["joint", "separate"], default="joint")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--epochs", type=int, default=30, help="Classifier training epochs."
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=20,
        help="Autoencoder epochs, --mode separate only.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    train_ds = CurveSequenceDataset(MANIFEST_PATH, split="train")
    val_ds = CurveSequenceDataset(MANIFEST_PATH, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_curve_batches,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_curve_batches,
    )

    n_temperatures = len(config.TEMPERATURES_K)
    encoder = CurveRNNEncoder(input_dim=3, hidden_dim=args.hidden_dim).to(device)
    model = RNNClassifier(
        encoder, n_temperatures=n_temperatures, num_classes=len(VALID_LABELS)
    ).to(device)

    if args.mode == "separate":
        print("Stage 1/2: unsupervised RNN pretraining (autoencoder) ...")
        pretrain_autoencoder(
            encoder,
            train_loader,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            device=device,
        )

        print("\nStage 2/2: training NN head on frozen RNN features ...")
        for param in encoder.parameters():
            param.requires_grad_(False)
        trainable_params = model.head.parameters()
    else:
        print("Joint end-to-end training (RNN + NN together) ...")
        trainable_params = model.parameters()

    best_val_acc = train_classifier(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        trainable_params=trainable_params,
    )
    print(f"\nBest val accuracy: {best_val_acc:.3f}. Checkpoint: {CHECKPOINT_PATH}")

    import pandas as pd

    manifest = pd.read_csv(MANIFEST_PATH)
    if (manifest["split"] == "test").any():
        test_ds = CurveSequenceDataset(MANIFEST_PATH, split="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_curve_batches,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_acc, confusion = evaluate(model, test_loader, device)
        print(f"Test loss {test_loss:.4f} acc {test_acc:.3f}")
        print(f"Confusion matrix (rows=true, cols=pred), order={VALID_LABELS}:")
        print(confusion)


if __name__ == "__main__":
    main()
