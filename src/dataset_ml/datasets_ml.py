"""
PyTorch Dataset classes shared by both model-design branches.

  ImageSampleDataset    -- for IMAGES (CNN)
  CurveSequenceDataset  -- for TABULAR DATA (RNN)

Both read directly from dataset/ml_manifest.csv (written by
build_ml_dataset.py) and filter to rows with a resolved SPM/SD/MD label
and the requested split
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from src import config
except ModuleNotFoundError:
    import config

VALID_LABELS: list = ["SPM", "SD", "MD"]
LABEL_TO_IDX: dict = {label: i for i, label in enumerate(VALID_LABELS)}
IDX_TO_LABEL: dict = {i: label for label, i in LABEL_TO_IDX.items()}


def _load_split(manifest_path: Path, split: str) -> pd.DataFrame:
    """
    Load train, val or test split
    Args:
    - manifest_path (Path): path with all the information about the dataset
    - split (str): train/val/test
    Returns:
    (pd.DataFrame): DataFrame with the requested information
    """

    manifest = pd.read_csv(manifest_path)

    usable = manifest[manifest["label"].isin(VALID_LABELS)]

    subset = usable[usable["split"] == split].reset_index(drop=True)

    if subset.empty:
        raise ValueError(
            f"No usable (labeled) rows for split={split!r} in {manifest_path}. "
            "Did you run build_ml_dataset.py, and does it have that split?"
        )
    return subset


class ImageSampleDataset(Dataset):
    """
    One item = one sample's stacked-channel image tensor
    plus its SPM/SD/MD label.
    """

    def __init__(self, manifest_path: str | Path, split: str):
        """
        Args:
        - manifest_path (str): Path to ml dataset manifest
        - split (str): split to load
        """

        self.rows = _load_split(Path(manifest_path), split)

        if (
            "image_path" not in self.rows.columns
            or self.rows["image_path"].eq("").any()
        ):
            raise ValueError(
                "ml_manifest.csv has rows without an image_path. "
                "Run make_hysteresis_images.py before training the CNN."
            )

    def __len__(self) -> int:
        """
        Lenght of the dataset split
        Returns:
        (int): Lenght
        """
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Get the desired stack of images
        Args:
        - idx (int): Index to load
        Returns:
        (tuple): tensor + label (int)
        """

        row = self.rows.iloc[idx]

        image = np.load(row["image_path"]).astype(np.float32)
        label = LABEL_TO_IDX[row["label"]]

        return torch.from_numpy(image), label


class CurveSequenceDataset(Dataset):
    """
    One item = ALL of a sample's hysteresis curves (one per temperature,
    in config.TEMPERATURES_K order), each as a (seq_len, 3) sequence of
    [Bx_mT (normalized), mx_mean, mx_std] steps, plus the normalized
    temperature for each curve and the sample's SPM/SD/MD label.

    Curves are returned unpadded (as a list of variable-length tensors)
    """

    def __init__(self, manifest_path: str | Path, split: str):
        """
        Args:
        - manifest_path (str): Path to ml dataset manifest
        - split (str): split to load
        """

        self.rows = _load_split(Path(manifest_path), split)

        self.temperatures_in_order = list(config.TEMPERATURES_K)

        self._bx_max_mT = config.Hmax_Oe * 0.1

        self._t_min = min(self.temperatures_in_order)
        self._t_max = max(self.temperatures_in_order)

    def __len__(self) -> int:
        """
        Lenght of the dataset split
        Returns:
        (int): Lenght
        """
        return len(self.rows)

    def _normalize_temperature(self, T: float) -> float:
        """
        Normalize the temperature
        Args:
        - T (float): temperature to normalize
        Returns:
        (float): normalized temperature
        """

        if self._t_max == self._t_min:
            return 0.0

        return (T - self._t_min) / (self._t_max - self._t_min)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the desired hysteresis sample
        Args:
        - idx (int): Index to load
        Returns:
        (tuple): curves + tensor + label (int)
        """

        row = self.rows.iloc[idx]
        hyst_df = pd.read_csv(row["hysteresis_path"])

        curves = []
        temps_norm = []

        for T in self.temperatures_in_order:

            loop = hyst_df[np.isclose(hyst_df["temperature_K"], T)].sort_values("frame")

            bx = loop["Bx_mT"].to_numpy(dtype=np.float32) / self._bx_max_mT
            mx_mean = loop["mx_mean"].to_numpy(dtype=np.float32)
            mx_std = loop.get("mx_std", pd.Series(np.zeros(len(loop)))).to_numpy(
                dtype=np.float32
            )

            seq = np.stack([bx, mx_mean, mx_std], axis=-1)  # (seq_len, 3)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            curves.append(torch.from_numpy(seq))
            temps_norm.append(self._normalize_temperature(T))

        label = LABEL_TO_IDX[row["label"]]

        return curves, torch.tensor(temps_norm, dtype=torch.float32), label


def sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    (N,) lengths -> (N, max_len) float mask, 1.0 for valid (unpadded) steps.
    """

    device = lengths.device
    steps = torch.arange(max_len, device=device).unsqueeze(0)

    return (steps < lengths.unsqueeze(1)).float()


def collate_curve_batches(batch: torch.Tensor) -> tuple:
    """
    Pads every curve in the batch to the batch's longest sequence length
    (per temperature-slot, since different temperatures could in
    principle have different field_steps). Returns:

        padded_curves: (B, n_temperatures, max_len, 3)
        lengths:       (B, n_temperatures)          -- for pack_padded_sequence
        temperatures:  (B, n_temperatures)
        labels:        (B,)
    """

    curves_batch, temps_batch, labels = zip(*batch)
    n_temperatures = len(curves_batch[0])
    batch_size = len(curves_batch)

    max_len = max(curve.shape[0] for curves in curves_batch for curve in curves)
    feature_dim = curves_batch[0][0].shape[-1]

    padded = torch.zeros(
        batch_size, n_temperatures, max_len, feature_dim, dtype=torch.float32
    )
    lengths = torch.zeros(batch_size, n_temperatures, dtype=torch.long)

    for b, curves in enumerate(curves_batch):
        for t, curve in enumerate(curves):
            length = curve.shape[0]
            padded[b, t, :length] = curve
            lengths[b, t] = length

    temperatures = torch.stack(temps_batch, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded, lengths, temperatures, labels


class ImageCurveDataset(CurveSequenceDataset):
    """
    Dataset used by image models that also need the raw M(H) curves.

    One item contains the stacked hysteresis image (for the classifier), the
    variable-length curves (for M-data-check), the normalised temperatures,
    and the class label.  The rows/split are exactly the same as in
    ImageSampleDataset and CurveSequenceDataset.
    """

    def __init__(self, manifest_path: str | Path, split: str):
        super().__init__(manifest_path, split)
        if (
            "image_path" not in self.rows.columns
            or self.rows["image_path"].isna().any()
            or self.rows["image_path"].eq("").any()
        ):
            raise ValueError(
                "ml_manifest.csv has rows without an image_path. "
                "Run make_hysteresis_images.py before training the MLP/CNN."
            )

    def __getitem__(self, idx: int) -> tuple:
        curves, temperatures, label = super().__getitem__(idx)
        image = np.load(self.rows.iloc[idx]["image_path"]).astype(np.float32)
        return torch.from_numpy(image), curves, temperatures, label


def collate_image_curve_batches(batch: list[tuple]) -> tuple:
    """
    Collate ImageCurveDataset. Returns

        images:          (B, C, H, W)
        padded_curves:   (B, n_temperatures, max_len, 3)
        lengths:         (B, n_temperatures)
        temperatures:    (B, n_temperatures)
        labels:          (B,)
    """
    images, curves_batch, temps_batch, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    padded, lengths, temperatures, labels = collate_curve_batches(
        list(zip(curves_batch, temps_batch, labels))
    )
    return images, padded, lengths, temperatures, labels
