"""
Model definitions for the two MODEL_DESIGN branches this round covers.

  CNNClassifier         -- IMAGES -> CNN branch
  MLPClassifier         -- IMAGES -> pooled pixels -> MLP baseline
  CurveRNNEncoder       -- encode one curve -> one feature vector
  CurveAutoencoder      -- lets CurveRNNEncoder be pretrained UNSUPERVISED
  RNNClassifier         -- shared RNN over each temperature's curve ->
                            concatenated feature vectors -> NN -> class logits
"""

from __future__ import annotations

import torch
from torch import nn


class CNNClassifier(nn.Module):
    """
    Small conv stack over the stacked-channel hysteresis image.
    Ends in a global-average-pool so it doesn't hardcode the image size.
    """

    def __init__(self, in_channels: int, num_classes: int = 3, base_channels: int = 16):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c, 2 * c, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * c, 8 * c, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * c),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * c, 4 * c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4 * c, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """
    MLP baseline over the same stacked hysteresis images used by the CNN.

    A non-trainable adaptive average pool first maps any input image size to
    a fixed grid, then the pooled pixels from all temperature channels are
    flattened and passed through a configurable fully-connected network.
    This keeps the model independent of the raster size used by
    make_hysteresis_images.py while remaining a pure dense-network baseline.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        pool_size: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        input_dim = in_channels * pool_size * pool_size

        layers: list[nn.Module] = [nn.Flatten()]
        dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.pool(x)
        return self.classifier(x)


class CurveRNNEncoder(nn.Module):
    """
    One (Bx, mx_mean, mx_std) sequence -> one feature vector.
    A single GRU is shared across every temperature's curve (weight
    sharing = the model doesn't need separate parameters per
    temperature to recognize "this looks like an SD loop").
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        sequences: (N, max_len, input_dim)
        lengths:   (N,) actual (unpadded) length of each sequence
        returns:   (N, hidden_dim) -- the final hidden state of each sequence
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences,
            lengths.clamp(min=1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return h_n[-1]  # (N, hidden_dim), last layer's final hidden state


class CurveAutoencoder(nn.Module):
    """
    Unsupervised pretraining target for CurveRNNEncoder: encode a curve
    to a feature vector, then reconstruct the mx_mean trace from it and
    the known Bx_mT trace. Used only by train_rnn.py --mode separate;
    the classifier never sees this decoder.
    """

    def __init__(self, encoder: CurveRNNEncoder, input_dim: int = 3):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.hidden_dim
        self.decoder_cell = nn.GRUCell(1, hidden_dim)  # driven by Bx at each step
        self.to_mx = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed mx_mean, shape (N, max_len)."""
        feature = self.encoder(sequences, lengths)  # (N, hidden_dim)
        bx = sequences[
            ..., 0
        ]  # (N, max_len) -- reconstruction is conditioned on the known field path
        n, max_len = bx.shape

        h = feature
        outputs = []
        for t in range(max_len):
            h = self.decoder_cell(bx[:, t : t + 1], h)
            outputs.append(self.to_mx(h))
        return torch.cat(outputs, dim=1)  # (N, max_len)


class RNNClassifier(nn.Module):
    """
    Shared RNN over each temperature's curve ->
    concatenate the resulting feature vectors (+ each curve's
    temperature, appended so the NN head knows which curve is which) ->
    small NN -> class logits.
    """

    def __init__(
        self, encoder: CurveRNNEncoder, n_temperatures: int, num_classes: int = 3
    ):
        super().__init__()
        self.encoder = encoder
        hidden_dim = encoder.hidden_dim
        head_input_dim = n_temperatures * (
            hidden_dim + 1
        )  # +1 per curve for its normalized temperature
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 4 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4 * hidden_dim, num_classes),
        )

    def forward(
        self,
        padded_curves: torch.Tensor,
        lengths: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        """
        padded_curves: (B, n_temperatures, max_len, input_dim)
        lengths:       (B, n_temperatures)
        temperatures:  (B, n_temperatures) normalized in [0, 1]
        """
        batch_size, n_temperatures, max_len, input_dim = padded_curves.shape

        flat_curves = padded_curves.reshape(
            batch_size * n_temperatures, max_len, input_dim
        )
        flat_lengths = lengths.reshape(batch_size * n_temperatures)
        features = self.encoder(
            flat_curves, flat_lengths
        )  # (B * n_temperatures, hidden_dim)
        features = features.reshape(batch_size, n_temperatures, -1)

        features_with_T = torch.cat([features, temperatures.unsqueeze(-1)], dim=-1)
        flattened = features_with_T.reshape(batch_size, -1)
        return self.head(flattened)
