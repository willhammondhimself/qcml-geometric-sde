"""
Deep Learning Baseline Regime Detectors

Provides supervised deep learning baselines for regime detection comparison:
1. LSTMRegimeDetector — 2-layer LSTM with dropout → Linear → Sigmoid
2. TCNRegimeDetector — 3-layer dilated causal Conv1d → BatchNorm → ReLU → Linear → Sigmoid

Both implement the BaseRegimeDetector interface (fit_with_labels / compute_regime_scores)
and follow the same training protocol as RandomForestRegimeDetector.

Author: QCML Research
"""

import logging
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from qcml.regime.classical_baselines import BaseRegimeDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sequence dataset builder
# ---------------------------------------------------------------------------

def _build_sequences(
    X: np.ndarray, y: np.ndarray, seq_len: int = 20
) -> tuple:
    """Build (sequences, labels) for supervised sequence models.

    Args:
        X: Feature matrix (T, d).
        y: Binary labels (T,).
        seq_len: Sequence length for each sample.

    Returns:
        X_seq: (N, seq_len, d) array.
        y_seq: (N,) array — label at the last timestep of each window.
    """
    T, d = X.shape
    N = T - seq_len + 1
    X_seq = np.empty((N, seq_len, d), dtype=np.float32)
    y_seq = np.empty(N, dtype=np.float32)
    for i in range(N):
        X_seq[i] = X[i:i + seq_len]
        y_seq[i] = y[i + seq_len - 1]
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# 1. LSTM Regime Detector
# ---------------------------------------------------------------------------

class _LSTMModel(nn.Module):
    """2-layer LSTM → dropout → linear → sigmoid."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out)).squeeze(-1)


class LSTMRegimeDetector(BaseRegimeDetector):
    """LSTM-based supervised regime detector.

    Uses a 2-layer LSTM (hidden_dim=64, dropout=0.3) trained with
    BCELoss and Adam optimizer.  Training uses early stopping with
    patience=10.

    Args:
        hidden_dim: LSTM hidden dimension.
        seq_len: Sequence length for input windows.
        n_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        seed: Random seed.
        lookback: Rolling feature window (matches RF).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        seq_len: int = 20,
        n_epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        seed: int = 42,
        lookback: int = 20,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LSTMRegimeDetector")
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.lookback = lookback
        self._model = None
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "LSTM"

    def fit(self, X: np.ndarray, **kwargs) -> 'LSTMRegimeDetector':
        return self

    def fit_with_labels(
        self, X: np.ndarray, y: np.ndarray
    ) -> 'LSTMRegimeDetector':
        """Train LSTM on labeled feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0=normal, 1=crisis).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Build rolling features (same as RF)
        X_feat = self._build_ml_features(X)
        y_trimmed = y[self.lookback - 1:]

        # Build sequences
        X_seq, y_seq = _build_sequences(X_feat, y_trimmed, self.seq_len)

        # Create model
        input_dim = X_feat.shape[1]
        self._model = _LSTMModel(input_dim, self.hidden_dim).to(self._device)

        # Training setup
        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        # Training with early stopping
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"LSTM early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit_with_labels() before compute_regime_scores().")

        self._model.eval()
        X_feat = self._build_ml_features(X)

        # Build sequences (no labels needed at inference)
        T_feat = X_feat.shape[0]
        scores = np.full(len(X), np.nan)

        if T_feat < self.seq_len:
            return scores

        X_seq = np.empty((T_feat - self.seq_len + 1, self.seq_len, X_feat.shape[1]),
                         dtype=np.float32)
        for i in range(len(X_seq)):
            X_seq[i] = X_feat[i:i + self.seq_len]

        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
            # Process in chunks to avoid memory issues
            chunk_size = 256
            preds = []
            for start in range(0, len(X_t), chunk_size):
                preds.append(self._model(X_t[start:start + chunk_size]).cpu().numpy())
            pred = np.concatenate(preds)

        # Map back: sequence prediction at index i corresponds to time
        # (lookback-1) + (seq_len-1) + i in the original X
        offset = (self.lookback - 1) + (self.seq_len - 1)
        scores[offset:offset + len(pred)] = pred

        return scores

    def _build_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Build rolling features — identical to RF."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape
        features = []
        for t in range(self.lookback - 1, T):
            window = X[t - self.lookback + 1:t + 1]
            row = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.min(window, axis=0),
                np.max(window, axis=0),
            ])
            features.append(row)
        return np.array(features)


# ---------------------------------------------------------------------------
# 2. TCN Regime Detector
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Single TCN block: dilated causal Conv1d → BatchNorm → ReLU → Dropout."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        dilation: int, dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

        # Residual connection
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove future values (causal: keep only first seq_len outputs)
        out = out[:, :, :x.size(2)]
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class _TCNModel(nn.Module):
    """3-layer TCN with dilations [1, 2, 4]."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 64,
        kernel_size: int = 3, dropout: float = 0.2,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            _TCNBlock(input_dim, hidden_dim, kernel_size, dilation=1, dropout=dropout),
            _TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=2, dropout=dropout),
            _TCNBlock(hidden_dim, hidden_dim, kernel_size, dilation=4, dropout=dropout),
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        # x: (batch, seq_len, input_dim) → transpose for Conv1d
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.blocks(x)     # (batch, hidden_dim, seq_len)
        x = x[:, :, -1]        # last timestep
        return torch.sigmoid(self.fc(x)).squeeze(-1)


class TCNRegimeDetector(BaseRegimeDetector):
    """Temporal Convolutional Network regime detector.

    Uses a 3-layer TCN with dilated causal convolutions (dilation [1,2,4],
    kernel_size=3) trained with BCELoss and Adam optimizer.

    Args:
        hidden_dim: TCN channel dimension.
        kernel_size: Convolution kernel size.
        seq_len: Input sequence length.
        n_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        seed: Random seed.
        lookback: Rolling feature window (matches RF).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        seq_len: int = 20,
        n_epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 10,
        seed: int = 42,
        lookback: int = 20,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for TCNRegimeDetector")
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.n_epochs = n_epochs
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self.lookback = lookback
        self._model = None
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "TCN"

    def fit(self, X: np.ndarray, **kwargs) -> 'TCNRegimeDetector':
        return self

    def fit_with_labels(
        self, X: np.ndarray, y: np.ndarray
    ) -> 'TCNRegimeDetector':
        """Train TCN on labeled feature matrix.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (0=normal, 1=crisis).
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_feat = self._build_ml_features(X)
        y_trimmed = y[self.lookback - 1:]

        X_seq, y_seq = _build_sequences(X_feat, y_trimmed, self.seq_len)

        input_dim = X_feat.shape[1]
        self._model = _TCNModel(
            input_dim, self.hidden_dim, self.kernel_size
        ).to(self._device)

        dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)
                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"TCN early stopping at epoch {epoch+1}")
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        return self

    def compute_regime_scores(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit_with_labels() before compute_regime_scores().")

        self._model.eval()
        X_feat = self._build_ml_features(X)

        T_feat = X_feat.shape[0]
        scores = np.full(len(X), np.nan)

        if T_feat < self.seq_len:
            return scores

        X_seq = np.empty((T_feat - self.seq_len + 1, self.seq_len, X_feat.shape[1]),
                         dtype=np.float32)
        for i in range(len(X_seq)):
            X_seq[i] = X_feat[i:i + self.seq_len]

        with torch.no_grad():
            X_t = torch.tensor(X_seq, dtype=torch.float32).to(self._device)
            chunk_size = 256
            preds = []
            for start in range(0, len(X_t), chunk_size):
                preds.append(self._model(X_t[start:start + chunk_size]).cpu().numpy())
            pred = np.concatenate(preds)

        offset = (self.lookback - 1) + (self.seq_len - 1)
        scores[offset:offset + len(pred)] = pred

        return scores

    def _build_ml_features(self, X: np.ndarray) -> np.ndarray:
        """Build rolling features — identical to RF."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, d = X.shape
        features = []
        for t in range(self.lookback - 1, T):
            window = X[t - self.lookback + 1:t + 1]
            row = np.concatenate([
                np.mean(window, axis=0),
                np.std(window, axis=0),
                np.min(window, axis=0),
                np.max(window, axis=0),
            ])
            features.append(row)
        return np.array(features)
