from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp


def detect_drift(reference: pd.DataFrame, window: pd.DataFrame, features: list[str], alpha: float = 0.01) -> list[str]:
    """Features whose distribution in `window` differs from `reference` (two-sample KS test,
    Bonferroni-corrected across features)."""
    threshold = alpha / max(len(features), 1)
    return [f for f in features if ks_2samp(reference[f], window[f]).pvalue < threshold]


class DriftMonitor:
    """Buffers incoming feature rows and tests each full window against the training reference."""

    def __init__(self, reference: pd.DataFrame, features: list[str], window_size: int = 1000, alpha: float = 0.01):
        self.reference, self.features = reference, features
        self.window_size, self.alpha = window_size, alpha
        self._buffer: list[pd.DataFrame] = []
        self._buffered = 0

    def update(self, batch: pd.DataFrame) -> list[str]:
        self._buffer.append(batch[self.features])
        self._buffered += len(batch)
        if self._buffered < self.window_size:
            return []
        window = pd.concat(self._buffer)
        self._buffer, self._buffered = [], 0
        return detect_drift(self.reference, window, self.features, self.alpha)
