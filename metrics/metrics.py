from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch import Tensor


def _as_array(values: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def rmse(predictions: Tensor | np.ndarray, targets: Tensor | np.ndarray) -> float:
    error = _as_array(predictions) - _as_array(targets)
    return float(np.sqrt(np.mean(error**2)))


def mae(predictions: Tensor | np.ndarray, targets: Tensor | np.ndarray) -> float:
    error = _as_array(predictions) - _as_array(targets)
    return float(np.mean(np.abs(error)))


def mape(predictions: Tensor | np.ndarray, targets: Tensor | np.ndarray) -> float:
    predictions = _as_array(predictions)
    targets = _as_array(targets)
    denominator = np.maximum(np.abs(targets), np.finfo(np.float64).eps)
    return float(np.mean(np.abs(predictions - targets) / denominator) * 100.0)


def regression_metrics(
    predictions: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
) -> Dict[str, float]:
    predictions = _as_array(predictions)
    targets = _as_array(targets)
    return {
        "rmse": rmse(predictions, targets),
        "mae": mae(predictions, targets),
        "mape": mape(predictions, targets),
    }
