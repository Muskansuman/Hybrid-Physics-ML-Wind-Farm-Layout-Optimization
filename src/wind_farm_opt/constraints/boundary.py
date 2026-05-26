from __future__ import annotations

import numpy as np


def boundary_violation(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple[int, int],
    margin: float,
) -> float:
    width = grid_shape[1]
    height = grid_shape[0]
    violation = 0.0
    violation += np.sum(np.maximum(margin - x, 0.0))
    violation += np.sum(np.maximum(x - (width - margin), 0.0))
    violation += np.sum(np.maximum(margin - y, 0.0))
    violation += np.sum(np.maximum(y - (height - margin), 0.0))
    return float(violation)


def clip_to_boundary(
    x: np.ndarray,
    y: np.ndarray,
    grid_shape: tuple[int, int],
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = grid_shape[1]
    height = grid_shape[0]
    return (
        np.clip(x, margin, width - margin),
        np.clip(y, margin, height - margin),
    )
