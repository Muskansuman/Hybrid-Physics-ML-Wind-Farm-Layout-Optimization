from __future__ import annotations

import numpy as np


def pack_layout(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def unpack_layout(z: np.ndarray, num_turbines: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(z, dtype=float)
    return z[:num_turbines].copy(), z[num_turbines:].copy()


def random_layout(
    num_turbines: int,
    grid_shape: tuple[int, int],
    margin: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    width = grid_shape[1]
    height = grid_shape[0]
    x = rng.uniform(margin, width - margin, num_turbines)
    y = rng.uniform(margin, height - margin, num_turbines)
    return x, y
