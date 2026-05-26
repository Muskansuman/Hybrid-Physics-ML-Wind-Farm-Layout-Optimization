from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_wind_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return pd.read_csv(path, header=None).values.astype(float)


def sample_wind_speed(wind_map: np.ndarray, x: float, y: float) -> float:
    """Bilinear interpolation of wind speed at grid coordinates."""
    height, width = wind_map.shape
    x = float(np.clip(x, 0.0, width - 1.001))
    y = float(np.clip(y, 0.0, height - 1.001))

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)

    tx = x - x0
    ty = y - y0

    top = (1.0 - tx) * wind_map[y0, x0] + tx * wind_map[y0, x1]
    bottom = (1.0 - tx) * wind_map[y1, x0] + tx * wind_map[y1, x1]
    return float((1.0 - ty) * top + ty * bottom)
