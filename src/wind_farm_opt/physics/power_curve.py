from __future__ import annotations

import numpy as np


def turbine_power_mw(wind_speed: float | np.ndarray) -> float | np.ndarray:
    """Piecewise turbine power curve in MW."""
    wind_speed = np.asarray(wind_speed, dtype=float)
    power = np.zeros_like(wind_speed, dtype=float)

    partial = (wind_speed >= 4.0) & (wind_speed < 12.0)
    rated = (wind_speed >= 12.0) & (wind_speed < 25.0)

    power[partial] = 0.5 * (wind_speed[partial] - 4.0) ** 2
    power[rated] = 40.0 + 1.2 * (wind_speed[rated] - 12.0)

    if power.ndim == 0:
        return float(power)
    return power
