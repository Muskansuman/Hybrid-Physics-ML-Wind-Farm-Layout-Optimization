from __future__ import annotations

import numpy as np


def jensen_wake_deficit(
    x_i: float,
    y_i: float,
    x_j: float,
    y_j: float,
    wind_direction_deg: float,
    wind_speed: float,
    rotor_diameter: float,
    wake_decay: float = 0.075,
    thrust_coefficient: float = 0.88,
) -> float:
    """Return wake-induced wind speed deficit at turbine i from turbine j."""
    theta = np.radians(wind_direction_deg)
    dx = x_i - x_j
    dy = y_i - y_j
    distance = np.hypot(dx, dy)
    if distance <= 0.0:
        return 0.0

    wake_radius = rotor_diameter / 2.0 + wake_decay * distance
    crosswind_distance = abs(-np.sin(theta) * dx + np.cos(theta) * dy)
    if crosswind_distance > wake_radius:
        return 0.0

    deficit = (1.0 - np.sqrt(1.0 - thrust_coefficient)) * (
        rotor_diameter / (2.0 * wake_radius)
    ) ** 2
    deficit *= np.exp(-0.5 * (crosswind_distance / wake_radius) ** 2)
    return float(deficit * wind_speed)
