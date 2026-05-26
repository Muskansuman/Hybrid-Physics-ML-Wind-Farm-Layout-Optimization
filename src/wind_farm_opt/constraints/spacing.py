from __future__ import annotations

import numpy as np


def spacing_violation(x: np.ndarray, y: np.ndarray, min_spacing: float) -> float:
    violation = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distance = np.hypot(x[i] - x[j], y[i] - y[j])
            if distance < min_spacing:
                violation += min_spacing - distance
    return float(violation)


def spacing_penalty(x: np.ndarray, y: np.ndarray, min_spacing: float) -> float:
    penalty = 0.0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distance = np.hypot(x[i] - x[j], y[i] - y[j])
            if distance < min_spacing:
                penalty += 5.0 * ((min_spacing - distance) / min_spacing) ** 2
    return float(penalty)
