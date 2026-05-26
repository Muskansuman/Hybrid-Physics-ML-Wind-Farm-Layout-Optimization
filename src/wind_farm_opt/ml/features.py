from __future__ import annotations

import numpy as np

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.data.wind_field import sample_wind_speed
from wind_farm_opt.physics.wake import jensen_wake_deficit


def extract_layout_features(
    wind_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: SiteConfig,
) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rotor_diameter = config.rotor_diameter_grid(wind_map.shape)
    min_spacing = config.min_spacing_grid(wind_map.shape)

    wind_speeds = np.array([sample_wind_speed(wind_map, xi, yi) for xi, yi in zip(x, y)])

    pairwise_distances = []
    wake_exposure = 0.0
    for i in range(len(x)):
        local_wake = 0.0
        for j in range(len(x)):
            if i == j:
                continue
            distance = np.hypot(x[i] - x[j], y[i] - y[j])
            pairwise_distances.append(distance)
            for direction in config.wind_directions_deg:
                local_wake += jensen_wake_deficit(
                    x[i],
                    y[i],
                    x[j],
                    y[j],
                    direction,
                    wind_speeds[i],
                    rotor_diameter,
                )
        wake_exposure += local_wake

    distances = np.array(pairwise_distances, dtype=float)
    spacing_violations = np.maximum(min_spacing - distances, 0.0)

    return {
        "mean_wind_speed": float(np.mean(wind_speeds)),
        "std_wind_speed": float(np.std(wind_speeds)),
        "min_wind_speed": float(np.min(wind_speeds)),
        "max_wind_speed": float(np.max(wind_speeds)),
        "mean_pairwise_distance": float(np.mean(distances)) if len(distances) else 0.0,
        "min_pairwise_distance": float(np.min(distances)) if len(distances) else 0.0,
        "spacing_violation_sum": float(np.sum(spacing_violations)),
        "wake_exposure": float(wake_exposure / max(len(x), 1)),
        "centroid_x": float(np.mean(x)),
        "centroid_y": float(np.mean(y)),
        "spread_x": float(np.std(x)),
        "spread_y": float(np.std(y)),
    }


def features_to_vector(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.array([features[name] for name in feature_names], dtype=float)


def layout_to_feature_vector(
    wind_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: SiteConfig,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    features = extract_layout_features(wind_map, x, y, config)
    names = feature_names or sorted(features.keys())
    return features_to_vector(features, names)
