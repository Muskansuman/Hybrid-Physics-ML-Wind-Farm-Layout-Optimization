from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.ml.features import extract_layout_features, layout_to_feature_vector
from wind_farm_opt.optimization.layout import random_layout
from wind_farm_opt.physics.aep import AEPEvaluator


FEATURE_NAMES = sorted(
    [
        "mean_wind_speed",
        "std_wind_speed",
        "min_wind_speed",
        "max_wind_speed",
        "mean_pairwise_distance",
        "min_pairwise_distance",
        "spacing_violation_sum",
        "wake_exposure",
        "centroid_x",
        "centroid_y",
        "spread_x",
        "spread_y",
    ]
)


def generate_layout_dataset(
    wind_map: np.ndarray,
    config: SiteConfig,
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    evaluator = AEPEvaluator(wind_map, config)
    margin = config.margin_grid(wind_map.shape)
    rows: list[dict[str, float]] = []

    for sample_index in range(n_samples):
        x, y = random_layout(config.num_turbines, wind_map.shape, margin, rng)
        features = extract_layout_features(wind_map, x, y, config)
        metrics = evaluator.evaluate(x, y)
        row = {
            "sample_id": sample_index,
            "aep_mw": metrics["aep_mw"],
            **features,
        }
        row["layout_x"] = ",".join(f"{value:.6f}" for value in x)
        row["layout_y"] = ",".join(f"{value:.6f}" for value in y)
        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset(dataset: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return output_path


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_matrix(dataset: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x_matrix = dataset[FEATURE_NAMES].to_numpy(dtype=float)
    y_vector = dataset["aep_mw"].to_numpy(dtype=float)
    return x_matrix, y_vector


def parse_layout_row(row: pd.Series, num_turbines: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([float(value) for value in row["layout_x"].split(",")], dtype=float)
    y = np.array([float(value) for value in row["layout_y"].split(",")], dtype=float)
    if len(x) != num_turbines or len(y) != num_turbines:
        raise ValueError("Stored layout does not match expected turbine count")
    return x, y
