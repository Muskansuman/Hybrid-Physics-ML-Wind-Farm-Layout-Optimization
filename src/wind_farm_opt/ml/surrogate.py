from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from wind_farm_opt.ml.dataset import FEATURE_NAMES, build_feature_matrix, load_dataset


@dataclass
class SurrogateMetrics:
    rmse: float
    mae: float
    r2: float


@dataclass
class SurrogateModel:
    model: XGBRegressor
    feature_names: list[str]

    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    def predict_from_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(feature_matrix)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(output_path))

    @classmethod
    def load(cls, path: str | Path) -> "SurrogateModel":
        model = XGBRegressor()
        model.load_model(str(path))
        return cls(model=model, feature_names=FEATURE_NAMES.copy())


def train_surrogate(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[SurrogateModel, SurrogateMetrics, pd.DataFrame]:
    x_matrix, y_vector = build_feature_matrix(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix,
        y_vector,
        test_size=test_size,
        random_state=random_state,
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = SurrogateMetrics(
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        mae=float(mean_absolute_error(y_test, predictions)),
        r2=float(r2_score(y_test, predictions)),
    )

    comparison = pd.DataFrame(
        {
            "true_aep_mw": y_test,
            "predicted_aep_mw": predictions,
        }
    )
    return SurrogateModel(model=model, feature_names=FEATURE_NAMES.copy()), metrics, comparison
