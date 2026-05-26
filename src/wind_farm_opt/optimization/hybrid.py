from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.ml.dataset import FEATURE_NAMES, generate_layout_dataset
from wind_farm_opt.ml.features import layout_to_feature_vector
from wind_farm_opt.ml.surrogate import SurrogateModel
from wind_farm_opt.optimization.gradient_descent import OptimizationResult, optimize_gradient_descent
from wind_farm_opt.optimization.layout import random_layout
from wind_farm_opt.optimization.scipy_solvers import SolverConfig, optimize_scipy
from wind_farm_opt.physics.aep import AEPEvaluator


@dataclass
class HybridResult:
    best_result: OptimizationResult
    surrogate_evaluations: int
    physics_evaluations: int
    candidate_table: pd.DataFrame


def _refine_layout(
    evaluator: AEPEvaluator,
    x_init: np.ndarray,
    y_init: np.ndarray,
    solver: str,
    max_iter: int,
) -> OptimizationResult:
    if solver == "gradient_descent":
        return optimize_gradient_descent(evaluator, x_init, y_init, max_iter=max_iter)
    if solver == "lbfgsb":
        return optimize_scipy(
            evaluator,
            x_init,
            y_init,
            SolverConfig(method="L-BFGS-B", maxiter=max_iter),
        )
    return optimize_scipy(
        evaluator,
        x_init,
        y_init,
        SolverConfig(method="SLSQP", maxiter=max_iter),
    )


def run_hybrid_optimization(
    wind_map: np.ndarray,
    config: SiteConfig,
    surrogate: SurrogateModel,
    n_candidates: int = 2000,
    top_k: int = 10,
    seed: int = 42,
    refine_solver: str = "slsqp",
    max_iter: int = 100,
) -> HybridResult:
    rng = np.random.default_rng(seed)
    evaluator = AEPEvaluator(wind_map, config)
    margin = config.margin_grid(wind_map.shape)

    feature_rows = []
    layouts = []
    for _ in range(n_candidates):
        x, y = random_layout(config.num_turbines, wind_map.shape, margin, rng)
        features = layout_to_feature_vector(wind_map, x, y, config, FEATURE_NAMES)
        feature_rows.append(features)
        layouts.append((x, y))

    feature_matrix = np.vstack(feature_rows)
    predicted_aep = surrogate.predict_from_matrix(feature_matrix)
    ranked_indices = np.argsort(predicted_aep)[::-1][:top_k]

    candidate_rows = []
    best: OptimizationResult | None = None
    physics_evaluations = 0

    for rank, candidate_index in enumerate(ranked_indices, start=1):
        x_init, y_init = layouts[candidate_index]
        refined = _refine_layout(
            evaluator,
            x_init,
            y_init,
            solver=refine_solver,
            max_iter=max_iter,
        )
        refined.solver = f"hybrid_{refine_solver}"
        physics_evaluations += refined.n_evaluations

        candidate_rows.append(
            {
                "rank": rank,
                "predicted_aep_mw": float(predicted_aep[candidate_index]),
                "refined_aep_mw": refined.aep_mw,
            }
        )
        if best is None or refined.aep_mw > best.aep_mw:
            best = refined

    assert best is not None
    return HybridResult(
        best_result=best,
        surrogate_evaluations=n_candidates,
        physics_evaluations=physics_evaluations,
        candidate_table=pd.DataFrame(candidate_rows),
    )


def train_and_run_hybrid(
    wind_map: np.ndarray,
    config: SiteConfig,
    n_samples: int = 3000,
    n_candidates: int = 2000,
    top_k: int = 10,
    seed: int = 42,
    refine_solver: str = "slsqp",
) -> tuple[SurrogateModel, HybridResult]:
    dataset = generate_layout_dataset(wind_map, config, n_samples=n_samples, seed=seed)
    surrogate, _, _ = train_surrogate_from_dataset(dataset)
    hybrid_result = run_hybrid_optimization(
        wind_map,
        config,
        surrogate,
        n_candidates=n_candidates,
        top_k=top_k,
        seed=seed,
        refine_solver=refine_solver,
    )
    return surrogate, hybrid_result


def train_surrogate_from_dataset(dataset: pd.DataFrame):
    from wind_farm_opt.ml.surrogate import train_surrogate

    return train_surrogate(dataset)
