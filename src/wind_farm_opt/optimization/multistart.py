from __future__ import annotations

import numpy as np

from wind_farm_opt.optimization.gradient_descent import (
    OptimizationResult,
    optimize_gradient_descent,
)
from wind_farm_opt.optimization.layout import random_layout
from wind_farm_opt.optimization.scipy_solvers import SolverConfig, optimize_scipy
from wind_farm_opt.physics.aep import AEPEvaluator


def optimize_multistart(
    evaluator: AEPEvaluator,
    num_turbines: int,
    n_starts: int = 5,
    seed: int = 42,
    solver: str = "slsqp",
    max_iter: int = 200,
    learning_rate: float = 0.1,
) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    margin = evaluator.config.margin_grid(evaluator.wind_map.shape)
    best: OptimizationResult | None = None

    for start_index in range(n_starts):
        x_init, y_init = random_layout(
            num_turbines,
            evaluator.wind_map.shape,
            margin,
            rng,
        )

        if solver == "gradient_descent":
            result = optimize_gradient_descent(
                evaluator,
                x_init,
                y_init,
                max_iter=max_iter,
                learning_rate=learning_rate,
            )
        elif solver == "lbfgsb":
            result = optimize_scipy(
                evaluator,
                x_init,
                y_init,
                SolverConfig(method="L-BFGS-B", maxiter=max_iter),
            )
        else:
            result = optimize_scipy(
                evaluator,
                x_init,
                y_init,
                SolverConfig(method="SLSQP", maxiter=max_iter),
            )

        result.solver = f"multistart_{result.solver}_{start_index}"
        if best is None or result.aep_mw > best.aep_mw:
            best = result

    assert best is not None
    best.solver = f"multistart_{solver}"
    return best
