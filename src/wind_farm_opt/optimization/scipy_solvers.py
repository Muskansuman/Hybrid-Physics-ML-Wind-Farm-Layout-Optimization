from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from wind_farm_opt.constraints.boundary import clip_to_boundary
from wind_farm_opt.optimization.gradient_descent import OptimizationResult
from wind_farm_opt.optimization.layout import pack_layout, unpack_layout
from wind_farm_opt.physics.aep import AEPEvaluator


@dataclass
class SolverConfig:
    method: str = "SLSQP"
    maxiter: int = 200


def _build_bounds(evaluator: AEPEvaluator, num_turbines: int):
    margin = evaluator.config.margin_grid(evaluator.wind_map.shape)
    width = evaluator.wind_map.shape[1]
    height = evaluator.wind_map.shape[0]
    low = margin
    high_x = width - margin
    high_y = height - margin
    return [(low, high_x)] * num_turbines + [(low, high_y)] * num_turbines


def _build_spacing_constraints(evaluator: AEPEvaluator, num_turbines: int):
    min_spacing = evaluator.min_spacing_grid()
    constraints = []

    for i in range(num_turbines):
        for j in range(i + 1, num_turbines):

            def constraint(values, i=i, j=j):
                x, y = unpack_layout(values, num_turbines)
                distance = np.hypot(x[i] - x[j], y[i] - y[j])
                return distance - min_spacing

            constraints.append({"type": "ineq", "fun": constraint})

    return constraints


def optimize_scipy(
    evaluator: AEPEvaluator,
    x_init: np.ndarray,
    y_init: np.ndarray,
    solver_config: SolverConfig | None = None,
) -> OptimizationResult:
    solver_config = solver_config or SolverConfig()
    num_turbines = len(x_init)
    z0 = pack_layout(x_init, y_init)
    n_evaluations = 0
    history: list[float] = []

    def objective(values: np.ndarray) -> float:
        nonlocal n_evaluations
        x, y = unpack_layout(values, num_turbines)
        value = evaluator.penalized_objective(x, y)
        n_evaluations += 1
        history.append(-value)
        return value

    bounds = _build_bounds(evaluator, num_turbines)
    options = {"maxiter": solver_config.maxiter, "disp": False}

    if solver_config.method.upper() == "L-BFGS-B":
        result = minimize(objective, z0, method="L-BFGS-B", bounds=bounds, options=options)
    else:
        constraints = _build_spacing_constraints(evaluator, num_turbines)
        result = minimize(
            objective,
            z0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

    x, y = unpack_layout(result.x, num_turbines)
    margin = evaluator.config.margin_grid(evaluator.wind_map.shape)
    x, y = clip_to_boundary(x, y, evaluator.wind_map.shape, margin)
    metrics = evaluator.evaluate(x, y)

    return OptimizationResult(
        x=x,
        y=y,
        aep_mw=metrics["aep_mw"],
        history=history,
        n_evaluations=n_evaluations,
        solver=solver_config.method.lower(),
    )
