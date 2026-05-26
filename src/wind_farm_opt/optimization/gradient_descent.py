from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wind_farm_opt.constraints.boundary import clip_to_boundary
from wind_farm_opt.physics.aep import AEPEvaluator


@dataclass
class OptimizationResult:
    x: np.ndarray
    y: np.ndarray
    aep_mw: float
    history: list[float]
    n_evaluations: int
    solver: str


def numerical_gradient(
    evaluator: AEPEvaluator,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    num_turbines = len(x)
    grad_x = np.zeros(num_turbines)
    grad_y = np.zeros(num_turbines)

    for index in range(num_turbines):
        x_plus = x.copy()
        x_plus[index] += epsilon
        x_minus = x.copy()
        x_minus[index] -= epsilon
        grad_x[index] = (
            10.0
            * (
                evaluator.penalized_objective(x_plus, y)
                - evaluator.penalized_objective(x_minus, y)
            )
            / (2.0 * epsilon)
        )

        y_plus = y.copy()
        y_plus[index] += epsilon
        y_minus = y.copy()
        y_minus[index] -= epsilon
        grad_y[index] = (
            10.0
            * (
                evaluator.penalized_objective(x, y_plus)
                - evaluator.penalized_objective(x, y_minus)
            )
            / (2.0 * epsilon)
        )

    return grad_x, grad_y


def optimize_gradient_descent(
    evaluator: AEPEvaluator,
    x_init: np.ndarray,
    y_init: np.ndarray,
    max_iter: int = 200,
    learning_rate: float = 0.1,
    tol: float = 1e-5,
) -> OptimizationResult:
    x = x_init.copy().astype(float)
    y = y_init.copy().astype(float)
    margin = evaluator.config.margin_grid(evaluator.wind_map.shape)
    grid_shape = evaluator.wind_map.shape

    best_x = x.copy()
    best_y = y.copy()
    best_objective = float("inf")
    history: list[float] = []
    n_evaluations = 0

    for _ in range(max_iter):
        grad_x, grad_y = numerical_gradient(evaluator, x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        x, y = clip_to_boundary(x, y, grid_shape, margin)

        objective = evaluator.penalized_objective(x, y)
        n_evaluations += 1 + 4 * len(x)
        history.append(-objective)

        if objective < best_objective:
            best_objective = objective
            best_x = x.copy()
            best_y = y.copy()

        if np.all(np.abs(grad_x) < tol) and np.all(np.abs(grad_y) < tol):
            break

    metrics = evaluator.evaluate(best_x, best_y)
    return OptimizationResult(
        x=best_x,
        y=best_y,
        aep_mw=metrics["aep_mw"],
        history=history,
        n_evaluations=n_evaluations,
        solver="gradient_descent",
    )
