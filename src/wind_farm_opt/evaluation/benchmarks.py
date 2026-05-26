from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.data.wind_field import load_wind_map
from wind_farm_opt.ml.dataset import generate_layout_dataset
from wind_farm_opt.ml.surrogate import train_surrogate
from wind_farm_opt.optimization.gradient_descent import optimize_gradient_descent
from wind_farm_opt.optimization.hybrid import run_hybrid_optimization
from wind_farm_opt.optimization.layout import random_layout
from wind_farm_opt.optimization.multistart import optimize_multistart
from wind_farm_opt.physics.aep import AEPEvaluator


@dataclass
class BenchmarkRow:
    method: str
    aep_mw: float
    runtime_sec: float
    n_evaluations: int
    feasible: float
    improvement_pct: float


def _improvement_pct(baseline_aep: float, result_aep: float) -> float:
    if baseline_aep <= 0:
        return 0.0
    return 100.0 * (result_aep - baseline_aep) / baseline_aep


def run_benchmarks(
    wind_map: np.ndarray,
    config: SiteConfig,
    seed: int = 42,
    multistart_runs: int = 3,
    surrogate_samples: int = 1500,
    hybrid_candidates: int = 1000,
    hybrid_top_k: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    evaluator = AEPEvaluator(wind_map, config)
    margin = config.margin_grid(wind_map.shape)
    x_init, y_init = random_layout(config.num_turbines, wind_map.shape, margin, rng)

    baseline_metrics = evaluator.evaluate(x_init, y_init)
    baseline_aep = baseline_metrics["aep_mw"]
    rows: list[BenchmarkRow] = []

    start = time.perf_counter()
    gd_result = optimize_gradient_descent(evaluator, x_init, y_init, max_iter=25, learning_rate=0.1)
    gd_metrics = evaluator.evaluate(gd_result.x, gd_result.y)
    rows.append(
        BenchmarkRow(
            method="gradient_descent",
            aep_mw=gd_result.aep_mw,
            runtime_sec=time.perf_counter() - start,
            n_evaluations=gd_result.n_evaluations,
            feasible=gd_metrics["feasible"],
            improvement_pct=_improvement_pct(baseline_aep, gd_result.aep_mw),
        )
    )

    start = time.perf_counter()
    slsqp_result = optimize_multistart(
        evaluator,
        config.num_turbines,
        n_starts=multistart_runs,
        seed=seed,
        solver="slsqp",
        max_iter=40,
    )
    slsqp_metrics = evaluator.evaluate(slsqp_result.x, slsqp_result.y)
    rows.append(
        BenchmarkRow(
            method="multistart_slsqp",
            aep_mw=slsqp_result.aep_mw,
            runtime_sec=time.perf_counter() - start,
            n_evaluations=slsqp_result.n_evaluations,
            feasible=slsqp_metrics["feasible"],
            improvement_pct=_improvement_pct(baseline_aep, slsqp_result.aep_mw),
        )
    )

    dataset = generate_layout_dataset(
        wind_map,
        config,
        n_samples=surrogate_samples,
        seed=seed,
    )
    surrogate, surrogate_metrics, _ = train_surrogate(dataset, random_state=seed)

    start = time.perf_counter()
    hybrid_result = run_hybrid_optimization(
        wind_map,
        config,
        surrogate,
        n_candidates=hybrid_candidates,
        top_k=hybrid_top_k,
        seed=seed,
        refine_solver="slsqp",
        max_iter=30,
    )
    hybrid_metrics = evaluator.evaluate(
        hybrid_result.best_result.x,
        hybrid_result.best_result.y,
    )
    rows.append(
        BenchmarkRow(
            method="hybrid_xgboost_slsqp",
            aep_mw=hybrid_result.best_result.aep_mw,
            runtime_sec=time.perf_counter() - start,
            n_evaluations=hybrid_result.physics_evaluations,
            feasible=hybrid_metrics["feasible"],
            improvement_pct=_improvement_pct(baseline_aep, hybrid_result.best_result.aep_mw),
        )
    )

    benchmark_df = pd.DataFrame([asdict(row) for row in rows])
    benchmark_df.attrs["baseline_aep_mw"] = baseline_aep
    benchmark_df.attrs["surrogate_metrics"] = asdict(surrogate_metrics)
    return benchmark_df
