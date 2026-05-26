from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.optimization.gradient_descent import OptimizationResult
from wind_farm_opt.physics.aep import AEPEvaluator


def plot_layout(
    wind_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: SiteConfig,
    title: str,
    output_path: str | Path,
) -> Path:
    evaluator = AEPEvaluator(wind_map, config)
    metrics = evaluator.evaluate(x, y)
    rotor_diameter = config.rotor_diameter_grid(wind_map.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        wind_map,
        cmap="coolwarm",
        origin="lower",
        extent=[0, config.size_km, 0, config.size_km],
    )
    fig.colorbar(im, ax=ax, label="Wind Speed (m/s)")
    x_km = x / wind_map.shape[1] * config.size_km
    y_km = y / wind_map.shape[0] * config.size_km
    ax.scatter(x_km, y_km, c="black", marker="x", s=80, label="Turbines")

    radius_km = rotor_diameter / wind_map.shape[1] * config.size_km
    for xi, yi in zip(x_km, y_km):
        circle = plt.Circle(
            (xi, yi),
            radius_km,
            fill=False,
            linestyle="--",
            color="gray",
            alpha=0.5,
        )
        ax.add_patch(circle)

    ax.set_title(f"{title} - AEP: {metrics['aep_mw']:.2f} MW")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_convergence(result: OptimizationResult, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.history, linewidth=2)
    ax.set_title(f"Convergence - {result.solver}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("AEP (MW)")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_benchmarks(benchmark_df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(benchmark_df["method"], benchmark_df["aep_mw"], color="steelblue")
    axes[0].set_title("Final AEP by Method")
    axes[0].set_ylabel("AEP (MW)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(benchmark_df["method"], benchmark_df["runtime_sec"], color="darkorange")
    axes[1].set_title("Runtime by Method")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=20)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_surrogate_validation(comparison_df: pd.DataFrame, output_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        comparison_df["true_aep_mw"],
        comparison_df["predicted_aep_mw"],
        alpha=0.5,
    )
    min_value = min(comparison_df["true_aep_mw"].min(), comparison_df["predicted_aep_mw"].min())
    max_value = max(comparison_df["true_aep_mw"].max(), comparison_df["predicted_aep_mw"].max())
    ax.plot([min_value, max_value], [min_value, max_value], "--", color="black")
    ax.set_xlabel("True AEP (MW)")
    ax.set_ylabel("Predicted AEP (MW)")
    ax.set_title("Surrogate Validation")
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
