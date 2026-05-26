from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from wind_farm_opt.config import load_config, resolve_path
from wind_farm_opt.data.wind_field import load_wind_map
from wind_farm_opt.evaluation.benchmarks import run_benchmarks
from wind_farm_opt.evaluation.plots import (
    plot_benchmarks,
    plot_convergence,
    plot_layout,
    plot_surrogate_validation,
)
from wind_farm_opt.ml.dataset import generate_layout_dataset, save_dataset
from wind_farm_opt.ml.surrogate import SurrogateModel, train_surrogate
from wind_farm_opt.optimization.gradient_descent import optimize_gradient_descent
from wind_farm_opt.optimization.hybrid import run_hybrid_optimization
from wind_farm_opt.optimization.layout import random_layout
from wind_farm_opt.optimization.multistart import optimize_multistart
from wind_farm_opt.physics.aep import AEPEvaluator


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_experiment_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def cmd_generate_dataset(args: argparse.Namespace) -> None:
    root = _project_root()
    config = load_config(root / args.config)
    raw = _load_experiment_config(root / args.config)
    wind_map = load_wind_map(resolve_path(root, config.wind_map_path))
    dataset = generate_layout_dataset(
        wind_map,
        config,
        n_samples=args.samples,
        seed=args.seed,
    )
    output_path = save_dataset(dataset, root / args.output)
    print(f"Saved dataset with {len(dataset)} samples to {output_path}")


def cmd_train_surrogate(args: argparse.Namespace) -> None:
    root = _project_root()
    from wind_farm_opt.ml.dataset import load_dataset

    dataset = load_dataset(root / args.train)
    raw = _load_experiment_config(root / args.config)
    surrogate_cfg = raw.get("surrogate", {})
    surrogate, metrics, comparison = train_surrogate(
        dataset,
        test_size=float(surrogate_cfg.get("test_size", 0.2)),
        random_state=int(surrogate_cfg.get("random_state", 42)),
    )
    model_path = root / args.model
    surrogate.save(model_path)
    plot_surrogate_validation(comparison, root / "outputs" / "surrogate_validation.png")
    print(f"Saved surrogate model to {model_path}")
    print(f"Surrogate metrics: RMSE={metrics.rmse:.2f}, MAE={metrics.mae:.2f}, R2={metrics.r2:.3f}")


def cmd_optimize(args: argparse.Namespace) -> None:
    root = _project_root()
    config = load_config(root / args.config)
    raw = _load_experiment_config(root / args.config)
    wind_map = load_wind_map(resolve_path(root, config.wind_map_path))
    evaluator = AEPEvaluator(wind_map, config)
    margin = config.margin_grid(wind_map.shape)
    rng = __import__("numpy").random.default_rng(args.seed)
    x_init, y_init = random_layout(config.num_turbines, wind_map.shape, margin, rng)

    if args.solver == "gradient_descent":
        gd_cfg = raw.get("optimization", {}).get("gradient_descent", {})
        result = optimize_gradient_descent(
            evaluator,
            x_init,
            y_init,
            max_iter=int(gd_cfg.get("max_iter", 200)),
            learning_rate=float(gd_cfg.get("learning_rate", 0.1)),
        )
    elif args.solver == "multistart":
        ms_cfg = raw.get("optimization", {}).get("multistart", {})
        result = optimize_multistart(
            evaluator,
            config.num_turbines,
            n_starts=int(ms_cfg.get("n_starts", 5)),
            seed=args.seed,
            solver="slsqp",
        )
    elif args.solver == "hybrid":
        hybrid_cfg = raw.get("optimization", {}).get("hybrid", {})
        surrogate = SurrogateModel.load(root / args.surrogate)
        hybrid = run_hybrid_optimization(
            wind_map,
            config,
            surrogate,
            n_candidates=int(hybrid_cfg.get("n_candidates", 2000)),
            top_k=int(hybrid_cfg.get("top_k", 10)),
            seed=args.seed,
            refine_solver=str(hybrid_cfg.get("refine_solver", "slsqp")),
        )
        result = hybrid.best_result
        print(
            "Hybrid search used "
            f"{hybrid.surrogate_evaluations} surrogate evals and "
            f"{hybrid.physics_evaluations} physics evals"
        )
    else:
        result = optimize_multistart(
            evaluator,
            config.num_turbines,
            n_starts=1,
            seed=args.seed,
            solver=args.solver,
        )

    output_dir = root / "outputs"
    plot_layout(wind_map, x_init, y_init, config, "Initial Layout", output_dir / "initial_layout.png")
    plot_layout(wind_map, result.x, result.y, config, "Optimized Layout", output_dir / "optimized_layout.png")
    if result.history:
        plot_convergence(result, output_dir / "convergence.png")

    metrics = evaluator.evaluate(result.x, result.y)
    print(f"Solver: {result.solver}")
    print(f"AEP: {metrics['aep_mw']:.2f} MW")
    print(f"Feasible: {bool(metrics['feasible'])}")
    print(f"Saved plots to {output_dir}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    root = _project_root()
    config = load_config(root / args.config)
    raw = _load_experiment_config(root / args.config)
    wind_map = load_wind_map(resolve_path(root, config.wind_map_path))
    benchmark_cfg = raw.get("benchmark", {})
    surrogate_cfg = raw.get("surrogate", {})
    hybrid_cfg = raw.get("optimization", {}).get("hybrid", {})
    ms_cfg = raw.get("optimization", {}).get("multistart", {})

    benchmark_df = run_benchmarks(
        wind_map,
        config,
        seed=int(benchmark_cfg.get("seed", 42)),
        multistart_runs=int(ms_cfg.get("n_starts", 3)),
        surrogate_samples=int(surrogate_cfg.get("n_samples", 1500)),
        hybrid_candidates=int(hybrid_cfg.get("n_candidates", 1000)),
        hybrid_top_k=int(hybrid_cfg.get("top_k", 5)),
    )

    output_dir = Path(benchmark_cfg.get("output_dir", "outputs"))
    output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    benchmark_df.to_csv(csv_path, index=False)
    plot_benchmarks(benchmark_df, output_dir / "benchmark_comparison.png")

    print(f"Baseline AEP: {benchmark_df.attrs['baseline_aep_mw']:.2f} MW")
    print(benchmark_df.to_string(index=False))
    print(f"Surrogate metrics: {benchmark_df.attrs['surrogate_metrics']}")
    print(f"Saved benchmark outputs to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wind farm layout optimization")
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Path to experiment config",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("generate-dataset", help="Generate labeled layout dataset")
    dataset_parser.add_argument("--samples", type=int, default=3000)
    dataset_parser.add_argument("--output", default="data/layouts.csv")
    dataset_parser.add_argument("--seed", type=int, default=42)
    dataset_parser.set_defaults(func=cmd_generate_dataset)

    train_parser = subparsers.add_parser("train-surrogate", help="Train XGBoost surrogate")
    train_parser.add_argument("--train", default="data/layouts.csv")
    train_parser.add_argument("--model", default="models/aep_surrogate.json")
    train_parser.set_defaults(func=cmd_train_surrogate)

    optimize_parser = subparsers.add_parser("optimize", help="Run optimization pipeline")
    optimize_parser.add_argument(
        "--solver",
        choices=["gradient_descent", "slsqp", "lbfgsb", "multistart", "hybrid"],
        default="slsqp",
    )
    optimize_parser.add_argument("--surrogate", default="models/aep_surrogate.json")
    optimize_parser.add_argument("--seed", type=int, default=42)
    optimize_parser.set_defaults(func=cmd_optimize)

    benchmark_parser = subparsers.add_parser("benchmark", help="Compare optimization methods")
    benchmark_parser.set_defaults(func=cmd_benchmark)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
