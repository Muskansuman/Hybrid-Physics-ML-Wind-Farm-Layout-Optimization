# Hybrid Physics–ML Wind Farm Layout Optimization

**Author:** Muskan Suman  
**Repository:** [github.com/Muskansuman/new_project](https://github.com/Muskansuman/new_project)

Optimizes wind turbine placement on a 20 km × 20 km site to **maximize Annual Energy Production (AEP)** under wake effects, spacing, and boundary constraints.

Combines **Jensen wake physics simulation**, **constrained optimization** (gradient descent, SLSQP, multistart), and an **XGBoost surrogate** for fast layout screening. ML accelerates search; physics validates the final layout.

---

## Key Skills

**Python · NumPy · SciPy · Pandas · XGBoost · Matplotlib · Jupyter · pytest**

Constrained optimization · Surrogate modeling · Feature engineering · Jensen wake model · AEP estimation · Hybrid ML + physics pipelines · CLI & modular package design

---

## Results

Benchmark: **30 turbines**, **20×20 wind map**, seed `42`.

| Method | AEP (MW) | Improvement | Runtime (s) | Feasible |
|--------|----------|-------------|-------------|----------|
| Random baseline | 285.53 | — | — | — |
| Gradient descent | 292.17 | +2.33% | 53.6 | Yes |
| Multistart SLSQP | **345.74** | **+21.09%** | 251.5 | Yes |
| Hybrid XGBoost + SLSQP | 343.91 | +20.45% | 185.7 | Yes |

**XGBoost surrogate:** R² = 0.90 · RMSE = 3.11 MW · MAE = 2.27 MW (800 training layouts)

**Outputs:** `outputs/optimized_layout.png` · `outputs/benchmark_comparison.png` · `outputs/surrogate_validation.png`

---

## Setup & Usage

```bash
git clone https://github.com/Muskansuman/new_project.git
cd new_project

python -m venv .venv
source .venv/bin/activate
pip install -e .

python -m wind_farm_opt generate-dataset --samples 800 --output data/layouts.csv
python -m wind_farm_opt train-surrogate --train data/layouts.csv --model models/aep_surrogate.json
python -m wind_farm_opt optimize --solver hybrid --seed 42
python -m wind_farm_opt benchmark
```

---

## Project Structure

```text
src/wind_farm_opt/   # physics, optimization, ML, evaluation, CLI
configs/             # experiment settings
tests/               # unit tests
outputs/             # plots and benchmark results
wind_speed_map.csv   # input wind resource grid
```

---

## License

MIT License — see [LICENSE](LICENSE).
