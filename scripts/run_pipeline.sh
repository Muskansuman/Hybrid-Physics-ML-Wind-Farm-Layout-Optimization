#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e .
pip install pytest

python -m wind_farm_opt generate-dataset --samples 200 --output data/layouts.csv --seed 42
python -m wind_farm_opt train-surrogate --train data/layouts.csv --model models/aep_surrogate.json
python -m wind_farm_opt optimize --solver hybrid --seed 42
python -m wind_farm_opt benchmark

echo "Pipeline completed. See outputs/ for results."
