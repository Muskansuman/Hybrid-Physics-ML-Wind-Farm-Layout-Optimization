from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SiteConfig:
    size_km: float = 20.0
    wind_map_path: str = "wind_speed_map.csv"
    num_turbines: int = 30
    rotor_diameter_m: float = 80.0
    min_spacing_factor: float = 5.0
    wind_directions_deg: list[float] = field(default_factory=lambda: [0.0])
    wind_probabilities: list[float] = field(default_factory=lambda: [1.0])
    penalty_coefficient: float = 2000.0

    @property
    def min_spacing_m(self) -> float:
        return self.min_spacing_factor * self.rotor_diameter_m

    def grid_spacing_km(self, grid_shape: tuple[int, int]) -> float:
        return self.size_km / grid_shape[1]

    def rotor_diameter_grid(self, grid_shape: tuple[int, int]) -> float:
        spacing_km = self.grid_spacing_km(grid_shape)
        return self.rotor_diameter_m / (spacing_km * 1000.0)

    def min_spacing_grid(self, grid_shape: tuple[int, int]) -> float:
        return self.min_spacing_factor * self.rotor_diameter_grid(grid_shape)

    def margin_grid(self, grid_shape: tuple[int, int]) -> float:
        return self.rotor_diameter_grid(grid_shape)


def load_config(path: str | Path) -> SiteConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    site = raw.get("site", {})
    turbine = raw.get("turbine", {})
    wind = raw.get("wind", {})
    optimization = raw.get("optimization", {})

    directions = [float(v) for v in wind.get("directions_deg", [0.0])]
    probabilities = [float(v) for v in wind.get("probabilities", [1.0])]
    if len(directions) != len(probabilities):
        raise ValueError("wind directions and probabilities must have the same length")
    if not probabilities:
        raise ValueError("wind probabilities must not be empty")

    prob_sum = sum(probabilities)
    probabilities = [value / prob_sum for value in probabilities]

    return SiteConfig(
        size_km=float(site.get("size_km", 20.0)),
        wind_map_path=str(site.get("wind_map_path", "wind_speed_map.csv")),
        num_turbines=int(turbine.get("count", 30)),
        rotor_diameter_m=float(turbine.get("rotor_diameter_m", 80.0)),
        min_spacing_factor=float(turbine.get("min_spacing_factor", 5.0)),
        wind_directions_deg=directions,
        wind_probabilities=probabilities,
        penalty_coefficient=float(optimization.get("penalty_coefficient", 2000.0)),
    )


def resolve_path(base_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
