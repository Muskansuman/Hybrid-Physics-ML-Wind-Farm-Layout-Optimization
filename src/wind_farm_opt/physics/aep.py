from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wind_farm_opt.config import SiteConfig
from wind_farm_opt.constraints.boundary import boundary_violation
from wind_farm_opt.constraints.spacing import spacing_penalty, spacing_violation
from wind_farm_opt.data.wind_field import sample_wind_speed
from wind_farm_opt.physics.power_curve import turbine_power_mw
from wind_farm_opt.physics.wake import jensen_wake_deficit


@dataclass
class AEPEvaluator:
    wind_map: np.ndarray
    config: SiteConfig

    def rotor_diameter_grid(self) -> float:
        return self.config.rotor_diameter_grid(self.wind_map.shape)

    def min_spacing_grid(self) -> float:
        return self.config.min_spacing_grid(self.wind_map.shape)

    def total_power(
        self,
        x: np.ndarray,
        y: np.ndarray,
        wind_direction_deg: float | None = None,
    ) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        rotor_diameter = self.rotor_diameter_grid()
        directions = (
            [wind_direction_deg]
            if wind_direction_deg is not None
            else self.config.wind_directions_deg
        )
        probabilities = (
            [1.0]
            if wind_direction_deg is not None
            else self.config.wind_probabilities
        )

        total = 0.0
        for direction, probability in zip(directions, probabilities):
            for index in range(len(x)):
                wind_speed = sample_wind_speed(self.wind_map, x[index], y[index])
                wake_effect = 0.0
                for other in range(len(x)):
                    if other == index:
                        continue
                    wake_effect += jensen_wake_deficit(
                        x[index],
                        y[index],
                        x[other],
                        y[other],
                        direction,
                        wind_speed,
                        rotor_diameter,
                    )
                effective_wind_speed = max(0.0, wind_speed - wake_effect)
                total += probability * 2.0 * float(turbine_power_mw(effective_wind_speed))
        return total

    def penalized_objective(self, x: np.ndarray, y: np.ndarray) -> float:
        penalty = spacing_penalty(x, y, self.min_spacing_grid())
        return -(self.total_power(x, y) - self.config.penalty_coefficient * penalty)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        spacing = spacing_violation(x, y, self.min_spacing_grid())
        boundary = boundary_violation(
            x,
            y,
            self.wind_map.shape,
            self.config.margin_grid(self.wind_map.shape),
        )
        power = self.total_power(x, y)
        return {
            "aep_mw": power,
            "spacing_violation": spacing,
            "boundary_violation": boundary,
            "feasible": float(spacing <= 0.0 and boundary <= 0.0),
        }


def evaluate_aep(
    wind_map: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    config: SiteConfig,
) -> float:
    return AEPEvaluator(wind_map, config).total_power(x, y)
