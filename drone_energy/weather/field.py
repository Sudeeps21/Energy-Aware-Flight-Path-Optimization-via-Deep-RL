"""
field.py — WeatherField unified interface for simulation environment.

Implements the interface contract defined in TEAM_PLAN_TASKS_TIMELINE_RESOURCES.md Section 3.3:
  - Inputs: position (x, y, z in metres), time (s), scenario_id
  - Outputs: mean wind, gust, ambient temperature, pressure, density, lookahead obs
"""

import numpy as np
from typing import Dict, Any, Tuple

from drone_energy.weather.cards import ScenarioCard
from drone_energy.weather.gusts import DrydenGustModel
from drone_energy.weather.zones import WindZoneMap, make_scenario


class WeatherField:
    """
    Unified weather field combining spatial wind/temp zones, terrain lapse rates,
    altitude density scaling, and dynamic Dryden turbulence gusts.
    """

    def __init__(
        self,
        scenario_id: str = "calm",
        seed: int = 42,
        world_size: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        dt: float = 0.02,
    ):
        self.scenario_id = scenario_id
        self.seed = seed
        self.world_size = np.array(world_size, dtype=np.float32)
        self.dt = dt

        # Load card if available, else fallback to preset
        try:
            self.card = ScenarioCard.load(scenario_id)
        except (FileNotFoundError, ValueError):
            self.card = None

        # Underlying spatial grid map
        self.zone_map = make_scenario(scenario_id, seed=seed, world_size=world_size)

        # Gust model
        mean_speed = self.card.mean_wind_speed_ms if self.card else 3.0
        gust_factor = self.card.gust_factor if self.card else 1.3
        self.gust_model = DrydenGustModel(
            dt=dt,
            mean_wind_speed=mean_speed,
            gust_factor=gust_factor,
            seed=seed,
        )

    def reset(self, seed: int = None):
        """Reset weather field and gust random state."""
        if seed is not None:
            self.seed = seed
        self.zone_map = make_scenario(self.scenario_id, seed=self.seed, world_size=self.world_size)
        self.gust_model.reset(seed=self.seed)

    def get_state(self, pos: np.ndarray, time_s: float = 0.0) -> Dict[str, Any]:
        """
        Query wind, gust, temperature, pressure, and air density at position (x,y,z).

        Parameters
        ----------
        pos : np.ndarray (3,)
            Position vector (m).
        time_s : float
            Simulation time in seconds.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
              - mean_wind : (3,) array (m/s)
              - gust_wind : (3,) array (m/s)
              - total_wind: (3,) array (m/s)
              - temperature_c: float (°C)
              - pressure_hpa : float (hPa)
              - air_density  : float (kg/m³)
        """
        cell = self.zone_map.get_cell(pos)
        gust = self.gust_model.step()

        mean_wind = cell.wind_vector
        total_wind = mean_wind + gust
        temp_c = cell.temperature_c

        # ISA lapse rate for pressure & density
        press_hpa = self.card.mean_surface_pressure_hpa if self.card else 1013.25
        temp_k = temp_c + 273.15
        air_density = (press_hpa * 100.0) / (287.05 * temp_k)

        return {
            "mean_wind": mean_wind,
            "gust_wind": gust,
            "total_wind": total_wind,
            "temperature_c": temp_c,
            "pressure_hpa": press_hpa,
            "air_density": air_density,
        }

    def get_lookahead_obs(self, pos: np.ndarray) -> np.ndarray:
        """
        Return 7-cell neighbourhood wind & temperature observation (28 floats)
        around pos for predictive RL observation.
        """
        return self.zone_map.get_neighbourhood_obs(pos)
