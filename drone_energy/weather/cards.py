"""
cards.py — Loader and manager for ERA5 weather scenario cards.
"""

import os
import json

CARDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "scenario_cards")


class ScenarioCard:
    """Represents real-world weather statistics for one environment scenario."""

    def __init__(self, data: dict):
        self.scenario_id = data.get("scenario_id", "custom")
        self.name = data.get("name", "Custom Scenario")
        self.data_source = data.get("data_source", "Synthetic / ERA5")
        self.period = data.get("period", "N/A")
        
        stats = data.get("statistics", {})
        self.mean_wind_speed_ms = float(stats.get("mean_wind_speed_ms", 3.0))
        self.std_wind_speed_ms = float(stats.get("std_wind_speed_ms", 1.0))
        self.primary_wind_dir_rad = float(stats.get("primary_wind_dir_rad", 0.0))
        self.gust_factor = float(stats.get("gust_factor", 1.3))
        self.mean_temperature_c = float(stats.get("mean_temperature_c", 20.0))
        self.std_temperature_c = float(stats.get("std_temperature_c", 2.0))
        self.mean_surface_pressure_hpa = float(stats.get("mean_surface_pressure_hpa", 1013.25))
        self.air_density_kg_m3 = float(stats.get("air_density_kg_m3", 1.225))
        self.shear_exponent = float(stats.get("shear_exponent", 0.14))
        self.neighbour_contrast = float(stats.get("neighbour_contrast", 0.15))

    @classmethod
    def load(cls, scenario_id: str) -> "ScenarioCard":
        """Load scenario card JSON by ID from data/scenario_cards/."""
        path = os.path.join(CARDS_DIR, f"{scenario_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scenario card not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)


def get_available_cards() -> list:
    """Return list of available scenario card IDs."""
    if not os.path.exists(CARDS_DIR):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(CARDS_DIR) if f.endswith(".json")]
