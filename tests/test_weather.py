"""
test_weather.py — Unit test suite for drone_energy.weather package.

Verifies:
  1. ScenarioCard loading and statistics
  2. DrydenGustModel stochastic process properties
  3. WeatherField interface outputs and lookahead observation shape
"""

import unittest
import numpy as np

from drone_energy.weather.cards import ScenarioCard, get_available_cards
from drone_energy.weather.gusts import DrydenGustModel
from drone_energy.weather.field import WeatherField


class TestWeatherPackage(unittest.TestCase):

    def test_scenario_cards_loading(self):
        cards = get_available_cards()
        self.assertIn("monsoon_coastal", cards)
        self.assertIn("cold_winter", cards)
        self.assertIn("calm_temperate", cards)

        card = ScenarioCard.load("monsoon_coastal")
        self.assertGreater(card.mean_wind_speed_ms, 0.0)
        self.assertGreater(card.air_density_kg_m3, 0.5)

    def test_dryden_gust_model(self):
        gust_model = DrydenGustModel(dt=0.02, mean_wind_speed=5.0, gust_factor=2.0, seed=42)
        samples = [gust_model.step() for _ in range(500)]
        samples = np.array(samples)

        self.assertEqual(samples.shape, (500, 3))
        # Mean should be near 0 over time
        self.assertLess(abs(np.mean(samples)), 1.5)
        # Gusts should fluctuate
        self.assertGreater(np.std(samples), 0.01)

    def test_weather_field_interface(self):
        field = WeatherField(scenario_id="monsoon_coastal", seed=42)
        pos = np.array([5.0, 5.0, 2.0], dtype=np.float32)

        state = field.get_state(pos, time_s=0.0)
        self.assertIn("mean_wind", state)
        self.assertIn("gust_wind", state)
        self.assertIn("total_wind", state)
        self.assertIn("temperature_c", state)
        self.assertIn("air_density", state)

        self.assertEqual(state["total_wind"].shape, (3,))
        self.assertGreater(state["air_density"], 0.5)

        lookahead = field.get_lookahead_obs(pos)
        self.assertEqual(lookahead.shape, (28,))


if __name__ == "__main__":
    unittest.main()
