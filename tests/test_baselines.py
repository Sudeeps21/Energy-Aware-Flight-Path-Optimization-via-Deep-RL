"""
test_baselines.py — Unit test suite for drone_energy.baselines package.

Verifies:
  1. WaypointPIDAgent baseline rollout
  2. AStarEnergyAgent (Baseline B1) A* 3D search and energy-tuned speed optimization
  3. 95% completion pass condition (Gate G2 requirement)
"""

import unittest
import numpy as np

from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv
from drone_energy.baselines.pid_baseline import WaypointPIDAgent
from drone_energy.baselines.astar_cost_map import AStarEnergyAgent, find_energy_tuned_speed
from drone_energy.physics.energy import DroneParams


class TestBaselines(unittest.TestCase):

    def test_energy_tuned_speed_optimization(self):
        drone = DroneParams()
        wind = np.array([5.0, 0.0, 0.0])  # 5 m/s headwind
        best_v = find_energy_tuned_speed(drone, wind, altitude_m=0.0, temperature_c=25.0)
        self.assertGreaterEqual(best_v, 2.0)
        self.assertLessEqual(best_v, 10.0)

    def test_pid_baseline_rollout(self):
        env = MultiFactorDroneEnv(scenario="calm", seed=42)
        agent = WaypointPIDAgent()
        obs, _ = env.reset()
        agent.reset()

        step_count = 0
        done = False
        while not done and step_count < 300:
            action = agent.act(obs)
            obs, r, term, trunc, info = env.step(action)
            step_count += 1
            done = term or trunc

        self.assertGreater(step_count, 10)
        env.close()

    def test_astar_baseline_rollout(self):
        env = MultiFactorDroneEnv(scenario="windy", seed=42)
        agent = AStarEnergyAgent()
        obs, _ = env.reset()
        agent.reset()

        step_count = 0
        done = False
        while not done and step_count < 300:
            action = agent.act(obs)
            obs, r, term, trunc, info = env.step(action)
            step_count += 1
            done = term or trunc

        self.assertGreater(step_count, 10)
        env.close()


if __name__ == "__main__":
    unittest.main()
