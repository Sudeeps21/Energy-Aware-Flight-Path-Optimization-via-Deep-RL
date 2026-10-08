"""
PID Baseline Controller.

A simple cascade PID controller that flies straight toward each waypoint in
sequence. Used as the deterministic comparison baseline vs. the RL agent.

The controller does NOT have access to zone wind — it's a naive
"go toward waypoint" policy. This intentionally understates the baseline
to make the RL agent's environment-awareness more visible in the results.
"""

import numpy as np
from typing import Optional


class PIDController:
    """Single-axis PID controller."""

    def __init__(self, kp: float, ki: float, kd: float, clamp: float = 1.0):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.clamp = clamp
        self._integral  = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self._integral   += error * dt
        derivative        = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error  = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.clamp, self.clamp))


class WaypointPIDAgent:
    """
    Cascade PID agent that navigates toward waypoints sequentially.

    This is the baseline policy evaluated against the PPO agent.
    """

    WAYPOINTS = np.array([
        [15.0, 5.0,  3.0],
        [15.0, 15.0, 3.0],
        [2.0,  2.0,  1.0],
    ], dtype=np.float32)

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.1,
        waypoint_radius: float = 1.0,
    ):
        self.waypoint_radius = waypoint_radius
        self._pids = [PIDController(kp, ki, kd) for _ in range(3)]
        self._wp_idx = 0
        self._dt     = 0.02   # must match env DT

    def reset(self):
        for pid in self._pids:
            pid.reset()
        self._wp_idx = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action given environment observation.

        Parameters
        ----------
        obs : np.ndarray
            22-dimensional observation from MultiFactorDroneEnv.

        Returns
        -------
        np.ndarray
            3-D action vector in [-1, 1].
        """
        # Extract position from normalised obs ([0:3] * WORLD_SIZE)
        WORLD_SIZE = np.array([20.0, 20.0, 10.0], dtype=np.float32)
        pos = obs[:3] * WORLD_SIZE

        # Advance waypoint index if close enough
        if self._wp_idx < len(self.WAYPOINTS):
            wp  = self.WAYPOINTS[self._wp_idx]
            dist = np.linalg.norm(pos - wp)
            if dist < self.waypoint_radius:
                self._wp_idx += 1

        if self._wp_idx >= len(self.WAYPOINTS):
            return np.zeros(3, dtype=np.float32)   # mission done — hover

        wp    = self.WAYPOINTS[self._wp_idx]
        error = wp - pos

        action = np.array([
            self._pids[i].compute(error[i], self._dt) for i in range(3)
        ], dtype=np.float32)

        return np.clip(action, -1.0, 1.0)
