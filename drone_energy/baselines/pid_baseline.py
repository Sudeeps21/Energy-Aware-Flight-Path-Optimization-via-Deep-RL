"""
PID Baseline Controller.

A simple cascade PID controller that flies straight toward each waypoint in
sequence. Used as the deterministic comparison baseline vs. the RL agent.

The controller reads the active waypoint direction and distance directly from
the observation vector (indices 12–15), so it works correctly with both fixed
and episode-randomised waypoints.

The controller does NOT observe zone wind — it's a naive "go toward waypoint"
policy. This intentionally exposes the RL agent's weather-awareness advantage.
"""

import numpy as np
from typing import Optional

# Must match MultiFactorDroneEnv constants
_WORLD_SIZE = np.array([20.0, 20.0, 10.0], dtype=np.float32)
_DT         = 0.02   # seconds


class PIDController:
    """Single-axis PID controller."""

    def __init__(self, kp: float, ki: float, kd: float, clamp: float = 1.0):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.clamp = clamp
        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self._integral  += error * dt
        derivative       = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return float(np.clip(output, -self.clamp, self.clamp))


class WaypointPIDAgent:
    """
    Cascade PID agent that navigates toward waypoints sequentially.

    Reads waypoint direction (obs[12:15]) and distance (obs[15]) directly from
    the environment observation so it correctly tracks episode-randomised targets.
    """

    def __init__(
        self,
        kp: float = 0.5,
        ki: float = 0.01,
        kd: float = 0.1,
        waypoint_radius: float = 1.0,
    ):
        self.waypoint_radius = waypoint_radius
        self._pids   = [PIDController(kp, ki, kd) for _ in range(3)]
        self._dt     = _DT

    def reset(self):
        for pid in self._pids:
            pid.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action given environment observation.

        Parameters
        ----------
        obs : np.ndarray
            22-dimensional observation from MultiFactorDroneEnv.
              obs[12:15]  unit direction to current active waypoint
              obs[15]     normalised distance to current active waypoint

        Returns
        -------
        np.ndarray
            3-D action vector in [-1, 1].
        """
        # Decode current position
        pos = obs[:3] * _WORLD_SIZE

        # Decode waypoint target from observation
        wp_dir  = obs[12:15].astype(np.float32)          # unit vector to waypoint
        wp_dist = float(obs[15]) * float(np.linalg.norm(_WORLD_SIZE))  # metres
        wp_target = pos + wp_dir * wp_dist

        error = wp_target - pos

        action = np.array(
            [self._pids[i].compute(error[i], self._dt) for i in range(3)],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)
