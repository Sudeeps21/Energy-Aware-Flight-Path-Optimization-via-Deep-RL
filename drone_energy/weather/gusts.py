"""
gusts.py — Dryden-style Gauss-Markov turbulence and gust model for drone simulation.

Based on MIL-HDBK-1797 flying qualities standard:
  u_g[k+1] = a * u_g[k] + sigma_g * sqrt(1 - a^2) * w[k]
where:
  a = exp(-dt / tau), tau = L_scale / airspeed
  w[k] ~ N(0, 1) standard normal process noise
"""

import numpy as np


class DrydenGustModel:
    """
    First-order Gauss-Markov stochastic turbulence process per 3D axis.
    """

    def __init__(
        self,
        dt: float = 0.02,               # physics timestep (s)
        mean_wind_speed: float = 3.0,   # m/s
        gust_factor: float = 1.3,       # peak gust over mean speed
        scale_length: float = 100.0,    # metres (MIL-HDBK-1797 low-altitude scale)
        seed: int = 42,
    ):
        self.dt = dt
        self.mean_wind_speed = max(mean_wind_speed, 0.5)
        self.gust_factor = gust_factor
        self.scale_length = scale_length
        self._rng = np.random.RandomState(seed)

        # Gust intensity sigma_g per axis
        # Peak 3-sigma gust addition = mean_wind * (gust_factor - 1.0)
        max_addition = self.mean_wind_speed * max(0.0, gust_factor - 1.0)
        self.sigma_g = max_addition / 3.0

        # Time constant tau = L / V
        self.tau = scale_length / self.mean_wind_speed
        self.a = float(np.exp(-self.dt / self.tau))

        # Current gust state vector (3D m/s)
        self.state = np.zeros(3, dtype=np.float32)

    def reset(self, seed: int = None):
        """Reset gust process state."""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.state = np.zeros(3, dtype=np.float32)

    def step(self) -> np.ndarray:
        """
        Advance turbulence process by one timestep dt.

        Returns
        -------
        np.ndarray
            3D gust velocity vector (m/s) in world frame.
        """
        noise = self._rng.normal(0.0, 1.0, size=3).astype(np.float32)
        gain = np.sqrt(1.0 - self.a ** 2)
        self.state = self.a * self.state + self.sigma_g * gain * noise
        return self.state.copy()
