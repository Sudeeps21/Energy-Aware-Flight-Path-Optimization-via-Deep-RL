"""
MultiFactorDroneEnv — Custom Gymnasium environment for multi-factor
drone delivery RL research.

Key features (matching the 3-factor core from the implementation plan):
  1. Spatial wind zones  — per-cell wind vector injected as extra force
  2. Temperature-aware battery — capacity reduced in cold zones
  3. Dynamic payload     — mass changes at pick-up / delivery waypoints

Bonus (free, reuses temp pipeline):
  4. Altitude / air density — hover power increases with altitude

The environment does NOT use PyBullet directly — it implements its own
3-D point-mass physics. This design choice makes the environment:
  - Zero-dependency (no pybullet install required to train)
  - Fully deterministic (reproducible, debuggable)
  - Fast (~10-20x faster than PyBullet for large sweeps)
  - Easy to swap for PyBullet later if GUI visualisation is needed

Physics summary
---------------
  pos_next  = pos  + vel * dt
  vel_next  = vel  + acc * dt
  acc       = (thrust_vector / mass) + gravity + wind_drag_acc

Where thrust_vector is the RL action scaled to the drone's thrust envelope.
"""

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from typing import Optional, Tuple, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_energy.physics.energy import (
    DroneParams,
    BatteryState,
    flight_power,
    hover_power,
    GRAVITY,
)
from drone_energy.weather.zones import WindZoneMap, make_scenario


# ─── Environment constants ────────────────────────────────────────────────────

WORLD_SIZE  = np.array([20.0, 20.0, 10.0], dtype=np.float32)   # metres
DT          = 0.02                                               # seconds (50 Hz)
MAX_EPISODE_STEPS = 2000                                         # ~40 s
MAX_SPEED   = 10.0                                               # m/s clamp
MAX_THRUST  = 30.0                                               # N total (4 rotors)
PAYLOAD_MASS_DEFAULT = 0.3                                       # kg at pickup


class MultiFactorDroneEnv(gym.Env):
    """
    Gymnasium-compatible environment implementing the 3-factor drone RL setup.

    Observation space (22 floats)
    ─────────────────────────────
      [0:3]   position  (x, y, z)         normalised to [0, 1]
      [3:6]   velocity  (vx, vy, vz)      normalised by MAX_SPEED
      [6:9]   wind vector of current cell (m/s)
      [9]     zone temperature (°C)
      [10]    battery state-of-charge     [0, 1]
      [11]    payload carried             {0, 1}
      [12:15] vector to next waypoint     normalised
      [15]    distance to next waypoint   normalised
      [16]    altitude (z)                normalised
      [17:20] previous action             (for smooth control signal)
      [20]    time remaining              normalised to [0, 1]
      [21]    episode step fraction

    Action space (3 floats, continuous)
    ────────────────────────────────────
      [0]  desired acceleration x  [-1, 1]
      [1]  desired acceleration y  [-1, 1]
      [2]  desired acceleration z  [-1, 1]

    The agent controls acceleration directly (simplified thrust allocation).
    Max acceleration = MAX_THRUST / drone_mass.

    Reward
    ──────
      +20   on reaching each waypoint
      +100  on completing the full mission
      -10   on battery depletion (crash)
      -10   on leaving the bounding box
       -energy_used * 50  (per step, in Wh)   — efficiency incentive
       -0.1              (per step)            — time pressure
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        scenario: str = "calm",
        drone_params: Optional[DroneParams] = None,
        payload_mass: float = PAYLOAD_MASS_DEFAULT,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        randomise_scenario: bool = False,
    ):
        super().__init__()

        self.scenario           = scenario
        self.drone              = drone_params or DroneParams()
        self.payload_mass_init  = payload_mass
        self.render_mode        = render_mode
        self._seed              = seed
        self.randomise_scenario = randomise_scenario

        # ── Spaces ───────────────────────────────────────────────────────
        obs_dim = 22
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # ── Mission waypoints ─────────────────────────────────────────────
        # Waypoint 0 = pickup point, Waypoint 1 = delivery, Waypoint 2 = home
        self._waypoints_template = np.array([
            [15.0, 5.0,  3.0],   # pickup
            [15.0, 15.0, 3.0],   # delivery
            [2.0,  2.0,  1.0],   # home / landing
        ], dtype=np.float32)

        # ── Internal state (initialised in reset) ─────────────────────────
        self._pos:          np.ndarray = np.zeros(3)
        self._vel:          np.ndarray = np.zeros(3)
        self._battery:      Optional[BatteryState] = None
        self._zone_map:     Optional[WindZoneMap]  = None
        self._step_count:   int = 0
        self._waypoint_idx: int = 0
        self._payload:      float = 0.0
        self._prev_action:  np.ndarray = np.zeros(3)
        self._total_energy_wh: float = 0.0
        self._mission_complete: bool = False
        self._rng = np.random.default_rng(seed)

        # Logging
        self.episode_stats: Dict[str, Any] = {}

    # ─── Reset ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Rebuild zone map
        scenario = "random" if self.randomise_scenario else self.scenario
        self._zone_map = make_scenario(
            scenario,
            grid_shape=(4, 4, 2),
            world_size=tuple(WORLD_SIZE),
            seed=int(self._rng.integers(0, 2**31)),
        )

        # Initial position — drone starts near origin
        self._pos = np.array([2.0, 2.0, 1.0], dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)

        # Get starting cell temperature for battery initialisation
        cell      = self._zone_map.get_cell(self._pos)
        self._battery = BatteryState(self.drone, cell.temperature_c)

        self._step_count    = 0
        self._waypoint_idx  = 0
        self._payload       = 0.0          # must pick up at waypoint 0
        self._prev_action   = np.zeros(3, dtype=np.float32)
        self._total_energy_wh = 0.0
        self._mission_complete = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._step_count += 1

        # ── 1. Physics ────────────────────────────────────────────────────
        current_mass = self.drone.mass + self._payload
        max_acc      = MAX_THRUST / current_mass
        thrust_acc   = action * max_acc           # desired acceleration

        cell         = self._zone_map.get_cell(self._pos)
        wind         = cell.wind_vector.astype(np.float64)
        temperature  = cell.temperature_c

        # Aerodynamic drag on body (opposes relative airspeed)
        rel_vel      = self._vel - wind[:3]
        drag_acc     = -0.5 * 0.3 * 0.04 / current_mass * np.linalg.norm(rel_vel) * rel_vel

        # Net acceleration
        gravity_acc  = np.array([0.0, 0.0, -GRAVITY])
        total_acc    = thrust_acc + gravity_acc + drag_acc

        # Integrate
        self._vel += total_acc * DT
        self._vel  = np.clip(self._vel, -MAX_SPEED, MAX_SPEED)
        self._pos += self._vel * DT

        # ── 2. Energy draw ────────────────────────────────────────────────
        power = flight_power(
            drone=self.drone,
            velocity_ms=self._vel,
            wind_vector=wind[:3],
            altitude_m=float(self._pos[2]),
            temperature_c=temperature,
            payload_mass=self._payload,
        )
        used_wh = self._battery.consume(power, DT)
        self._total_energy_wh += used_wh

        # ── 3. Waypoint logic ─────────────────────────────────────────────
        reward         = 0.0
        waypoint_hit   = False
        terminated     = False
        truncated      = False

        if self._waypoint_idx < len(self._waypoints_template):
            wp_target = self._waypoints_template[self._waypoint_idx]
            dist      = float(np.linalg.norm(self._pos - wp_target))

            if dist < 1.0:    # within 1 m — waypoint reached
                waypoint_hit = True
                reward += 20.0

                # Waypoint 0 = pickup
                if self._waypoint_idx == 0:
                    self._payload = self.payload_mass_init

                # Waypoint 1 = delivery (drop payload)
                elif self._waypoint_idx == 1:
                    self._payload = 0.0

                # Waypoint 2 = mission complete
                elif self._waypoint_idx == 2:
                    self._mission_complete = True
                    reward += 100.0
                    terminated = True

                self._waypoint_idx += 1

        # ── 4. Termination conditions ─────────────────────────────────────
        out_of_bounds = bool(
            np.any(self._pos < -1.0) or
            np.any(self._pos[:2] > WORLD_SIZE[:2] + 1.0) or
            self._pos[2] > WORLD_SIZE[2] + 1.0
        )

        if self._battery.is_depleted:
            reward    -= 10.0
            terminated = True

        if out_of_bounds:
            reward    -= 10.0
            terminated = True

        if self._step_count >= MAX_EPISODE_STEPS:
            truncated = True

        # ── 5. Per-step reward shaping ────────────────────────────────────
        reward -= used_wh * 50.0       # penalise energy use
        reward -= 0.1                  # time penalty

        # Shaping: small reward for moving toward next waypoint
        if self._waypoint_idx < len(self._waypoints_template) and not waypoint_hit:
            wp_target = self._waypoints_template[self._waypoint_idx]
            new_dist  = float(np.linalg.norm(self._pos - wp_target))
            old_dist  = float(np.linalg.norm(
                (self._pos - self._vel * DT) - wp_target
            ))
            reward += (old_dist - new_dist) * 0.5   # progress reward

        self._prev_action = action.copy()

        obs  = self._get_obs()
        info = self._get_info()

        if terminated or truncated:
            info.update({
                "total_energy_wh"  : self._total_energy_wh,
                "mission_complete" : self._mission_complete,
                "waypoints_reached": self._waypoint_idx,
                "steps"            : self._step_count,
                "final_soc"        : self._battery.state_of_charge,
            })
            self.episode_stats = {
                k: v for k, v in info.items()
                if k in ("total_energy_wh", "mission_complete",
                         "waypoints_reached", "steps", "final_soc")
            }

        return obs, reward, terminated, truncated, info

    # ─── Observation ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        cell = self._zone_map.get_cell(self._pos)

        # Next waypoint
        if self._waypoint_idx < len(self._waypoints_template):
            wp     = self._waypoints_template[self._waypoint_idx]
            wp_vec = wp - self._pos
            wp_dist = np.linalg.norm(wp_vec)
            wp_norm = wp_vec / (wp_dist + 1e-6)
        else:
            wp_norm = np.zeros(3, dtype=np.float32)
            wp_dist = 0.0

        obs = np.concatenate([
            self._pos            / WORLD_SIZE,                    # [0:3]  normalised pos
            self._vel            / MAX_SPEED,                     # [3:6]  normalised vel
            cell.wind_vector,                                     # [6:9]  wind m/s
            [cell.temperature_c / 40.0],                         # [9]    temp normalised
            [self._battery.state_of_charge],                     # [10]   SoC
            [float(self._payload > 0)],                          # [11]   payload flag
            wp_norm.astype(np.float32),                          # [12:15] wp direction
            [wp_dist / (np.linalg.norm(WORLD_SIZE) + 1e-6)],    # [15]   wp distance
            [self._pos[2] / WORLD_SIZE[2]],                      # [16]   altitude
            self._prev_action,                                    # [17:20] prev action
            [1.0 - self._step_count / MAX_EPISODE_STEPS],        # [20]   time remaining
            [self._step_count / MAX_EPISODE_STEPS],              # [21]   step fraction
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        return {
            "pos"                : self._pos.tolist(),
            "vel"                : self._vel.tolist(),
            "battery_soc"        : self._battery.state_of_charge,
            "payload_kg"         : self._payload,
            "waypoint_idx"       : self._waypoint_idx,
            "total_energy_wh"    : self._total_energy_wh,
            "scenario"           : self.scenario,
        }

    # ─── Render ───────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            print(
                f"Step {self._step_count:4d} | "
                f"pos={self._pos.round(2)} | "
                f"vel={self._vel.round(2)} | "
                f"SoC={self._battery.state_of_charge:.2f} | "
                f"payload={self._payload:.2f} kg | "
                f"wp={self._waypoint_idx}/{len(self._waypoints_template)}"
            )

    def close(self):
        pass
