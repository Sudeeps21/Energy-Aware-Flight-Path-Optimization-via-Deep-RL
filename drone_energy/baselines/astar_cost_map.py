"""
astar_cost_map.py — Baseline B1: A* Path Planner over a 3D Wind-Energy Cost Map.

Implements Baseline B1 from TEAM_PLAN_TASKS_TIMELINE_RESOURCES.md Section 1.7:
  - Finds the energy-tuned optimal cruise speed v_cruise that minimizes Wh/km.
  - Runs 3D A* search over the spatial wind & temperature grid where edge cost
    equals electrical energy draw (Wh) for each leg.
  - Provides AStarEnergyAgent interface compatible with env.step() evaluation.
"""

import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional

from drone_energy.physics.energy import DroneParams, flight_power, air_density, GRAVITY
from drone_energy.weather.field import WeatherField


def find_energy_tuned_speed(
    drone: DroneParams,
    wind_vector: np.ndarray,
    altitude_m: float = 0.0,
    temperature_c: float = 25.0,
    payload_mass: float = 0.0,
    speed_range: Tuple[float, float] = (2.0, 10.0),
    num_samples: int = 50,
) -> float:
    """
    Compute the cruise speed (m/s) that minimizes energy per unit distance (Wh/km).

    Cost per metre = P_flight(v, wind) / v_ground
    """
    speeds = np.linspace(speed_range[0], speed_range[1], num_samples)
    best_speed = speeds[0]
    min_cost = float("inf")

    headwind = float(np.dot(wind_vector[:2], [1.0, 0.0]))  # reference direction

    for v in speeds:
        v_ground = max(0.1, v - headwind)
        p_elec = flight_power(
            drone=drone,
            velocity_ms=np.array([v, 0.0, 0.0]),
            wind_vector=wind_vector,
            altitude_m=altitude_m,
            temperature_c=temperature_c,
            payload_mass=payload_mass,
        )
        cost_per_m = p_elec / v_ground
        if cost_per_m < min_cost:
            min_cost = cost_per_m
            best_speed = v

    return float(best_speed)


class AStarPlanner3D:
    """
    3D A* Planner operating over spatial energy cost grid.
    """

    def __init__(
        self,
        weather_field: WeatherField,
        drone: DroneParams = None,
        grid_res: float = 2.0,  # 2.0 m grid cell resolution for fast search
    ):
        self.weather_field = weather_field
        self.drone = drone or DroneParams()
        self.grid_res = grid_res
        self.world_size = weather_field.world_size

    def _pos_to_grid(self, pos: np.ndarray) -> Tuple[int, int, int]:
        gx = int(np.clip(pos[0] / self.grid_res, 0, self.world_size[0] / self.grid_res - 1))
        gy = int(np.clip(pos[1] / self.grid_res, 0, self.world_size[1] / self.grid_res - 1))
        gz = int(np.clip(pos[2] / self.grid_res, 0, self.world_size[2] / self.grid_res - 1))
        return (gx, gy, gz)

    def _grid_to_pos(self, grid_idx: Tuple[int, int, int]) -> np.ndarray:
        return np.array([
            (grid_idx[0] + 0.5) * self.grid_res,
            (grid_idx[1] + 0.5) * self.grid_res,
            (grid_idx[2] + 0.5) * self.grid_res,
        ], dtype=np.float32)

    def plan_path(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        payload_mass: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Run 3D A* search to find energy-minimal path from start_pos to goal_pos.
        """
        start_grid = self._pos_to_grid(start_pos)
        goal_grid = self._pos_to_grid(goal_pos)

        open_set = []
        heapq.heappush(open_set, (0.0, start_grid))

        came_from = {}
        g_score = {start_grid: 0.0}

        # 6-connected 3D face neighbors
        neighbors = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        max_nodes = 3000
        nodes_expanded = 0

        while open_set and nodes_expanded < max_nodes:
            nodes_expanded += 1
            _, current = heapq.heappop(open_set)

            if current == goal_grid:
                # Reconstruct path
                path = [self._grid_to_pos(current)]
                curr = current
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(self._grid_to_pos(curr))
                path.reverse()
                path[0] = start_pos.copy()
                path[-1] = goal_pos.copy()
                return path

            curr_pos = self._grid_to_pos(current)

            for dx, dy, dz in neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                
                # Boundary check
                if not (0 <= neighbor[0] < self.world_size[0] / self.grid_res and
                        0 <= neighbor[1] < self.world_size[1] / self.grid_res and
                        0 <= neighbor[2] < self.world_size[2] / self.grid_res):
                    continue

                neighbor_pos = self._grid_to_pos(neighbor)
                dist = float(np.linalg.norm(neighbor_pos - curr_pos))

                # Query wind & temp at cell
                weather = self.weather_field.get_state(curr_pos)
                wind = weather["mean_wind"]
                temp_c = weather["temperature_c"]

                # Leg direction and optimal speed
                leg_dir = (neighbor_pos - curr_pos) / max(1e-6, dist)
                v_cruise = find_energy_tuned_speed(
                    self.drone, wind, curr_pos[2], temp_c, payload_mass
                )
                
                # Leg energy cost = power * dt_leg
                p_elec = flight_power(
                    self.drone, leg_dir * v_cruise, wind, curr_pos[2], temp_c, payload_mass
                )
                dt_leg = dist / max(0.1, v_cruise)
                energy_cost = (p_elec * dt_leg) / 3600.0  # Wh

                tentative_g = g_score[current] + energy_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Heuristic: straight-line distance to goal in Wh
                    h_dist = float(np.linalg.norm(self._grid_to_pos(goal_grid) - neighbor_pos))
                    h_cost = (h_dist * 55.0 / 10.0) / 3600.0  # estimate
                    f_score = tentative_g + h_cost
                    heapq.heappush(open_set, (f_score, neighbor))

        # Fallback to straight line if search budget exceeded
        return [start_pos.copy(), goal_pos.copy()]


class AStarEnergyAgent:
    """
    Baseline B1 Agent Adapter: Uses A* energy path planning + PID guidance along planned path.
    """

    WAYPOINTS = np.array([
        [15.0, 5.0,  3.0],
        [15.0, 15.0, 3.0],
        [2.0,  2.0,  1.0],
    ], dtype=np.float32)

    def __init__(self, weather_field: WeatherField = None, waypoint_radius: float = 1.0):
        self.weather_field = weather_field or WeatherField(scenario_id="calm")
        self.planner = AStarPlanner3D(self.weather_field)
        self.waypoint_radius = waypoint_radius
        self._wp_idx = 0
        self._dt = 0.02
        self._pids = [PIDController(0.5, 0.01, 0.1) for _ in range(3)]
        self.path = []

    def reset(self):
        """Reset agent path and PID state."""
        for pid in self._pids:
            pid.reset()
        self._wp_idx = 0
        self.path = []

    def act(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action in [-1, 1]^3 from observation vector.
        """
        WORLD_SIZE = np.array([20.0, 20.0, 10.0], dtype=np.float32)
        pos = obs[:3] * WORLD_SIZE

        # Plan full path across all mission waypoints on first step
        if not self.path:
            full_path = [pos.copy()]
            for wp in self.WAYPOINTS:
                leg_path = self.planner.plan_path(full_path[-1], wp)
                full_path.extend(leg_path[1:])
            self.path = full_path
            self._wp_idx = 0

        # Advance along planned waypoints
        while self._wp_idx < len(self.path):
            wp = self.path[self._wp_idx]
            if np.linalg.norm(pos - wp) < self.waypoint_radius:
                self._wp_idx += 1
            else:
                break

        if self._wp_idx >= len(self.path):
            return np.zeros(3, dtype=np.float32)

        wp = self.path[self._wp_idx]
        error = wp - pos

        action = np.array([
            self._pids[i].compute(error[i], self._dt) for i in range(3)
        ], dtype=np.float32)

        return np.clip(action, -1.0, 1.0)


class PIDController:
    """Helper PID Controller for baseline guidance."""

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative
