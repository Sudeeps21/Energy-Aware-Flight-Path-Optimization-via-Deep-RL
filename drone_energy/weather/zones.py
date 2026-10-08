"""
Wind Zone System — spatial grid of wind vectors and temperatures.

Each cell in the grid has:
  - wind_vector : np.ndarray (3,) — (vx, vy, vz) in m/s
  - temperature : float           — degrees Celsius

The agent's observation includes the current cell's wind and temperature,
plus optionally neighbour cells (forecast uncertainty).

ERA5/NOAA-grounded default values are used so the environment reflects
realistic regional weather rather than arbitrary numbers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class ZoneCell:
    """One grid cell in the wind-zone map."""
    wind_vector: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temperature_c: float = 25.0

    def __post_init__(self):
        self.wind_vector = np.asarray(self.wind_vector, dtype=np.float32)


class WindZoneMap:
    """
    Axis-aligned 3-D grid of ZoneCells, spanning an (x, y, z) bounding box.

    Coordinates are in metres (world frame matches gym-pybullet-drones).

    Parameters
    ----------
    grid_shape   : (nx, ny, nz) — number of cells per axis
    world_size   : (Lx, Ly, Lz) — total physical size of the environment (m)
    seed         : optional random seed for reproducible generation
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (4, 4, 2),
        world_size: Tuple[float, float, float] = (20.0, 20.0, 10.0),
        seed: Optional[int] = None,
    ):
        self.grid_shape = grid_shape
        self.world_size = np.array(world_size, dtype=np.float32)
        self.cell_size  = self.world_size / np.array(grid_shape, dtype=np.float32)

        nx, ny, nz = grid_shape
        self._cells: list = [
            [
                [ZoneCell() for _ in range(nz)]
                for _ in range(ny)
            ]
            for _ in range(nx)
        ]

        self._rng = np.random.default_rng(seed)

    # ── Cell access ────────────────────────────────────────────────────────

    def cell_index(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """
        Map a world-frame position to a grid-cell index.

        Positions outside the grid are clamped to border cells.
        """
        pos = np.asarray(pos, dtype=float)
        idx = (pos / self.cell_size).astype(int)
        nx, ny, nz = self.grid_shape
        ix = int(np.clip(idx[0], 0, nx - 1))
        iy = int(np.clip(idx[1], 0, ny - 1))
        iz = int(np.clip(idx[2], 0, nz - 1))
        return ix, iy, iz

    def get_cell(self, pos: np.ndarray) -> ZoneCell:
        ix, iy, iz = self.cell_index(pos)
        return self._cells[ix][iy][iz]

    def set_cell(self, ix: int, iy: int, iz: int, cell: ZoneCell):
        self._cells[ix][iy][iz] = cell

    # ── Preset scenarios ───────────────────────────────────────────────────

    def load_calm(self):
        """Calm scenario — light uniform breeze, 20 °C."""
        for ix in range(self.grid_shape[0]):
            for iy in range(self.grid_shape[1]):
                for iz in range(self.grid_shape[2]):
                    self._cells[ix][iy][iz] = ZoneCell(
                        wind_vector=np.array([1.0, 0.5, 0.0]),
                        temperature_c=20.0,
                    )

    def load_windy(self):
        """Windy scenario — strong crosswind gradient, cooler temps at altitude."""
        nx, ny, nz = self.grid_shape
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    # Wind increases with altitude and x-position (gradient)
                    wind_x = 3.0 + ix * 1.2 + iz * 0.8
                    wind_y = 1.0 + iy * 0.5
                    temp   = 15.0 - iz * 3.0   # colder at altitude
                    self._cells[ix][iy][iz] = ZoneCell(
                        wind_vector=np.array([wind_x, wind_y, 0.0]),
                        temperature_c=temp,
                    )

    def load_cold(self):
        """Cold scenario — moderate wind, sub-zero temperatures to stress battery."""
        nx, ny, nz = self.grid_shape
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    self._cells[ix][iy][iz] = ZoneCell(
                        wind_vector=np.array([2.0, 1.0, 0.0]),
                        temperature_c=-10.0 - iz * 2.0,
                    )

    def load_random(
        self,
        wind_mag_range: Tuple[float, float] = (0.0, 8.0),
        temp_range: Tuple[float, float] = (-15.0, 35.0),
        turbulence_std: float = 0.5,
    ):
        """
        Randomised zone map — useful for curriculum learning and ablations.

        Wind has a smooth base component plus Gaussian turbulence noise.
        Temperature decreases with altitude (realistic lapse rate).
        """
        nx, ny, nz = self.grid_shape
        # Base wind field (spatially smooth)
        base_wx = self._rng.uniform(*wind_mag_range)
        base_wy = self._rng.uniform(-wind_mag_range[1] * 0.5, wind_mag_range[1] * 0.5)
        base_temp = self._rng.uniform(*temp_range)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    noise = self._rng.normal(0, turbulence_std, size=3).astype(np.float32)
                    wv    = np.array([base_wx, base_wy, 0.0], dtype=np.float32) + noise
                    temp  = base_temp - iz * 2.5 + self._rng.normal(0, 1.5)
                    self._cells[ix][iy][iz] = ZoneCell(
                        wind_vector=wv,
                        temperature_c=float(temp),
                    )

    def load_card(self, card_or_id):
        """Load wind and temperature zones dynamically from a ScenarioCard."""
        from drone_energy.weather.cards import ScenarioCard
        card = card_or_id if isinstance(card_or_id, ScenarioCard) else ScenarioCard.load(card_or_id)

        nx, ny, nz = self.grid_shape
        spd = card.mean_wind_speed_ms
        ang = card.primary_wind_dir_rad
        wx = spd * np.cos(ang)
        wy = spd * np.sin(ang)

        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    shear_mult = 1.0 + iz * 0.15
                    cell_wx = wx * shear_mult + self._rng.normal(0, card.std_wind_speed_ms * 0.2)
                    cell_wy = wy * shear_mult + self._rng.normal(0, card.std_wind_speed_ms * 0.2)
                    cell_wz = self._rng.normal(0, 0.05)
                    cell_temp = card.mean_temperature_c - iz * 1.5 + self._rng.normal(0, 0.5)
                    self._cells[ix][iy][iz] = ZoneCell(
                        wind_vector=np.array([cell_wx, cell_wy, cell_wz], dtype=np.float32),
                        temperature_c=float(cell_temp),
                    )

    # ── Observation helper ─────────────────────────────────────────────────

    def get_local_obs(self, pos: np.ndarray) -> np.ndarray:
        """
        Return a flat observation vector for the current cell:
            [wx, wy, wz, temperature_c]   (4 floats)
        """
        cell = self.get_cell(pos)
        return np.array(
            [*cell.wind_vector, cell.temperature_c],
            dtype=np.float32,
        )

    def get_neighbourhood_obs(self, pos: np.ndarray) -> np.ndarray:
        """
        Return observations for the current cell plus its 6 face-adjacent
        neighbours (forecast context, 7 * 4 = 28 floats).

        Missing neighbours (at map boundaries) repeat the current cell value.
        """
        cx, cy, cz = self.cell_index(pos)
        nx, ny, nz = self.grid_shape

        offsets = [
            (0, 0, 0),   # current
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]

        obs_parts = []
        for dx, dy, dz in offsets:
            ix = int(np.clip(cx + dx, 0, nx - 1))
            iy = int(np.clip(cy + dy, 0, ny - 1))
            iz = int(np.clip(cz + dz, 0, nz - 1))
            c  = self._cells[ix][iy][iz]
            obs_parts.append([*c.wind_vector, c.temperature_c])

        return np.array(obs_parts, dtype=np.float32).flatten()


# ── Convenience factory functions ──────────────────────────────────────────────

def make_scenario(
    name: str,
    grid_shape=(4, 4, 2),
    world_size=(20.0, 20.0, 10.0),
    seed: Optional[int] = None,
) -> WindZoneMap:
    """
    Factory that returns a pre-configured WindZoneMap by scenario name or card ID.
    """
    zone_map = WindZoneMap(grid_shape=grid_shape, world_size=world_size, seed=seed)
    loaders = {
        "calm"   : zone_map.load_calm,
        "windy"  : zone_map.load_windy,
        "cold"   : zone_map.load_cold,
        "random" : zone_map.load_random,
    }
    if name in loaders:
        loaders[name]()
    else:
        try:
            zone_map.load_card(name)
        except Exception as e:
            raise ValueError(f"Unknown scenario '{name}'. Choose from {list(loaders.keys())} or valid scenario cards. ({e})")
    return zone_map
