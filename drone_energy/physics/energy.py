"""
Energy model for multi-factor drone simulation.

Covers three tightly-coupled physical mechanisms:
  1. Thrust-to-power (quadrotor power equation)
  2. Temperature-aware battery capacity (Peukert / temperature derating)
  3. Altitude / air-density correction (ISA standard atmosphere)

All units are SI unless noted.
"""

import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────
RHO_SEA_LEVEL   = 1.225      # kg/m³  — air density at sea level, 15 °C
GRAVITY         = 9.80665    # m/s²
TEMP_STD_K      = 288.15     # K      — standard sea-level temperature (15 °C)
LAPSE_RATE      = 0.0065     # K/m    — ISA temperature lapse rate
GAS_CONSTANT    = 287.05     # J/kg·K — specific gas constant for dry air

# Battery derating reference: Li-Po characteristic
BATTERY_TEMP_REF     = 25.0   # °C — nominal (full-capacity) temperature
BATTERY_TEMP_MIN     = -20.0  # °C — below this, capacity → ~0 (safety floor added)
BATTERY_CAPACITY_MIN = 0.30   # fraction — max derating floor (30 % of nominal)

# Quadrotor frame (DJI F450-like defaults; override via DroneParams)
DRAG_COEFF_DEFAULT   = 1.3e-5   # N·m / (rad/s)²  — blade drag coefficient k
THRUST_COEFF_DEFAULT = 6.5e-7   # N / (rad/s)²    — blade thrust coefficient b
NUM_ROTORS_DEFAULT   = 4


class DroneParams:
    """Physical parameters of the simulated drone."""

    def __init__(
        self,
        mass: float = 1.0,           # kg  (including nominal payload)
        num_rotors: int = NUM_ROTORS_DEFAULT,
        thrust_coeff: float = THRUST_COEFF_DEFAULT,
        drag_coeff: float = DRAG_COEFF_DEFAULT,
        rotor_radius: float = 0.1,   # m
        battery_capacity_wh: float = 50.0,  # Wh
        battery_voltage: float = 11.1,      # V (3S Li-Po nominal)
    ):
        self.mass               = mass
        self.num_rotors         = num_rotors
        self.thrust_coeff       = thrust_coeff
        self.drag_coeff         = drag_coeff
        self.rotor_radius       = rotor_radius
        self.battery_capacity_wh = battery_capacity_wh
        self.battery_voltage    = battery_voltage

    @property
    def disc_area(self) -> float:
        """Total rotor disc area (m²)."""
        return self.num_rotors * np.pi * self.rotor_radius ** 2


# ─── Air density ──────────────────────────────────────────────────────────────

def air_density(altitude_m: float) -> float:
    """
    International Standard Atmosphere (ISA) air density at a given altitude.

    Parameters
    ----------
    altitude_m : float
        Altitude above sea level in metres (0 to ~11000 m troposphere).

    Returns
    -------
    float
        Air density in kg/m3.
    """
    altitude_m = np.clip(altitude_m, 0.0, 11_000.0)
    T = TEMP_STD_K - LAPSE_RATE * altitude_m
    rho = RHO_SEA_LEVEL * (T / TEMP_STD_K) ** (GRAVITY / (LAPSE_RATE * GAS_CONSTANT) - 1)
    return float(rho)


# ─── Battery capacity ─────────────────────────────────────────────────────────

def battery_capacity_factor(temperature_c: float) -> float:
    """
    Temperature-dependent capacity derating factor for a Li-Po cell.

    Parameters
    ----------
    temperature_c : float
        Ambient temperature in degrees Celsius.

    Returns
    -------
    float
        Fraction of nominal capacity available (0.30 to 1.0).
    """
    if temperature_c >= BATTERY_TEMP_REF:
        return 1.0

    span = BATTERY_TEMP_REF - BATTERY_TEMP_MIN          # 45 degrees C span
    drop = (BATTERY_TEMP_REF - temperature_c) / span
    factor = 1.0 - drop * (1.0 - BATTERY_CAPACITY_MIN)
    return float(np.clip(factor, BATTERY_CAPACITY_MIN, 1.0))


def available_energy_wh(
    drone: DroneParams,
    temperature_c: float,
) -> float:
    """Effective battery energy available at a given temperature (Wh)."""
    return drone.battery_capacity_wh * battery_capacity_factor(temperature_c)


# ─── Thrust-to-power ──────────────────────────────────────────────────────────

def hover_power(
    drone: DroneParams,
    altitude_m: float = 0.0,
    temperature_c: float = 25.0,
    payload_mass: float = 0.0,
) -> float:
    """
    Power required for steady hover (W), accounting for altitude and payload.

    Uses actuator disk theory: P = T^(3/2) / sqrt(2 * rho * A)

    Parameters
    ----------
    drone        : DroneParams
    altitude_m   : float    — current altitude above sea level
    temperature_c: float    — ambient temperature (degrees C)
    payload_mass : float    — additional carried mass (kg)

    Returns
    -------
    float
        Hover power in Watts.
    """
    rho        = air_density(altitude_m)
    total_mass = drone.mass + payload_mass
    thrust     = total_mass * GRAVITY
    A          = drone.disc_area

    power = (thrust ** 1.5) / np.sqrt(2.0 * rho * A)
    return float(power)


def flight_power(
    drone: DroneParams,
    velocity_ms: np.ndarray,
    wind_vector: np.ndarray,
    altitude_m: float = 0.0,
    temperature_c: float = 25.0,
    payload_mass: float = 0.0,
    drag_coeff_body: float = 0.3,
    frontal_area: float = 0.04,
) -> float:
    """
    Estimated total electrical power draw (W) during forward flight.

    Model: P_total = P_hover (altitude+payload corrected) + P_drag

    Parameters
    ----------
    drone          : DroneParams
    velocity_ms    : (3,) array — drone body velocity in world frame (m/s)
    wind_vector    : (3,) array — wind velocity in world frame (m/s)
    altitude_m     : float
    temperature_c  : float
    payload_mass   : float

    Returns
    -------
    float
        Total power draw in Watts.
    """
    rho          = air_density(altitude_m)
    airspeed     = np.asarray(velocity_ms) - np.asarray(wind_vector)
    airspeed_mag = float(np.linalg.norm(airspeed))

    p_hover = hover_power(drone, altitude_m, temperature_c, payload_mass)
    p_drag  = 0.5 * rho * drag_coeff_body * frontal_area * airspeed_mag ** 3

    return float(p_hover + p_drag)


# ─── Energy accounting ────────────────────────────────────────────────────────

class BatteryState:
    """
    Running battery state tracker for one episode.

    Usage
    -----
    bat = BatteryState(drone, temperature_c)
    bat.consume(power_watts, dt_seconds)
    if bat.is_depleted:
        handle_crash()
    """

    def __init__(self, drone: DroneParams, temperature_c: float, initial_soc: float = 1.0):
        self.capacity_wh  = available_energy_wh(drone, temperature_c)
        self.remaining_wh = self.capacity_wh * float(np.clip(initial_soc, 0.0, 1.0))

    def consume(self, power_watts: float, dt_s: float) -> float:
        """Deduct energy used over dt_s seconds. Returns energy consumed (Wh)."""
        used_wh = (power_watts * dt_s) / 3600.0
        self.remaining_wh = max(0.0, self.remaining_wh - used_wh)
        return used_wh

    @property
    def state_of_charge(self) -> float:
        """Battery state-of-charge [0, 1]."""
        if self.capacity_wh == 0:
            return 0.0
        return self.remaining_wh / self.capacity_wh

    @property
    def is_depleted(self) -> bool:
        return self.remaining_wh <= 0.0
