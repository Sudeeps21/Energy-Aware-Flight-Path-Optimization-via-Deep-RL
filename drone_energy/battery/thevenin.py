"""
FINAL battery model: step-based 1-RC (Thevenin) pack model with a 2D (SoC, temperature)
lookup AND lumped self-heating.  Two different temperature effects are modelled:

  1. ENVIRONMENT temperature (the `temp_c` argument of step(), e.g. the wind-zone temperature):
       - sets the pack's starting temperature (cold-soaked) unless reset(cell_temp=...) is used
       - is the heat-sink temperature the pack exchanges heat with
  2. SELF-HEATING: current flowing through R0 and R1 dissipates heat inside the pack,
       C_th * dT/dt = Q - hA * (T_pack - T_ambient),   Q = I^2*R0 + V1^2/R1   [W]
     and it is the PACK temperature T_pack that is used for the table lookup:
       cold -> higher R0 -> more heat -> warmer pack -> lower R0.

Electrical model (all parameters from the table at the pack temperature):
    V_terminal = OCV(SoC,T) - V1 - I*R0(SoC,T)
    dV1/dt     = -V1/(R1*C1) + I/C1            (exact zero-order-hold update)
    dSoC/dt    = -I / Q                        (coulomb counting)
The load is a constant POWER (motors demand watts, not amps):
    P = (OCV - V1 - I*R0) * I   ->   R0*I^2 - (OCV - V1)*I + P = 0
If the pack cannot deliver P (negative discriminant) the step is flagged as a brownout
and the pack delivers its maximum power instead.

Modelling choices and limits (see ThermalParams for the parameter provenance):
  * Only IRREVERSIBLE heat (I^2*R0 + V1^2/R1) is generated.  Reversible (entropic) heat is
    omitted: the lookup table's OCV has no temperature dependence, so there is no entropic
    information to use.
  * Single thermal node (no core/surface gradient), constant h_a (no dependence on airspeed).
  * Table range is SoC 0.05..1.0 and -20..60 C; lookups are clamped, and the returned
    `overtemp` flag tells the caller when the pack is above the table's upper limit.
  * thermal=None switches self-heating off (pack temperature = ambient), which is the mode
    used for the PyBaMM validation (PyBaMM was run isothermal).
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalParams:
    """Lumped single-node thermal model of the pack.  BOTH VALUES ARE ASSUMPTIONS until
    measured (see scripts/fit_thermal_params.py for how to identify them from a bench log).

    c_th : heat capacity [J/K] = n_cells * cell_mass * specific_heat.
           Default: 6 cells x ~69 g x ~950 J/(kg K) ~ 390 J/K  (check the cell datasheet).
    h_a  : heat loss to air [W/K] = convection coefficient x exposed area.  Depends on the
           pack housing and the propeller airflow; 0.3-1.5 W/K is a plausible range, so
           treat it as a sensitivity parameter, not a fact.
    """
    c_th: float = 390.0
    h_a: float = 0.75

    @classmethod
    def from_cells(cls, n_cells=6, cell_mass_kg=0.069, cp_j_per_kg_k=950.0, h_a=0.75):
        return cls(c_th=n_cells * cell_mass_kg * cp_j_per_kg_k, h_a=h_a)


DEFAULT_THERMAL = ThermalParams()
T_PACK_MAX_C = 60.0       # upper end of the lookup table; above it the lookup is clamped


# drone_energy_rl/drone_energy/battery/thevenin.py -> parents[2] = drone_energy_rl/
DEFAULT_LUT = (Path(__file__).resolve().parents[2]
               / "data" / "battery_luts" / "battery_6s_2d_lut_clean.npz")


class TheveninBattery:
    def __init__(self, capacity_ah, soc_grid, temp_grid, ocv, r0, r1, c1,
                 v_cutoff=18.0, thermal=None):
        """
        capacity_ah : pack capacity in Ah (6S1P of 5 Ah cells -> 5.0)
        *_grid      : 1D axes (SoC as 0..1 fraction, temperature in deg C)
        ocv, r0, r1, c1 : pack-level tables, shape (len(soc_grid), len(temp_grid))
        v_cutoff    : terminal voltage [V] below which the pack counts as empty
                      (3.0 V/cell x 6)
        """
        self.q = float(capacity_ah) * 3600.0          # coulombs
        self.v_cutoff = float(v_cutoff)
        self.thermal = thermal
        self.soc_grid = np.asarray(soc_grid, dtype=float)
        self.temp_grid = np.asarray(temp_grid, dtype=float)

        # One interpolator for all four tables -> one call per step (fast).
        stacked = np.stack([ocv, r0, r1, c1], axis=-1)      # (nsoc, ntemp, 4)
        self._lut = RegularGridInterpolator(
            (self.soc_grid, self.temp_grid), stacked, method="linear",
            bounds_error=False, fill_value=None)

        self.reset()

    @classmethod
    def from_npz(cls, path=DEFAULT_LUT, capacity_ah=5.0, v_cutoff=18.0, thermal=DEFAULT_THERMAL):
        """Loads the clean lookup table.  Self-heating is ON by default; pass thermal=None for an
        isothermal pack (temperature argument = pack temperature)."""
        d = np.load(path)
        return cls(capacity_ah, d["soc"], d["temp"], d["ocv"], d["r0"],
                   d["r1"], d["c1"], v_cutoff=v_cutoff, thermal=thermal)

    # ------------------------------------------------------------------ state
    def reset(self, soc=1.0, v1=0.0, cell_temp=None):
        """cell_temp: initial pack temperature [C] for the thermal model (None = equal to the
        ambient of the first step, i.e. a cold/heat-soaked pack)."""
        self.soc = float(np.clip(soc, 0.0, 1.0))
        self.v1 = float(v1)
        self.energy_out_wh = 0.0
        self.heat_j = 0.0               # total heat generated inside the pack [J]
        self.cell_temp = None if cell_temp is None else float(cell_temp)   # pack temperature [C]
        self.v_terminal = None
        self.depleted = False
        return self.soc

    def _params(self, soc, temp_c):
        """OCV, R0, R1, C1 at (soc, temp). Inputs are clamped to the table range
        (SoC 0.05..1.0, T -20..60 C) instead of extrapolating."""
        s = np.clip(soc, self.soc_grid[0], self.soc_grid[-1])
        t = np.clip(temp_c, self.temp_grid[0], self.temp_grid[-1])
        ocv, r0, r1, c1 = self._lut([[s, t]])[0]
        return float(ocv), float(r0), float(r1), float(c1)

    def max_power(self, temp_c):
        """Maximum power [W] the pack can deliver right now: (OCV-V1)^2 / (4*R0).
        With the thermal model on, `temp_c` (ambient) is ignored once the pack temperature is known."""
        if self.thermal is not None and self.cell_temp is not None:
            temp_c = self.cell_temp       # use the pack's own temperature, not the ambient
        ocv, r0, _, _ = self._params(self.soc, temp_c)
        e = ocv - self.v1
        return e * e / (4.0 * r0)

    # ------------------------------------------------------------------- step
    def step(self, power_w, dt, temp_c):
        """
        Advance the battery by dt seconds under a constant electrical power draw.

        Returns a dict:
            v_terminal [V], current [A], soc [0..1], v1 [V],
            power_delivered [W], brownout (bool), depleted (bool)
        """
        power_w = max(float(power_w), 0.0)
        amb = temp_c
        temp_c = self._lookup_temp(temp_c)
        ocv, r0, r1, c1 = self._params(self.soc, temp_c)

        e = ocv - self.v1
        disc = e * e - 4.0 * r0 * power_w
        brownout = disc < 0.0
        if brownout:
            disc = 0.0                      # deliver maximum power only
        i = (e - np.sqrt(disc)) / (2.0 * r0)
        return self._advance(i, dt, ocv, r0, r1, c1, brownout, amb)

    def step_current(self, current_a, dt, temp_c):
        """Same as step() but for a prescribed current [A] (used for validation
        against constant-current references). Negative current = charging."""
        temp_c_amb = temp_c
        temp_c = self._lookup_temp(temp_c)
        ocv, r0, r1, c1 = self._params(self.soc, temp_c)
        return self._advance(float(current_a), dt, ocv, r0, r1, c1, False, temp_c_amb)

    def _lookup_temp(self, ambient_c):
        """Temperature used for the table lookup: the pack's own (thermal on) or ambient."""
        if self.thermal is None:
            return ambient_c
        if self.cell_temp is None:
            self.cell_temp = float(ambient_c)
        return self.cell_temp

    def _advance(self, i, dt, ocv, r0, r1, c1, brownout, ambient_c):
        v_t = ocv - self.v1 - i * r0
        p_del = v_t * i
        q_heat = i * i * r0 + self.v1 * self.v1 / r1          # dissipated in R0 and R1 [W]
        self.heat_j += q_heat * dt
        if self.thermal is not None:
            th = self.thermal
            if th.h_a > 0.0:       # exact solution for constant Q over the step (always stable)
                t_eq = ambient_c + q_heat / th.h_a
                self.cell_temp = t_eq + (self.cell_temp - t_eq) * np.exp(-dt * th.h_a / th.c_th)
            else:                  # adiabatic
                self.cell_temp += dt * q_heat / th.c_th

        # state update (SoC and V1 use this step's parameters, as in Models 2.3/2.4)
        alpha = np.exp(-dt / (r1 * c1))
        self.v1 = self.v1 * alpha + i * r1 * (1.0 - alpha)
        self.soc = float(np.clip(self.soc - i * dt / self.q, 0.0, 1.0))
        self.energy_out_wh += p_del * dt / 3600.0
        self.v_terminal = v_t
        self.depleted = bool(v_t < self.v_cutoff or self.soc <= 0.0)

        pack_temp = self.cell_temp if self.thermal is not None else ambient_c
        return {
            "v_terminal": v_t, "current": i, "soc": self.soc, "v1": self.v1,
            "power_delivered": p_del, "brownout": brownout,
            "depleted": self.depleted, "heat_w": q_heat,
            "cell_temp": pack_temp,
            "overtemp": bool(pack_temp > T_PACK_MAX_C),
        }
