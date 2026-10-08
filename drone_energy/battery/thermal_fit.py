"""
Identify the pack's thermal parameters (C_th, h_a) from a bench log.

Bench protocol (needs a power meter/logger and a thermistor or thermocouple taped to the
MIDDLE of the pack, covered by a little foam so it reads cell temperature, not air):
  1. Soak the pack at a known temperature (e.g. in a room, or a fridge for the cold case).
  2. Discharge at constant power for ~10 min (heating phase), then at very low power or
     rest for ~30 min (cool-down phase).  Log time, current, voltage, pack temperature and
     ambient temperature at >= 1 Hz.  Both phases are needed: heating alone cannot separate
     C_th from h_a.
  3. Repeat with airflow comparable to the drone's (a fan) - h_a depends on airspeed.
Heat is estimated from the log as  Q = I * (OCV(SoC, T) - V_terminal)  (power dissipated in
R0 and the RC branch; it is the quantity the battery model calls heat_w).

    python scripts/fit_thermal_params.py --csv my_log.csv      (real data)
    python scripts/fit_thermal_params.py --selftest            (synthetic check)
"""
import numpy as np
from scipy.optimize import least_squares

from drone_energy.battery.thevenin import TheveninBattery, ThermalParams


def simulate_pack_temp(time_s, heat_w, ambient_c, c_th, h_a, t0):
    """Pack temperature at the log times for piecewise-constant heat (exact per interval,
    identical to the update inside TheveninBattery)."""
    time_s = np.asarray(time_s, float)
    heat_w = np.broadcast_to(np.asarray(heat_w, float), time_s.shape)
    amb = np.broadcast_to(np.asarray(ambient_c, float), time_s.shape)
    out = np.empty_like(time_s)
    out[0] = t0
    for k in range(len(time_s) - 1):
        dt = time_s[k + 1] - time_s[k]
        t_eq = amb[k] + heat_w[k] / h_a
        out[k + 1] = t_eq + (out[k] - t_eq) * np.exp(-dt * h_a / c_th)
    return out


def fit_thermal_params(time_s, heat_w, pack_temp_c, ambient_c, x0=(390.0, 0.75)):
    """Least-squares fit of (c_th, h_a).  Returns (ThermalParams, info dict with rmse [K] and
    1-sigma standard errors).  Large standard errors mean the log does not excite both
    phases well (see the bench protocol above)."""
    time_s = np.asarray(time_s, float)
    pack = np.asarray(pack_temp_c, float)

    def resid(logp):
        c_th, h_a = np.exp(logp)
        return simulate_pack_temp(time_s, heat_w, ambient_c, c_th, h_a, pack[0]) - pack

    sol = least_squares(resid, np.log(x0), x_scale=1.0)
    c_th, h_a = np.exp(sol.x)
    n, dof = len(pack), len(pack) - 2
    sigma2 = np.sum(sol.fun ** 2) / max(dof, 1)
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * sigma2        # covariance of log-parameters
        se = np.sqrt(np.diag(cov)) * np.array([c_th, h_a])       # delta method
    except np.linalg.LinAlgError:
        se = np.array([np.inf, np.inf])
    return ThermalParams(float(c_th), float(h_a)), {
        "rmse_k": float(np.sqrt(np.mean(sol.fun ** 2))), "se_c_th": float(se[0]), "se_h_a": float(se[1])}


def heat_from_log(time_s, current_a, voltage_v, pack_temp_c, soc0=1.0, capacity_ah=5.0, battery=None):
    """Heat power [W] per interval from a measured log:  Q = max(I * (OCV - V), 0)."""
    bat = battery or TheveninBattery.from_npz(thermal=None)
    time_s, i = np.asarray(time_s, float), np.asarray(current_a, float)
    dt = np.diff(time_s, prepend=time_s[0])
    soc = np.clip(soc0 - np.cumsum(i * dt) / (capacity_ah * 3600.0), 0.05, 1.0)
    ocv = np.array([bat._params(s, t)[0] for s, t in zip(soc, pack_temp_c)])
    return np.maximum(i * (ocv - np.asarray(voltage_v, float)), 0.0)


def synthetic_log(true=ThermalParams(c_th=420.0, h_a=0.55), ambient_c=5.0, noise_k=0.05, seed=0):
    """A fake bench log produced by the battery model itself (for self-tests)."""
    rng = np.random.default_rng(seed)
    bat = TheveninBattery.from_npz(thermal=true)
    t, rows = 0.0, []
    while t < 2400.0:
        power = 250.0 if t < 600.0 else 15.0           # heating phase, then cool-down phase
        o = bat.step(power, 1.0, ambient_c)
        rows.append((t, o["current"], o["v_terminal"] + rng.normal(0, 0.01),
                     o["cell_temp"] + rng.normal(0, noise_k), ambient_c))
        t += 1.0
    return np.array(rows).T      # time, current, voltage, pack_temp, ambient
