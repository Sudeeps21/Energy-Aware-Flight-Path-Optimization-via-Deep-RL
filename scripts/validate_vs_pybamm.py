"""
Validate TheveninBattery against the full PyBaMM SPMe (Chen2020) model that the
lookup table was extracted from.  PyBaMM is the "ground truth" here.

Run from anywhere:  python scripts/validate_vs_pybamm.py
Needs pybamm (already used by BatteryTemperatureMatrix.py).  Takes a few minutes.

Outputs: console table + results/battery_validation.png

Scenarios (per cell: pack current == cell current, pack power == 6 x cell power):
  cc1C   constant current 5 A  until 3.0 V/cell
  cc2C   constant current 10 A until 3.0 V/cell
  cp25   constant power 25 W/cell (=150 W pack)
  cp35   constant power 35 W/cell (=210 W pack)  [50 W made PyBaMM's solver fail]
  drone  repeating 5 A (60 s) / 12 A (20 s) / 2 A (30 s) pattern, a hover-climb-descend-like load

Metrics (per-cell millivolts, so they are comparable to cell-level literature):
  RMSE and max voltage error over the common time window,
  time-to-cutoff error [%]  (constant-current / constant-power scenarios).
Pass thresholds below are PROVISIONAL starting points, not published standards.
"""
from pathlib import Path

import numpy as np
import pybamm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_energy.battery.thevenin import TheveninBattery

pybamm.set_logging_level("ERROR")

ROOT = Path(__file__).resolve().parents[1]
N_CELLS = 6
R_WIRING = 0.002            # ohm, added to the pack R0 by BatteryTemperatureMatrix.py
V_CUT_CELL = 3.0
DT = 0.5                    # s, our model's time step
TEMPS_C = [-10, 0, 25, 40]

# provisional acceptance thresholds
TOL_RMSE_MV_CELL = 30.0
TOL_CUTOFF_TIME_PCT = 10.0

DRONE_PATTERN = [(5.0, 60), (12.0, 20), (2.0, 30)]      # (amps, seconds)
DRONE_REPEATS = 20

SCENARIOS = {
    "cc1C":  dict(kind="cc", value=5.0,  steps=["Discharge at 5 A until 3 V"]),
    "cc2C":  dict(kind="cc", value=10.0, steps=["Discharge at 10 A until 3 V"]),
    "cp25":  dict(kind="cp", value=25.0, steps=["Discharge at 25 W until 3 V"]),
    "cp35":  dict(kind="cp", value=35.0, steps=["Discharge at 35 W until 3 V"]),
    "drone": dict(kind="profile", value=DRONE_PATTERN,
                  steps=[f"Discharge at {i} A for {d} seconds"
                         for _ in range(DRONE_REPEATS) for i, d in DRONE_PATTERN]),
}


def make_experiment(steps):
    try:
        return pybamm.Experiment(steps, period="1 second")
    except TypeError:                       # older/newer PyBaMM without `period`
        return pybamm.Experiment(steps)


def solve_pybamm(temp_c, steps):
    params = pybamm.ParameterValues("Chen2020")
    tk = temp_c + 273.15
    params.update({"Ambient temperature [K]": tk, "Initial temperature [K]": tk})
    sim = pybamm.Simulation(pybamm.lithium_ion.SPMe(), parameter_values=params,
                            experiment=make_experiment(steps))
    sol = sim.solve(initial_soc=1.0)
    t = np.asarray(sol["Time [s]"].entries)
    v = np.asarray(sol["Terminal voltage [V]"].entries)
    i = np.asarray(sol["Current [A]"].entries)
    # same series-pack equivalent our table was built with
    return t - t[0], N_CELLS * v - i * R_WIRING, i


def current_at(pattern, t):
    cycle = sum(d for _, d in pattern)
    tc = t % cycle
    for amps, dur in pattern:
        if tc < dur:
            return amps
        tc -= dur
    return 0.0


def simulate_ours(kind, value, temp_c, t_max):
    bat = TheveninBattery.from_npz(v_cutoff=N_CELLS * V_CUT_CELL, thermal=None)
    ts, vs, t = [], [], 0.0
    while t < t_max:
        if kind == "cc":
            o = bat.step_current(value, DT, temp_c)
        elif kind == "cp":
            o = bat.step(value * N_CELLS, DT, temp_c)
        else:
            o = bat.step_current(current_at(value, t), DT, temp_c)
        ts.append(t); vs.append(o["v_terminal"])
        t += DT
        if kind != "profile" and o["depleted"]:
            break
    return np.array(ts), np.array(vs)


def main():
    rows, curves = [], {}
    for name, sc in SCENARIOS.items():
        for T in TEMPS_C:
            try:
                tp, vp, ip = solve_pybamm(T, sc["steps"])
            except Exception as exc:               # PyBaMM may fail at harsh corners
                print(f"[skip] {name} @ {T} C: PyBaMM failed ({type(exc).__name__})")
                continue
            # let OUR model run to its own cutoff (up to 1.5x PyBaMM's duration);
            # truncating it at PyBaMM's end time would hide any runtime error.
            t_max = tp[-1] + 1.0 if sc["kind"] == "profile" else 1.5 * tp[-1]
            to, vo = simulate_ours(sc["kind"], sc["value"], T, t_max=t_max)

            t_common = min(tp[-1], to[-1])
            m = to <= t_common
            err = (vo[m] - np.interp(to[m], tp, vp)) / N_CELLS * 1000.0   # mV per cell
            rmse, emax = float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err)))
            t_err = (to[-1] - tp[-1]) / tp[-1] * 100.0 if sc["kind"] != "profile" else float("nan")

            ok = rmse < TOL_RMSE_MV_CELL and (sc["kind"] == "profile" or abs(t_err) < TOL_CUTOFF_TIME_PCT)
            rows.append((name, T, tp[-1], to[-1], t_err, rmse, emax, ok))
            curves[(name, T)] = (tp, vp, to, vo)

    print(f"\n{'scenario':<7} {'T[C]':>5} {'t_pybamm':>9} {'t_ours':>8} {'dt[%]':>7} "
          f"{'RMSE[mV/cell]':>14} {'max[mV/cell]':>13}  verdict")
    for name, T, tpe, toe, terr, rmse, emax, ok in rows:
        print(f"{name:<7} {T:>5} {tpe:>9.0f} {toe:>8.0f} {terr:>7.1f} {rmse:>14.1f} {emax:>13.1f}  "
              f"{'PASS' if ok else 'REVIEW'}")
    n_ok = sum(r[-1] for r in rows)
    print(f"\n{n_ok}/{len(rows)} within provisional thresholds "
          f"(RMSE < {TOL_RMSE_MV_CELL} mV/cell, cutoff time within {TOL_CUTOFF_TIME_PCT}%)")

    # plot
    names = list(SCENARIOS)
    fig, axes = plt.subplots(len(names), len(TEMPS_C), figsize=(4 * len(TEMPS_C), 2.6 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        for c, T in enumerate(TEMPS_C):
            ax = axes[r][c]
            if (name, T) in curves:
                tp, vp, to, vo = curves[(name, T)]
                ax.plot(tp, vp, "-", color="k", label="PyBaMM SPMe")
                ax.plot(to, vo, "--", color="tab:red", label="Thevenin LUT")
            ax.set_title(f"{name} @ {T} C", fontsize=9)
            if c == 0:
                ax.set_ylabel("pack V")
            if r == len(names) - 1:
                ax.set_xlabel("time [s]")
    axes[0][0].legend(fontsize=7)
    plt.tight_layout()
    out = ROOT / "results" / "battery_validation.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
