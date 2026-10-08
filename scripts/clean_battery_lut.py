"""
One-time cleanup of the PyBaMM-generated 6S battery lookup table.

Reads : data/battery_luts/battery_6s_2d_lut.npz        (original, never modified)
Writes: data/battery_luts/battery_6s_2d_lut_clean.npz  (same keys, repaired R1/C1)
        results/lut_cleanup.png                        (before/after plot)

Problems fixed (see BatteryTemperatureMatrix.py for how the table was built):
  1. Cells where the RC fit saturated its tau upper bound (100 s) are replaced by
     interpolation along SoC from the good neighbouring cells.
  2. R1 and C1 are noisy along SoC (a 9.8 s fit window cannot identify tau well),
     so an optional 3-point median filter is applied along the SoC axis only.

OCV and R0 are NOT touched. Run from anywhere:  python scripts/clean_battery_lut.py
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]          # drone_energy_rl/
IN_PATH = ROOT / "data" / "battery_luts" / "battery_6s_2d_lut.npz"
OUT_PATH = ROOT / "data" / "battery_luts" / "battery_6s_2d_lut_clean.npz"
PLOT_PATH = ROOT / "results" / "lut_cleanup.png"

TAU_CAP_S = 100.0     # upper bound used in curve_fit in BatteryTemperatureMatrix.py
TAU_TOL = 0.99        # tau >= 99 s counts as "stuck at the bound"


def find_bad_cells(r1, c1):
    return (r1 * c1) >= TAU_CAP_S * TAU_TOL


def repair_along_soc(table, bad, soc):
    out = table.copy()
    for j in range(table.shape[1]):
        good = ~bad[:, j]
        if good.sum() < 2:
            raise ValueError(f"Temperature column {j}: fewer than 2 good SoC points")
        out[bad[:, j], j] = np.interp(soc[bad[:, j]], soc[good], table[good, j])
    return out


def validate(d):
    for k in ("ocv", "r0", "r1", "c1"):
        assert np.all(np.isfinite(d[k])), f"{k} has NaN/inf"
        assert np.all(d[k] > 0), f"{k} has non-positive values"
    tau = d["r1"] * d["c1"]
    print(f"  tau range: {tau.min():.1f} to {tau.max():.1f} s")
    r0_falling = np.mean(np.diff(d["r0"], axis=1) < 0)
    print(f"  R0 decreases with temperature in {100 * r0_falling:.0f}% of steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-smooth", action="store_true", help="only repair saturated cells")
    ap.add_argument("--inp", type=Path, default=IN_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    raw = np.load(args.inp)
    d = {k: raw[k].copy() for k in raw.files}
    soc = d["soc"]

    bad = find_bad_cells(d["r1"], d["c1"])
    rows = sorted(set(np.where(bad)[0]))
    print(f"Saturated cells: {bad.sum()} (SoC rows: {[round(float(soc[i]), 2) for i in rows]})")

    clean = dict(d)
    clean["r1"] = repair_along_soc(d["r1"], bad, soc)
    clean["c1"] = repair_along_soc(d["c1"], bad, soc)
    if not args.no_smooth:
        clean["r1"] = median_filter(clean["r1"], size=(3, 1), mode="nearest")
        clean["c1"] = median_filter(clean["c1"], size=(3, 1), mode="nearest")

    print("Validation of cleaned table:")
    validate(clean)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **clean)
    print(f"Saved: {args.out}")

    # Before/after plot at 25 C
    j = int(np.argmin(np.abs(d["temp"] - 25.0)))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, (name, raw_y, new_y) in zip(ax, [
        ("R1 [ohm]", d["r1"][:, j], clean["r1"][:, j]),
        ("C1 [F]", d["c1"][:, j], clean["c1"][:, j]),
        ("tau = R1*C1 [s]", (d["r1"] * d["c1"])[:, j], (clean["r1"] * clean["c1"])[:, j]),
    ]):
        a.plot(soc, raw_y, "o-", color="tab:red", alpha=0.6, label="original")
        a.plot(soc, new_y, "s-", color="tab:blue", label="cleaned")
        a.set_title(f"{name} at {d['temp'][j]:.0f} C")
        a.set_xlabel("SoC")
        a.legend()
    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
