"""
Fit the thermal parameters (C_th, h_a) of the battery model from a bench log.

CSV columns (header required): time_s,current_a,voltage_v,pack_temp_c,ambient_c
Run from anywhere:
    python scripts/fit_thermal_params.py --csv my_log.csv [--soc0 1.0] [--capacity-ah 5.0]
    python scripts/fit_thermal_params.py --selftest
Read drone_energy/battery/thermal_fit.py for the bench protocol.
"""
import argparse
import numpy as np
from drone_energy.battery.thermal_fit import (fit_thermal_params, heat_from_log, synthetic_log)
from drone_energy.battery.thevenin import ThermalParams


def report(est, info, true=None):
    print(f"  C_th = {est.c_th:7.1f} +- {info['se_c_th']:.1f} J/K" + (f"   (true {true.c_th})" if true else ""))
    print(f"  h_a  = {est.h_a:7.3f} +- {info['se_h_a']:.3f} W/K" + (f"   (true {true.h_a})" if true else ""))
    print(f"  fit RMSE = {info['rmse_k']:.3f} K,  time constant C_th/h_a = {est.c_th / est.h_a:.0f} s")
    print("  NOTE: the +- values assume independent noise. On a real bench log the error is dominated by")
    print("        systematic effects (heat estimate, sensor placement, temperature gradients): expect +-10-30%.")
    if info["se_c_th"] > 0.3 * est.c_th or info["se_h_a"] > 0.3 * est.h_a:
        print("  WARNING: large uncertainty - the log probably lacks a cool-down phase or enough temperature rise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--soc0", type=float, default=1.0)
    ap.add_argument("--capacity-ah", type=float, default=5.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        true = ThermalParams(c_th=420.0, h_a=0.55)
        t, i, v, temp, amb = synthetic_log(true)
        q = heat_from_log(t, i, v, temp)
        est, info = fit_thermal_params(t, q, temp, amb)
        print("Self-test on a synthetic log (noise 0.05 K, 10 mV):")
        report(est, info, true)
        return
    if not a.csv:
        ap.error("give --csv FILE or --selftest")
    d = np.genfromtxt(a.csv, delimiter=",", names=True)
    q = heat_from_log(d["time_s"], d["current_a"], d["voltage_v"], d["pack_temp_c"], a.soc0, a.capacity_ah)
    est, info = fit_thermal_params(d["time_s"], q, d["pack_temp_c"], d["ambient_c"])
    print("Fitted thermal parameters:")
    report(est, info)


if __name__ == "__main__":
    main()
