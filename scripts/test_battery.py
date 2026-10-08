"""
Test the battery model alone (no simulator): constant power at several temperatures.
Run from anywhere:  python scripts/test_battery.py
"""
import numpy as np
from drone_energy.battery.thevenin import TheveninBattery

DT = 0.1            # s
POWER_W = 300.0     # constant electrical load
T_MAX_S = 1800.0    # stop after 30 min if not depleted


def run(temp_c, power_w=POWER_W):
    bat = TheveninBattery.from_npz(thermal=None)
    t, last, first = 0.0, None, None
    while t < T_MAX_S:
        out = bat.step(power_w, DT, temp_c)
        first = first or out
        last = out
        t += DT
        if out["depleted"]:
            break
    return bat, first, last, t


print(f"Constant {POWER_W:.0f} W load, dt={DT}s\n")
print(f"{'T [C]':>6} | {'V start':>8} | {'I start':>8} | {'V end':>7} | {'SoC end':>8} | "
      f"{'time [s]':>8} | {'Wh out':>7} | {'depleted':>8}")
results = {}
for T in [60, 25, 0, -20]:
    bat, first, last, t = run(T)
    results[T] = (first, last, t, bat.energy_out_wh)
    print(f"{T:>6} | {first['v_terminal']:>8.2f} | {first['current']:>8.2f} | "
          f"{last['v_terminal']:>7.2f} | {last['soc']:>8.3f} | {t:>8.0f} | "
          f"{bat.energy_out_wh:>7.1f} | {str(last['depleted']):>8}")

# --- Checks -------------------------------------------------------------------
warm, cold = results[25], results[-20]
assert cold[0]["v_terminal"] < warm[0]["v_terminal"], "cold should sag more"
assert cold[0]["current"] > warm[0]["current"], "cold needs more current for same power"
assert cold[2] <= warm[2], "cold pack should not last longer"

bat = TheveninBattery.from_npz(thermal=None)
step = bat.step(300.0, DT, 25.0)
assert abs(step["v_terminal"] * step["current"] - 300.0) < 1e-6, "power not conserved"

print("\nMax deliverable power [W] at full charge:")
for T in [60, 25, 0, -20]:
    bat.reset()
    print(f"  {T:>4} C : {bat.max_power(T):7.0f}")

bat.reset()
out = bat.step(5000.0, DT, 25.0)
assert out["brownout"], "5 kW request should brown out"
print(f"\n5000 W request -> brownout={out['brownout']}, delivered {out['power_delivered']:.0f} W")

print("\nAll battery checks passed.")
