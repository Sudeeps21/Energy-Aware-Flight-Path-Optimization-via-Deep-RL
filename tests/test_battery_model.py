"""
Verification suite for drone_energy.battery.thevenin.TheveninBattery.

Run (no pytest needed):   python tests/test_battery_model.py
Or with pytest:           pytest tests/test_battery_model.py -s

Group A - code/maths correctness  (must match analytic solutions exactly)
Group B - physical behaviour      (trends and magnitudes a real Li-ion pack must show)
A "WARN" is a soft check: it never fails the suite but flags a known model limitation.
"""
import sys
import numpy as np
from drone_energy.battery.thevenin import TheveninBattery, ThermalParams, DEFAULT_THERMAL
from drone_energy.battery.thermal_fit import (simulate_pack_temp, fit_thermal_params,
                                              heat_from_log, synthetic_log)

WARNINGS = []
NOMINAL_WH = 5.0 * 3.6 * 6      # 5 Ah x 3.6 V/cell x 6 cells = 108 Wh (check your cell datasheet)


def warn(msg):
    WARNINGS.append(msg)
    print(f"      WARN: {msg}")


def make_const_battery(ocv=22.0, r0=0.10, r1=0.05, c1=300.0, cap=5.0):
    """Battery with flat tables, so closed-form solutions apply exactly."""
    f = lambda v: np.full((2, 2), float(v))
    return TheveninBattery(cap, [0.05, 1.0], [-20.0, 60.0], f(ocv), f(r0), f(r1), f(c1))


def run_to_cutoff(power_w, temp_c, dt=0.1, max_t=40000.0):
    """Constant-power discharge until cutoff. Returns (Wh, soc_end, seconds, eta)."""
    b = TheveninBattery.from_npz(thermal=None)
    num = den = t = 0.0
    while t < max_t:
        o = b.step(power_w, dt, temp_c)
        ocv = b._params(b.soc, temp_c)[0]
        num += o["v_terminal"] * o["current"] * dt
        den += ocv * o["current"] * dt
        t += dt
        if o["depleted"]:
            break
    return b.energy_out_wh, b.soc, t, num / den


# =============================================================== GROUP A: maths
def test_A1_rc_step_response_matches_analytic():
    """Constant 10 A on flat tables: V1(t) must equal I*R1*(1-exp(-t/tau)) to 1e-9 V."""
    b = make_const_battery()
    tau, dt, I = 0.05 * 300.0, 0.1, 10.0
    worst = 0.0
    for k in range(600):
        o = b.step_current(I, dt, 25.0)
        analytic = I * 0.05 * (1 - np.exp(-(k + 1) * dt / tau))
        worst = max(worst, abs(o["v1"] - analytic))
    print(f"      max |V1 - analytic| = {worst:.2e} V")
    assert worst < 1e-9


def test_A2_rest_relaxation():
    """After load removal V1 decays with tau, SoC is frozen, V_t -> OCV."""
    b = make_const_battery()
    for _ in range(600):
        b.step_current(10.0, 0.1, 25.0)
    v1_0, soc_0 = b.v1, b.soc
    for _ in range(1200):                       # 120 s rest = 8 tau
        o = b.step_current(0.0, 0.1, 25.0)
    expected = v1_0 * np.exp(-120.0 / 15.0)
    print(f"      V1 after rest {b.v1:.3e} V (analytic {expected:.3e}), SoC drift {abs(b.soc - soc_0):.1e}")
    assert abs(b.v1 - expected) < 1e-9
    assert abs(b.soc - soc_0) < 1e-12
    assert abs(o["v_terminal"] - 22.0) < 1e-3


def test_A3_coulomb_counting():
    """5 A for 1800 s from a 5 Ah pack removes exactly 0.5 of the SoC."""
    b = make_const_battery()
    for _ in range(18000):
        b.step_current(5.0, 0.1, 25.0)
    print(f"      final SoC {b.soc:.12f}")
    assert abs(b.soc - 0.5) < 1e-9


def test_A4_lookup_orientation_and_clamping():
    """Catches transposed (SoC,T) axes, wrong interpolation, and extrapolation."""
    b = TheveninBattery.from_npz(thermal=None)
    soc, tmp = b.soc_grid, b.temp_grid
    tables = [b._lut.values[..., k] for k in range(4)]
    for (i, j) in [(0, 0), (7, 3), (19, 16), (10, 9)]:
        got = b._params(soc[i], tmp[j])
        want = tuple(t[i, j] for t in tables)
        assert np.allclose(got, want, rtol=1e-12), f"node ({i},{j}) mismatch"
    # bilinear midpoint of a cell
    i, j = 5, 4
    mid = b._params((soc[i] + soc[i + 1]) / 2, (tmp[j] + tmp[j + 1]) / 2)
    want = tuple(t[i:i + 2, j:j + 2].mean() for t in tables)
    assert np.allclose(mid, want, rtol=1e-12), "midpoint is not the bilinear average"
    # clamping outside the table
    assert np.allclose(b._params(0.0, -50.0), b._params(soc[0], tmp[0]))
    assert np.allclose(b._params(2.0, 100.0), b._params(soc[-1], tmp[-1]))


def test_A5_constant_power_closure_and_brownout_boundary():
    """V*I == P for feasible requests; above max_power the pack saturates at max_power."""
    rng = np.random.default_rng(0)
    b = TheveninBattery.from_npz(thermal=None)
    worst = 0.0
    for _ in range(300):
        s, T, v1 = rng.uniform(0.05, 1), rng.uniform(-20, 60), rng.uniform(0, 1.0)
        b.reset(soc=s); b.v1 = v1
        pmax = b.max_power(T)
        P = rng.uniform(1, 0.95 * pmax)
        o = b.step(P, 0.1, T)
        worst = max(worst, abs(o["v_terminal"] * o["current"] - P))
        assert not o["brownout"]
        b.reset(soc=s); b.v1 = v1
        o = b.step(1.05 * pmax, 0.1, T)
        assert o["brownout"]
        assert abs(o["power_delivered"] - pmax) / pmax < 1e-9
    print(f"      worst |V*I - P| = {worst:.2e} W over 300 random states")
    assert worst < 1e-8


def test_A6_timestep_convergence():
    """Result must not depend on dt (env runs at ~30 Hz, tests at 10 Hz)."""
    def final(dt):
        b = TheveninBattery.from_npz(thermal=None)
        for _ in range(int(round(120 / dt))):
            o = b.step(300.0, dt, 25.0)
        return o["soc"], o["v_terminal"]
    s_ref, v_ref = final(0.005)
    for dt in [1.0, 0.5, 1 / 30, 0.1, 0.01]:
        s, v = final(dt)
        print(f"      dt={dt:6.3f}: dSoC={s - s_ref:+.1e}  dV={1000 * (v - v_ref):+.2f} mV")
        if dt <= 0.5:
            assert abs(s - s_ref) < 1e-4 and abs(v - v_ref) < 5e-3


def test_A7_reset_and_determinism():
    runs = []
    b = TheveninBattery.from_npz(thermal=None)
    for _ in range(2):
        b.reset()
        for _ in range(500):
            o = b.step(200.0, 0.1, 10.0)
        runs.append((o["soc"], o["v_terminal"], b.energy_out_wh))
    assert runs[0] == runs[1], "reset() does not fully restore state"


# ========================================================== GROUP B: physics
def test_B1_voltage_sag_grows_as_temperature_drops():
    temps = np.arange(-20, 61, 5)
    v = []
    for T in temps:
        b = TheveninBattery.from_npz(thermal=None)
        v.append(b.step(300.0, 0.1, T)["v_terminal"])
    print(f"      V at 300 W: {v[0]:.2f} V (-20C) ... {v[-1]:.2f} V (60C)")
    assert np.all(np.diff(v) > 0), "terminal voltage should rise monotonically with temperature"


def test_B2_usable_energy_decreases_when_colder():
    E = [run_to_cutoff(220.0, T)[0] for T in [60, 40, 25, 10, 0, -10, -20]]
    print("      Wh at 220 W (60..-20 C): " + ", ".join(f"{e:.1f}" for e in E))
    assert all(a >= b - 1e-6 for a, b in zip(E, E[1:]))
    assert E[0] - E[-1] > 5.0, "cold should cost a visible amount of energy at 220 W"


def test_B3_rate_capacity_effect():
    """Higher discharge power -> less usable energy (every real cell does this)."""
    powers = [25, 55, 110, 220, 300]
    E = [run_to_cutoff(P, 25.0)[0] for P in powers]
    print("      Wh at 25 C: " + ", ".join(f"{P}W={e:.1f}" for P, e in zip(powers, E)))
    assert all(a > b for a, b in zip(E, E[1:]))


def test_B4_discharge_efficiency_in_realistic_band():
    """eta = (energy at terminals)/(energy released from OCV source)."""
    out = {}
    for P in [25, 110, 220]:
        out[P] = run_to_cutoff(P, 25.0)[3]
    print("      eta: " + ", ".join(f"{P}W={e:.3f}" for P, e in out.items()))
    assert out[25] > 0.97          # ~C/5
    assert 0.90 < out[110] < 0.99  # ~1C
    assert 0.85 < out[220] < 0.97  # ~2C
    assert out[25] > out[110] > out[220]


def test_B5_low_rate_energy_close_to_nominal():
    E = run_to_cutoff(25.0, 25.0)[0]
    print(f"      usable energy at ~C/5, 25 C: {E:.1f} Wh (nominal {NOMINAL_WH:.0f} Wh)")
    assert 0.88 * NOMINAL_WH < E < 1.08 * NOMINAL_WH


def test_B6_power_ceiling_ordering():
    b = TheveninBattery.from_npz(thermal=None)
    for T in [-20, 0, 25, 60]:
        b.reset(soc=1.0); p_full = b.max_power(T)
        b.reset(soc=0.2);  p_low = b.max_power(T)
        b.reset(soc=0.05); p_min = b.max_power(T)
        assert p_full > p_low > p_min, f"max power ordering wrong at {T} C"
    cold, warm = [TheveninBattery.from_npz(thermal=None) for _ in range(2)]
    assert cold.max_power(-20) < 0.75 * warm.max_power(25), "cold pack should lose >25% peak power"


def test_B7_voltage_rebound_when_load_removed():
    b = TheveninBattery.from_npz(thermal=None)
    for _ in range(300):
        loaded = b.step(300.0, 0.1, 25.0)
    rest0 = b.step_current(0.0, 0.1, 25.0)
    jump = rest0["v_terminal"] - loaded["v_terminal"]
    for _ in range(600):
        rest = b.step_current(0.0, 0.1, 25.0)
    print(f"      instant rebound {jump:.2f} V, further creep {rest['v_terminal'] - rest0['v_terminal']:.2f} V")
    assert jump > 1.0                                             # ~I*R0
    assert rest["v_terminal"] > rest0["v_terminal"]               # slow recovery
    assert abs(rest["soc"] - loaded["soc"]) < 1e-4                # no SoC change at rest


def test_B8_robustness_fuzz_and_drone_like_profile():
    rng = np.random.default_rng(1)
    for T in [-40.0, -20.0, 25.0, 80.0]:                          # incl. outside table
        b = TheveninBattery.from_npz(thermal=None)
        last_soc = 1.0
        for k in range(6000):                                     # 10 min at 10 Hz
            phase = (k // 100) % 4
            P = [100.0, 800.0, 30.0, 150.0][phase] * rng.uniform(0.8, 1.2)
            o = b.step(P, 0.1, T)
            assert all(np.isfinite([o["v_terminal"], o["current"], o["soc"], o["v1"]]))
            assert o["soc"] <= last_soc + 1e-15, "SoC rose during discharge"
            assert o["v_terminal"] > 0
            last_soc = o["soc"]
            if o["depleted"]:
                break
    # extreme random requests never produce NaN
    b = TheveninBattery.from_npz(thermal=None)
    for _ in range(2000):
        o = b.step(rng.uniform(0, 5000), 0.05, rng.uniform(-40, 80))
        assert all(np.isfinite(list(v for v in o.values() if not isinstance(v, bool))))
        if o["depleted"]:
            b.reset()


def test_B9_WARN_cold_capacity_at_low_rate():
    """Soft check. Real cells usually show a clear capacity loss at -20 C even at
    low rates (cell-dependent; compare with your cell's datasheet curves). This
    table has T-independent OCV and a T-independent RC branch, so low-rate cold
    capacity loss may be underestimated."""
    ratio = run_to_cutoff(55.0, -20.0)[0] / run_to_cutoff(55.0, 25.0)[0]
    print(f"      E(-20C)/E(25C) at ~C/2: {ratio:.3f}")
    if ratio > 0.95:
        warn(f"cold/warm energy ratio at C/2 is {ratio:.2f}; real cells typically lose more at -20 C. "
             "Model captures cold loss mainly at HIGH power (via R0 sag).")


# ==================================================== GROUP C: thermal (opt-in)
def test_C1_thermal_adiabatic_energy_balance():
    """With no heat loss, C_th * dT must equal the heat generated inside the pack."""
    b = TheveninBattery.from_npz(thermal=ThermalParams(c_th=390.0, h_a=0.0))
    for _ in range(3000):
        b.step(300.0, 0.1, -10.0)
    gained = 390.0 * (b.cell_temp + 10.0)
    print(f"      C*dT = {gained:.3f} J, heat generated = {b.heat_j:.3f} J")
    assert abs(gained - b.heat_j) < 1e-6 * b.heat_j


def test_C2_thermal_steady_state():
    """Flat tables, constant current: T_ss = T_amb + I^2 (R0 + R1) / h_a."""
    f = lambda v: np.full((2, 2), float(v))
    b = TheveninBattery(5.0, [0.05, 1.0], [-20.0, 60.0], f(22.0), f(0.10), f(0.05), f(300.0),
                        thermal=ThermalParams(c_th=100.0, h_a=1.0))
    for _ in range(20000):
        b.step_current(10.0, 0.1, 0.0)
    print(f"      T = {b.cell_temp:.4f} C, expected {10.0**2 * 0.15:.4f} C")
    assert abs(b.cell_temp - 15.0) < 1e-3


def test_C3_thermal_limit_matches_isothermal_and_is_stable():
    """Huge h_a pins the pack to ambient: results must equal the isothermal model, no NaN."""
    iso = TheveninBattery.from_npz(thermal=None)
    th = TheveninBattery.from_npz(thermal=ThermalParams(c_th=390.0, h_a=1e6))
    for _ in range(3000):
        a = iso.step(300.0, 0.1, -10.0)
        c = th.step(300.0, 0.1, -10.0)
    assert np.isfinite(c["v_terminal"]) and np.isfinite(th.cell_temp)
    assert abs(a["v_terminal"] - c["v_terminal"]) < 1e-3 and abs(a["soc"] - c["soc"]) < 1e-6


def test_C4_self_heating_physics_and_max_power_consistency():
    """Cold pack under load warms up, sags less than the isothermal pack, and max_power()
    uses the pack temperature (not the ambient)."""
    iso = TheveninBattery.from_npz(thermal=None)
    th = TheveninBattery.from_npz(thermal=ThermalParams())
    for _ in range(6000):                                  # 10 min at 200 W, -10 C ambient
        a = iso.step(200.0, 0.1, -10.0)
        c = th.step(200.0, 0.1, -10.0)
    print(f"      pack temp {th.cell_temp:.1f} C; V isothermal {a['v_terminal']:.2f} V vs self-heating {c['v_terminal']:.2f} V")
    assert th.cell_temp > -10.0 + 3.0
    assert c["v_terminal"] > a["v_terminal"]
    assert th.max_power(-10.0) == th.max_power(th.cell_temp)
    assert th.max_power(-10.0) > iso.max_power(-10.0)
    th.reset(cell_temp=-5.0)
    assert th.cell_temp == -5.0


def test_C5_ambient_change_moves_pack_temperature_smoothly():
    """Flying from a cold zone into a warm zone: the pack temperature must follow with the
    thermal time constant (no jump), and end up warmer."""
    b = TheveninBattery.from_npz()
    temps = []
    for k in range(9000):                                    # 15 min at 10 Hz
        amb = -10.0 if k < 3000 else 30.0                    # zone change after 5 min
        temps.append(b.step(150.0, 0.1, amb)["cell_temp"])
    temps = np.array(temps)
    jump = np.max(np.abs(np.diff(temps)))
    print(f"      pack temp: {temps[2999]:.1f} C at the zone change -> {temps[-1]:.1f} C, max step {jump:.4f} K")
    assert jump < 0.01
    assert np.all(np.diff(temps[3000:]) > 0) and temps[-1] > temps[2999] + 5.0


def test_C6_prewarmed_pack_cools_with_exact_time_constant():
    """At zero load there is no heat, so the pack must cool exactly as amb + dT*exp(-t*hA/C)."""
    th = ThermalParams(c_th=390.0, h_a=0.75)
    b = TheveninBattery.from_npz(thermal=th)
    b.reset(cell_temp=20.0)
    for _ in range(6000):                                    # 600 s
        o = b.step(0.0, 0.1, -10.0)
    expected = -10.0 + 30.0 * np.exp(-600.0 * 0.75 / 390.0)
    print(f"      pack temp after 600 s: {o['cell_temp']:.6f} C, analytic {expected:.6f} C")
    assert abs(o["cell_temp"] - expected) < 1e-6
    warm_first = TheveninBattery.from_npz()
    warm_first.reset(cell_temp=20.0)
    cold_first = TheveninBattery.from_npz()
    v_w = warm_first.step(300.0, 0.1, -10.0)["v_terminal"]
    v_c = cold_first.step(300.0, 0.1, -10.0)["v_terminal"]
    assert v_w > v_c, "a pre-warmed pack should sag less than a cold-soaked one"


def test_C7_overtemp_flag():
    b = TheveninBattery.from_npz(thermal=ThermalParams(c_th=390.0, h_a=0.0))
    first = b.step(300.0, 0.1, 25.0)
    assert first["overtemp"] is False
    flagged = False
    for _ in range(40000):
        o = b.step(300.0, 0.1, 25.0)
        if o["overtemp"]:
            flagged = True
            assert o["cell_temp"] > 60.0
            break
        if o["depleted"]:
            b.reset(cell_temp=b.cell_temp)
    assert flagged


def test_C8_defaults_and_parameter_helper():
    assert abs(ThermalParams.from_cells().c_th - DEFAULT_THERMAL.c_th) < 0.02 * DEFAULT_THERMAL.c_th
    assert TheveninBattery.from_npz().thermal is DEFAULT_THERMAL      # self-heating is ON by default
    assert TheveninBattery.from_npz(thermal=None).thermal is None     # isothermal on request


def test_C9_thermal_integration_identity_and_parameter_recovery():
    """(a) the fit module's integrator reproduces the battery's own pack temperature exactly;
    (b) the identification pipeline recovers known parameters from a noisy synthetic log."""
    b = TheveninBattery.from_npz()
    t, heat, temp = [0.0], [], [None]
    b.reset(cell_temp=0.0)
    for k in range(3000):
        o = b.step(200.0, 0.1, 0.0)
        heat.append(o["heat_w"]); temp.append(o["cell_temp"]); t.append((k + 1) * 0.1)
    sim = simulate_pack_temp(np.array(t), np.array(heat + [heat[-1]]), 0.0, 390.0, 0.75, 0.0)
    assert np.max(np.abs(sim[1:] - np.array(temp[1:]))) < 1e-9
    true = ThermalParams(c_th=420.0, h_a=0.55)
    tt, i, v, tp, amb = synthetic_log(true)
    est, info = fit_thermal_params(tt, heat_from_log(tt, i, v, tp), tp, amb)
    print(f"      recovered C_th {est.c_th:.1f} (true 420), h_a {est.h_a:.3f} (true 0.55), rmse {info['rmse_k']:.3f} K")
    assert abs(est.c_th - 420.0) < 0.1 * 420.0 and abs(est.h_a - 0.55) < 0.1 * 0.55


# ================================================================== runner
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        print(f"[RUN ] {name}")
        try:
            fn()
            print(f"[PASS] {name}\n")
        except AssertionError as e:
            failed.append(name)
            print(f"[FAIL] {name}  {e}\n")
    print("=" * 60)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed, {len(WARNINGS)} warning(s)")
    for w in WARNINGS:
        print(f"  WARN: {w}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
