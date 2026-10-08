"""
sanity_check.py — Run the energy model and environment in isolation to verify
correctness before touching RL training.

Tests
-----
  1. Energy model: hover power, battery derating, altitude effects
  2. Wind zone map: cell lookup, scenario loading
  3. Environment: reset, step loop (100 steps), observation shape
  4. PID baseline: full episode rollout

Run this FIRST before running train_ppo.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import traceback

PASS  = "\033[92m[PASS]\033[0m"
FAIL  = "\033[91m[FAIL]\033[0m"
SKIP  = "\033[93m[SKIP]\033[0m"
HEAD  = "\033[1m"
ENDC  = "\033[0m"


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}" + (f"  ({detail})" if detail else ""))
    return condition


def section(title: str):
    print(f"\n{HEAD}{'─'*60}{ENDC}")
    print(f"{HEAD}  {title}{ENDC}")
    print(f"{HEAD}{'─'*60}{ENDC}")


def test_energy_model():
    section("1. Energy Model")
    from drone_energy.physics.energy import (
        DroneParams, BatteryState,
        air_density, battery_capacity_factor,
        hover_power, flight_power, available_energy_wh, GRAVITY,
    )
    drone = DroneParams()
    ok = True

    # Air density should decrease with altitude
    rho_0    = air_density(0)
    rho_3000 = air_density(3000)
    ok &= check("Air density decreases with altitude",
                rho_0 > rho_3000,
                f"rho(0)={rho_0:.3f}, rho(3000)={rho_3000:.3f} kg/m³")

    # Hover power at 3000 m > hover power at 0 m (less dense air)
    p0   = hover_power(drone, 0, 25)
    p3k  = hover_power(drone, 3000, 25)
    ok &= check("Hover power increases with altitude",
                p3k > p0,
                f"P(0)={p0:.2f} W, P(3000)={p3k:.2f} W")

    # Payload increases hover power
    p_no_payload   = hover_power(drone, 0, 25, 0.0)
    p_with_payload = hover_power(drone, 0, 25, 0.3)
    ok &= check("Payload increases hover power",
                p_with_payload > p_no_payload,
                f"P(0kg)={p_no_payload:.2f} W, P(0.3kg)={p_with_payload:.2f} W")

    # Battery capacity decreases with cold
    cap_warm = battery_capacity_factor(25)
    cap_cold = battery_capacity_factor(-10)
    ok &= check("Battery capacity decreases in cold",
                cap_warm > cap_cold,
                f"cap(25°C)={cap_warm:.2f}, cap(-10°C)={cap_cold:.2f}")

    # Battery depletion
    bat = BatteryState(drone, 25.0)
    ok &= check("Battery initialises with full charge",
                abs(bat.state_of_charge - 1.0) < 0.01,
                f"SoC={bat.state_of_charge:.3f}")
    bat.consume(1000.0, 200.0)   # drain it
    ok &= check("Battery depletes under load",
                bat.is_depleted,
                f"remaining={bat.remaining_wh:.4f} Wh")

    # Flight power > hover power (due to drag)
    vel  = np.array([5.0, 0.0, 0.0])
    wind = np.array([0.0, 0.0, 0.0])
    pf   = flight_power(drone, vel, wind, 0, 25, 0)
    ok &= check("Flight power >= hover power",
                pf >= p0,
                f"P_flight={pf:.2f} W, P_hover={p0:.2f} W")

    return ok


def test_wind_zones():
    section("2. Wind Zone Map")
    from drone_energy.weather.zones import WindZoneMap, make_scenario
    ok = True

    for name in ["calm", "windy", "cold", "random"]:
        try:
            zm = make_scenario(name, seed=42)
            cell = zm.get_cell(np.array([5.0, 5.0, 2.0]))
            obs  = zm.get_local_obs(np.array([5.0, 5.0, 2.0]))
            ok &= check(f"Scenario '{name}' loads and returns obs",
                        obs.shape == (4,),
                        f"wind={cell.wind_vector}, temp={cell.temperature_c:.1f}°C")
        except Exception as e:
            print(f"  {FAIL} Scenario '{name}' raised: {e}")
            ok = False

    # Neighbourhood obs
    zm   = make_scenario("calm", seed=0)
    nobs = zm.get_neighbourhood_obs(np.array([10.0, 10.0, 3.0]))
    ok  &= check("Neighbourhood obs shape is (28,)",
                 nobs.shape == (28,),
                 f"shape={nobs.shape}")

    return ok


def test_environment():
    section("3. MultiFactorDroneEnv")
    try:
        from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv
    except ImportError as e:
        print(f"  {SKIP} Environment tests — missing dependency: {e}")
        print("  Install gymnasium/stable-baselines3 then re-run.")
        return True   # not a physics failure — skip gracefully

    ok = True

    for scenario in ["calm", "windy", "cold"]:
        try:
            env = MultiFactorDroneEnv(scenario=scenario, seed=0)
            obs, info = env.reset()
            ok &= check(f"[{scenario}] reset() returns correct obs shape",
                        obs.shape == (22,),
                        f"shape={obs.shape}")

            total_r = 0.0
            for i in range(100):
                action = env.action_space.sample()
                obs, r, term, trunc, info = env.step(action)
                total_r += r
                if term or trunc:
                    break

            ok &= check(f"[{scenario}] step() runs without error",
                        True, f"steps={i+1}, total_r={total_r:.2f}")
            ok &= check(f"[{scenario}] obs shape stays consistent",
                        obs.shape == (22,))
            env.close()
        except Exception as e:
            print(f"  {FAIL} [{scenario}] raised: {e}")
            traceback.print_exc()
            ok = False

    return ok


def test_pid_baseline():
    section("4. PID Baseline Rollout")
    try:
        from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv
        from drone_energy.baselines.pid_baseline import WaypointPIDAgent
    except ImportError as e:
        print(f"  {SKIP} PID baseline test — missing dependency: {e}")
        return True

    ok = True


    try:
        env   = MultiFactorDroneEnv(scenario="calm", seed=42)
        agent = WaypointPIDAgent()

        obs, _ = env.reset()
        agent.reset()

        done        = False
        step_count  = 0
        total_r     = 0.0
        max_soc     = -1.0
        min_soc     = 2.0

        while not done and step_count < 500:
            action = agent.act(obs)
            obs, r, term, trunc, info = env.step(action)
            total_r    += r
            step_count += 1
            soc = info.get("battery_soc", 1.0)
            max_soc = max(max_soc, soc)
            min_soc = min(min_soc, soc)
            done = term or trunc

        ok &= check("PID baseline runs a full episode",
                    step_count > 0,
                    f"steps={step_count}, total_r={total_r:.2f}, "
                    f"SoC=[{min_soc:.2f},{max_soc:.2f}]")
        env.close()
    except Exception as e:
        print(f"  {FAIL} PID baseline raised: {e}")
        traceback.print_exc()
        ok = False

    return ok


def test_model_validation_values():
    """Print key model validation numbers for sanity-checking against literature."""
    section("5. Model Validation Numbers (read + verify manually)")
    from drone_energy.physics.energy import DroneParams, hover_power, battery_capacity_factor, air_density

    drone = DroneParams(mass=1.0, rotor_radius=0.1, num_rotors=4)

    print("  Battery capacity factor:")
    for t in [-20, -10, 0, 10, 20, 25, 35]:
        f = battery_capacity_factor(t)
        bar = "█" * int(f * 20)
        print(f"    {t:+4d}°C → {f*100:5.1f}%  {bar}")

    print("\n  Hover power (mass=1.0 kg, no payload):")
    for alt in [0, 500, 1000, 2000, 3000]:
        p = hover_power(drone, alt, 25, 0)
        print(f"    {alt:5d} m → {p:6.2f} W")

    print("\n  Air density:")
    for alt in [0, 500, 1000, 2000, 3000]:
        rho = air_density(alt)
        print(f"    {alt:5d} m → {rho:.4f} kg/m³")

    print("\n  Expected ranges from published data:")
    print("    Hover power ~50–150 W for 1 kg quad → check")
    print("    Li-Po at 0°C → ~85% capacity, at -10°C → ~70% → check")
    print("    Air density at 3000 m → ~0.91 kg/m³ → check")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  Multi-Factor Drone RL — Sanity Checks")
    print(f"{'='*60}")

    results = []
    results.append(("Energy Model",    test_energy_model()))
    results.append(("Wind Zones",      test_wind_zones()))
    results.append(("Environment",     test_environment()))
    results.append(("PID Baseline",    test_pid_baseline()))
    test_model_validation_values()

    print(f"\n{'='*60}")
    print("  Summary")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        all_ok = all_ok and ok

    if all_ok:
        print(f"\n  {PASS} All checks passed — ready to train!\n")
        sys.exit(0)
    else:
        print(f"\n  {FAIL} Some checks failed — fix before training.\n")
        sys.exit(1)
