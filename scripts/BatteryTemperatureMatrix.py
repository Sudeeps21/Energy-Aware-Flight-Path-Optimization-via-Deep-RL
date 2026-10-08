# pyrefly: ignore [missing-import]
import os
import numpy as np
import scipy.optimize as opt
import pybamm

# Suppress verbose PyBaMM solver logging during batch runs
pybamm.set_logging_level("ERROR")

def fit_1rc_parameters(time, voltage, current):
    """
    Extracts OCV, R0, R1, and C1 from a single HPPC pulse profile using time-window slicing.
    """
    # 1. Locate the exact index where the 10A pulse turns ON (> 0.1 A)
    pulse_indices = np.where(current > 0.1)[0]
    if len(pulse_indices) == 0:
        return voltage[0], 0.015, 0.005, 1000.0

    idx_start = pulse_indices[0]
    
    # Baseline OCV right before pulse start (timestep before current load)
    ocv_cell = voltage[max(0, idx_start - 1)]
    
    # Instantaneous voltage drop at pulse start -> R0
    v_pulse_start = voltage[idx_start]
    i_pulse = current[idx_start]
    delta_v0 = abs(ocv_cell - v_pulse_start)
    r0 = delta_v0 / i_pulse

    # 2. Slice transient data using physical time window (0 to 9.8 seconds of pulse)
    t_start = time[idx_start]
    pulse_mask = (time >= t_start) & (time <= t_start + 9.8)
    
    t_pulse = time[pulse_mask] - t_start
    v_pulse = voltage[pulse_mask]
    v_transient = abs(v_pulse_start - v_pulse)

    # 3. Fit exponential curve: V_transient(t) = I * R1 * (1 - exp(-t / (R1*C1)))
    def transient_model(t, r1, tau):
        return i_pulse * r1 * (1.0 - np.exp(-t / np.maximum(tau, 1e-4)))

    try:
        popt, _ = opt.curve_fit(
            transient_model,
            t_pulse,
            v_transient,
            p0=[0.008, 5.0],
            bounds=([1e-4, 0.01], [0.10, 100.0])
        )
        r1 = popt[0]
        tau1 = popt[1]
        c1 = tau1 / np.maximum(r1, 1e-5)
    except Exception:
        # Robust fallback values if curve fitting fails to converge
        r1 = 0.006
        c1 = 1200.0

    return ocv_cell, r0, r1, c1


def run_high_res_extraction():
    print("=================================================================")
    print(" Starting High-Resolution 2D Matrix Derivation for 6S LiPo Pack  ")
    print("=================================================================")

    # High-Density Grids: 20 SoC points x 17 Temperature points
    soc_grid = np.linspace(0.05, 1.0, 20)                        # 5% SoC resolution
    temp_grid_c = np.linspace(-20.0, 60.0, 17)                   # 5°C Temp resolution

    n_soc = len(soc_grid)
    n_temp = len(temp_grid_c)

    print(f"Grid Dimensions: {n_soc} SoC steps x {n_temp} Temp steps = {n_soc * n_temp} simulations")

    # Matrices shape: (20, 17)
    ocv_cell_mat = np.zeros((n_soc, n_temp))
    r0_cell_mat   = np.zeros((n_soc, n_temp))
    r1_cell_mat   = np.zeros((n_soc, n_temp))
    c1_cell_mat   = np.zeros((n_soc, n_temp))

    params = pybamm.ParameterValues("Chen2020")
    model = pybamm.lithium_ion.SPMe()

    experiment = pybamm.Experiment([
        "Rest for 5 seconds",
        "Discharge at 10 A for 10 seconds",
        "Rest for 30 seconds"
    ])

    # Execution Loop across Temperature and SoC
    for j, temp_c in enumerate(temp_grid_c):
        temp_k = temp_c + 273.15
        params.update({
            "Ambient temperature [K]": temp_k,
            "Initial temperature [K]": temp_k,
        })
        print(f"\n---> [Temp {j+1:02d}/{n_temp}] {temp_c:5.1f}°C | Progress: ", end="", flush=True)

        for i, target_soc in enumerate(soc_grid):
            try:
                sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
                sol = sim.solve(initial_soc=target_soc)

                t_arr = sol["Time [s]"].entries
                v_arr = sol["Terminal voltage [V]"].entries
                i_arr = sol["Current [A]"].entries

                ocv_c, r0_c, r1_c, c1_c = fit_1rc_parameters(t_arr, v_arr, i_arr)

                ocv_cell_mat[i, j] = ocv_c
                r0_cell_mat[i, j]  = r0_c
                r1_cell_mat[i, j]  = r1_c
                c1_cell_mat[i, j]  = c1_c
                print(".", end="", flush=True)

            except Exception:
                # Fallback to neighbor value if extreme boundary fails
                ocv_cell_mat[i, j] = ocv_cell_mat[max(0, i-1), j]
                r0_cell_mat[i, j]  = r0_cell_mat[max(0, i-1), j]
                r1_cell_mat[i, j]  = r1_cell_mat[max(0, i-1), j]
                c1_cell_mat[i, j]  = c1_cell_mat[max(0, i-1), j]
                print("x", end="", flush=True)

    # 6S Series Pack Scaling & Wiring Offset
    R_WIRING_PACK = 0.002  # 2.0 mΩ added for nickel tabs, AWG wiring, and main connector

    ocv_6s_mat = ocv_cell_mat * 6.0
    r0_6s_mat  = (r0_cell_mat * 6.0) + R_WIRING_PACK
    r1_6s_mat  = r1_cell_mat * 6.0
    c1_6s_mat  = c1_cell_mat / 6.0

    # Save High-Density Matrix File
    os.makedirs("data", exist_ok=True)
    export_path = os.path.join("data", "battery_6s_2d_lut.npz")

    np.savez(
        export_path,
        soc=soc_grid,
        temp=temp_grid_c,
        ocv=ocv_6s_mat,
        r0=r0_6s_mat,
        r1=r1_6s_mat,
        c1=c1_6s_mat
    )

    print("\n\n=================================================================")
    print(f" SUCCESS: Exported high-resolution 6S lookup table to '{export_path}'!")
    print(f" Output shape: {ocv_6s_mat.shape} [20 SoC x 17 Temp]")
    print("=================================================================")

if __name__ == "__main__":
    run_high_res_extraction()