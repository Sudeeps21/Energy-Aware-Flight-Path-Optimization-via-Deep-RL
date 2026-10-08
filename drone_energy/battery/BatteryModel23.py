import numpy as np
import scipy.interpolate as interp

class BatteryModel23:
    """
    1-RC (Thévenin) Battery Model - Model 2.3 Architecture
    ------------------------------------------------------
    - Dynamic (SOC-dependent): OCV(SoC), R0(SoC)
    - Static (Constant scalars): R1, C1
    """
    def __init__(self, capacity_ah, soc_grid, ocv_grid, r0_grid, r1_const, c1_const):
        self.q_nominal = capacity_ah * 3600.0  # Convert Ah to Ampere-seconds (Coulombs)
        
        # Static Parameters (Model 2.3)
        self.r1 = float(r1_const)
        self.c1 = float(c1_const)
        self.tau1 = self.r1 * self.c1  # Polarization time constant (seconds)
        
        # 1D Dynamic Lookup Interpolators
        self.get_ocv = interp.interp1d(soc_grid, ocv_grid, kind='linear', fill_value='extrapolate')
        self.get_r0  = interp.interp1d(soc_grid, r0_grid,  kind='linear', fill_value='extrapolate')

    def simulate(self, current_profile, dt, initial_soc=1.0, initial_v1=0.0):
        """
        Simulates terminal voltage under a given current profile.
        
        Parameters:
            current_profile (array-like): Discharge current in Amps (positive = discharging)
            dt (float): Time step in seconds
            initial_soc (float): Initial State of Charge (0.0 to 1.0)
            initial_v1 (float): Initial RC circuit polarization voltage in Volts
            
        Returns:
            dict containing arrays for time, soc, terminal_voltage, v1, ocv, r0
        """
        n_steps = len(current_profile)
        
        # Preallocate output arrays
        time = np.arange(n_steps) * dt
        soc = np.zeros(n_steps)
        v_terminal = np.zeros(n_steps)
        v1 = np.zeros(n_steps)
        ocv_hist = np.zeros(n_steps)
        r0_hist = np.zeros(n_steps)
        
        # Initialize state variables
        current_soc = float(initial_soc)
        current_v1 = float(initial_v1)
        
        # Discrete-time exponential decay factor for the RC branch (exact solution for constant current step)
        alpha = np.exp(-dt / self.tau1)
        
        for k in range(n_steps):
            i_load = current_profile[k]
            
            # Lookup dynamic SOC-dependent parameters
            ocv = float(self.get_ocv(current_soc))
            r0 = float(self.get_r0(current_soc))
            
            # Save history
            soc[k] = current_soc
            ocv_hist[k] = ocv
            r0_hist[k] = r0
            v1[k] = current_v1
            
            # Terminal Voltage Equation: V_t = OCV - V1 - (I * R0)
            v_terminal[k] = ocv - current_v1 - (i_load * r0)
            
            # Update States for Next Time Step
            # 1. Update V1: Exact discrete zero-order hold (ZOH) update
            current_v1 = (current_v1 * alpha) + (i_load * self.r1 * (1.0 - alpha))
            
            # 2. Update SoC via Coulomb Counting
            current_soc -= (i_load * dt) / self.q_nominal
            current_soc = np.clip(current_soc, 0.0, 1.0)
            
        return {
            "time": time,
            "soc": soc,
            "v_terminal": v_terminal,
            "v1": v1,
            "ocv": ocv_hist,
            "r0": r0_hist
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Sample lookup tables for a 6S LiPo pack cell (~3.0V to 4.2V nominal range per cell)
    soc_lookup = np.linspace(0.0, 1.0, 11)
    ocv_lookup = np.array([3.00, 3.45, 3.62, 3.70, 3.75, 3.82, 3.88, 3.95, 4.02, 4.10, 4.20])
    r0_lookup  = np.array([0.015, 0.010, 0.008, 0.006, 0.005, 0.005, 0.005, 0.005, 0.006, 0.007, 0.008]) # Ohms

    # Model 2.3 Constants
    r1_mean = 0.012  # Ohms (constant)
    c1_mean = 1500.0 # Farads (constant)

    # Instantiate
    model = BatteryModel23(
        capacity_ah=5.0, # 5000 mAh
        soc_grid=soc_lookup,
        ocv_grid=ocv_lookup,
        r0_grid=r0_lookup,
        r1_const=r1_mean,
        c1_const=c1_mean
    )

    # 100-second pulsed current load profile (10A pulses)
    t_dt = 0.1
    current = np.zeros(1000)
    current[100:300] = 10.0  # Pulse 1
    current[500:800] = 15.0  # Pulse 2

    res = model.simulate(current_profile=current, dt=t_dt)
    print(f"Simulation completed. Final SoC: {res['soc'][-1]:.4f}, Final Voltage: {res['v_terminal'][-1]:.3f} V")
