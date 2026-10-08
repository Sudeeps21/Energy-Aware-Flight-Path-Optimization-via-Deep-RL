import numpy as np
import scipy.interpolate as interp

class BatteryModel24:
    """
    1-RC (Thévenin) Battery Model - Model 2.4 Architecture
    ------------------------------------------------------
    - Dynamic (SOC-dependent): OCV(SoC), R0(SoC), R1(SoC), C1(SoC)
    - Full 4-parameter lookup executed at every time step
    """
    def __init__(self, capacity_ah, soc_grid, ocv_grid, r0_grid, r1_grid, c1_grid):
        self.q_nominal = capacity_ah * 3600.0  # Convert Ah to Ampere-seconds (Coulombs)
        
        # 1D Dynamic Lookup Interpolators for ALL four parameters
        self.get_ocv = interp.interp1d(soc_grid, ocv_grid, kind='linear', fill_value='extrapolate')
        self.get_r0  = interp.interp1d(soc_grid, r0_grid,  kind='linear', fill_value='extrapolate')
        self.get_r1  = interp.interp1d(soc_grid, r1_grid,  kind='linear', fill_value='extrapolate')
        self.get_c1  = interp.interp1d(soc_grid, c1_grid,  kind='linear', fill_value='extrapolate')

    def simulate(self, current_profile, dt, initial_soc=1.0, initial_v1=0.0):
        """
        Simulates terminal voltage under a given current profile.
        
        Parameters:
            current_profile (array-like): Discharge current in Amps (positive = discharging)
            dt (float): Time step in seconds
            initial_soc (float): Initial State of Charge (0.0 to 1.0)
            initial_v1 (float): Initial RC circuit polarization voltage in Volts
            
        Returns:
            dict containing arrays for time, soc, terminal_voltage, v1, ocv, r0, r1, c1
        """
        n_steps = len(current_profile)
        
        # Preallocate output arrays
        time = np.arange(n_steps) * dt
        soc = np.zeros(n_steps)
        v_terminal = np.zeros(n_steps)
        v1 = np.zeros(n_steps)
        ocv_hist = np.zeros(n_steps)
        r0_hist = np.zeros(n_steps)
        r1_hist = np.zeros(n_steps)
        c1_hist = np.zeros(n_steps)
        
        # Initialize state variables
        current_soc = float(initial_soc)
        current_v1 = float(initial_v1)
        
        for k in range(n_steps):
            i_load = current_profile[k]
            
            # 1. Lookup ALL four dynamic SOC-dependent parameters
            ocv = float(self.get_ocv(current_soc))
            r0  = float(self.get_r0(current_soc))
            r1  = float(self.get_r1(current_soc))
            c1  = float(self.get_c1(current_soc))
            
            # Save history
            soc[k] = current_soc
            ocv_hist[k] = ocv
            r0_hist[k] = r0
            r1_hist[k] = r1
            c1_hist[k] = c1
            v1[k] = current_v1
            
            # 2. Terminal Voltage Equation: V_t = OCV - V1 - (I * R0)
            v_terminal[k] = ocv - current_v1 - (i_load * r0)
            
            # 3. Dynamic RC Time Constant & Exponential Decay Factor for current SoC
            tau1 = r1 * c1
            alpha = np.exp(-dt / tau1)
            
            # 4. State Updates for Next Time Step
            # Update V1: ZOH update with dynamic R1 and dynamic alpha(SoC)
            current_v1 = (current_v1 * alpha) + (i_load * r1 * (1.0 - alpha))
            
            # Update SoC via Coulomb Counting
            current_soc -= (i_load * dt) / self.q_nominal
            current_soc = np.clip(current_soc, 0.0, 1.0)
            
        return {
            "time": time,
            "soc": soc,
            "v_terminal": v_terminal,
            "v1": v1,
            "ocv": ocv_hist,
            "r0": r0_hist,
            "r1": r1_hist,
            "c1": c1_hist
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Sample 1D lookup tables across SoC steps (0.0 to 1.0)
    soc_lookup = np.linspace(0.0, 1.0, 11)
    ocv_lookup = np.array([3.00, 3.45, 3.62, 3.70, 3.75, 3.82, 3.88, 3.95, 4.02, 4.10, 4.20])
    r0_lookup  = np.array([0.015, 0.010, 0.008, 0.006, 0.005, 0.005, 0.005, 0.005, 0.006, 0.007, 0.008]) # Ohms
    
    # Model 2.4 dynamic lookup grids for R1 and C1
    r1_lookup  = np.array([0.025, 0.018, 0.014, 0.012, 0.010, 0.010, 0.010, 0.011, 0.012, 0.014, 0.016]) # Ohms
    c1_lookup  = np.array([1000,  1200,  1400,  1500,  1600,  1600,  1600,  1550,  1500,  1400,  1300]) # Farads

    # Instantiate Model 2.4
    model = BatteryModel24(
        capacity_ah=5.0, # 5000 mAh
        soc_grid=soc_lookup,
        ocv_grid=ocv_lookup,
        r0_grid=r0_lookup,
        r1_grid=r1_lookup,
        c1_grid=c1_lookup
    )
    # Because commercial 6S drone manufacturers don't publish multi-temperature HPPC tables, 
    # you use PyBaMM to run synthetic pulse experiments across temperature steps (e.g., −10∘C to 40∘C). 
    # PyBaMM outputs the exact 2D matrices for R0​(SoC,T), R1​(SoC,T), and C1​(SoC,T).

    # 100-second pulsed current load profile
    t_dt = 0.1
    current = np.zeros(1000)
    current[100:300] = 10.0  # Pulse 1
    current[500:800] = 15.0  # Pulse 2

    res = model.simulate(current_profile=current, dt=t_dt)
    print(f"Model 2.4 Simulation completed. Final SoC: {res['soc'][-1]:.4f}, Final Voltage: {res['v_terminal'][-1]:.3f} V")