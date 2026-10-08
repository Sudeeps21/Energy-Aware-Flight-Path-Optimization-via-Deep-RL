# Literature Gap Table & Related Work Analysis

> **Task S6 Deliverable** (`docs/literature_gap_table.md`)
> Time-boxed survey of prior UAV energy modeling, wind path planning, and battery-aware reinforcement learning research.

---

## 1. Comparative Analysis Matrix

| Paper / Source | Domain | Wind Field | Battery Model | Air Density | Dynamic Payload | Baseline Comparison | Core Limitation / Gap |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rodrigues et al. (Scientific Data 2021)** | Real DJI M100 Dataset | Measured 2D | Isothermal / Voltage log | Fixed | Fixed per flight | N/A (Dataset) | Empirical dataset only; no flight policy optimization or RL. |
| **Zeng et al. (IEEE TWC 2019)** | Aerodynamic Power Model | Constant | Fixed efficiency | Constant sea-level | Static mass | N/A (Analytical) | Focuses on communication energy; ignores thermal battery sag and spatial wind zones. |
| **Abeywickrama et al. (IEEE Access 2018)** | UAV Battery Model | No wind | Empirical C-rate | Sea-level | Static | N/A | Isothermal Li-Po model; no thermal self-heating or dynamic environmental interactions. |
| **Stolaroff et al. (Nature Comms 2018)** | Delivery Life Cycle | Statistical | Constant Wh bucket | Fixed | Delivery dropoff | Classical vehicle comparison | Macro life-cycle analysis; no trajectory planning or real-time flight control. |
| **NASA Wind-RL Path Paper (NASA TRS / IEEE)** | Wind-Aware RL Planning | Spatial wind | Constant Wh bucket | Constant | Static payload | Straight-line path | **Closest prior work**: Wind-only optimization; ignores temperature derating, density, and payload. |
| **Our Work (Multi-Factor Drone RL)** | Multi-Factor Energy RL | 3D ERA5 + Dryden Gusts | PyBaMM 1-RC + Thermal Sag | ISA Altitude Density | Mid-flight Pickup & Drop | A* (B1) & PID Baselines | **Fills the gap**: Evaluates joint multi-factor interactions under fair classical baselines. |

---

## 2. Key Novelty Statements for the Paper

1. **Compound Environmental & Operational Interaction**:
   While prior work treats wind planning, battery electrochemistry, or payload delivery in isolation, this paper evaluates policy performance under the **compound interaction** of spatial wind, temperature-dependent battery capacity sag, ISA altitude air density, and dynamic payload mass change.

2. **Fair Classical Baseline Benchmarking**:
   Unlike RL papers evaluated against weak or wind-blind references, our agent is benchmarked against an **energy-tuned 3D $A^*$ planner (B1)** operating on the exact same spatial wind-cost map with cruise speed optimized for minimum $Wh/km$.

3. **ERA5 Statistical Grounding**:
   Simulation environments are grounded in real-world meteorological data by extracting marginal distributions and gust factors from multi-year ERA5 hourly reanalysis datasets across coastal monsoon, cold altitude winter, and calm temperate regions.

---

## 3. Search Terms & Verified DOIs

- **Rodrigues et al. (2021)**: *In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery*. Scientific Data 8. DOI: [10.1038/s41597-021-00930-x](https://doi.org/10.1038/s41597-021-00930-x)
- **Zeng et al. (2019)**: *Energy minimization for wireless communication with rotary-wing UAV*. IEEE Trans. Wireless Comms. arXiv: [1804.02238](https://arxiv.org/abs/1804.02238)
- **Abeywickrama et al. (2018)**: *Comprehensive energy consumption model for unmanned aerial vehicles*. IEEE Access 6:58383-58394. DOI: [10.1109/ACCESS.2018.2875040](https://doi.org/10.1109/ACCESS.2018.2875040)
- **Stolaroff et al. (2018)**: *Energy use and life cycle greenhouse gas emissions of drones for commercial package delivery*. Nature Comms 9. DOI: [10.1038/s41467-017-02411-5](https://doi.org/10.1038/s41467-017-02411-5)
