# Energy-Aware Drone Delivery with Reinforcement Learning
## Team reference: project facts, systems plan, repository plan, tasks, timeline, resources

| | |
|---|---|
| Version | 1.0, written Wednesday 7 October 2026 |
| Deadline | Submit by **Friday 30 October 2026**. Saturday 31 October is an emergency buffer only, never a planned work day |
| Team | **Member A** = battery, vehicle, PyBullet environment. **Member B** = weather and wind, baselines, training and evaluation. Fill in names: A = ________, B = ________ |
| Overall status | About **25% complete by effort** (range 22 to 27%). The battery subsystem is essentially done; the drone is defined but not yet run inside PyBullet |
| Companion file | `PROJECT_OVERVIEW_AND_ROADMAP.md` (history and technical detail of what has been built) |

**How to use this file.** Both of you read Sections 0 to 4 (shared facts). Member A then reads the "Member A card" in Section 5, Member B reads the "Member B card". Section 6 is the day-by-day timetable for both. Update the status columns in this file at the end of every working day (it is the single source of truth for who does what).

> **Convention used throughout.** *META* = **M**ost **E**ffective **T**actic **A**vailable: the fastest route to a result that a reviewer would still accept as credible. Every META choice below names the tactic *and* the credibility guard that keeps it defensible (Section 5.1).

---

## 0. One-page summary

**What we are doing.** We train a reinforcement-learning agent (PPO) to fly a 1.5 kg delivery drone through a pickup, delivery and return mission while using as little battery energy as possible. The flight takes place in a PyBullet simulation where the energy cost depends on: spatial wind and gusts (statistically grounded in real ERA5 weather data), ambient temperature acting on a realistic battery model (derived from an electrochemical simulation, with self-heating), air density, and a payload that is dropped mid-mission. The agent is compared against fair classical baselines (A* path planning plus PID, including an energy-tuned version).

**Where we are.** Battery model, lookup tables, thermal model, verification (25 of 25 tests), comparison against the reference electrochemical simulator, drone definition, controller scaling and energy model are built. Nothing has yet run inside real PyBullet. The environment, weather module, mission logic, baselines, training, experiments, figures and paper are not started.

**What must happen in 24 days.** Three gates keep us honest:

| Gate | Date | Pass condition | If failed |
|---|---|---|---|
| G1 | Mon 12 Oct | The HL1 drone hovers in PyBullet with the battery stepping every control tick | Switch to Plan B (point-mass environment with our battery and energy models, Section 3.4) |
| G2 | Sat 17 Oct | Baselines finish missions in all scenarios; PPO shows learning on a simple scenario; environment tagged v1.0 | Apply the scope-cut ladder (Section 6.5) |
| G3 | Sat 24 Oct | Code freeze; all final numbers and figures produced | Reduce the evaluation grid, never skip the freeze |

**Minimum viable paper (MVP).** Environment with battery, thermal, wind zones, gusts, payload drop; baseline versus PPO over a reduced scenario grid; three training seeds; battery-fidelity ablation; honest limitations. **Stretch:** observation ablations, generalisation test, forecast-noise test, battery cross-check against measured cell data.

**Top three risks.** (1) First PyBullet run exposes loading or controller problems. (2) The agent "wins" only by flying faster. (3) Training time. Mitigations are in Sections 3.4, 1.7 and 6.5.

---

## 1. The project: everything about it

### 1.1 Research question, hypotheses and claimed contributions

**Question.** How much battery energy can a learned flight policy save on a delivery mission when wind, temperature-dependent battery behaviour (including self-heating), air density and a mid-mission payload drop all interact, compared with fair classical planners?

**Hypotheses (decided now, tested later; do not edit after seeing results).**
- **H1.** A temperature-aware battery model (with self-heating) predicts mission endurance materially differently from a constant-capacity energy bucket, especially in the cold and at high power.
- **H2.** A policy trained in the full environment uses less energy per mission than the strongest fair baseline, with a 95% confidence interval that excludes zero, across the scenario grid.
- **H3.** The saving does not come only from finishing sooner: energy per unit distance (or per second) also improves, or the policy shows identifiable wind and payload-aware behaviour.
- **H4 (stretch).** Giving the policy wind lookahead and pack-temperature observations improves energy use compared with local-only observations.

**Contributions we may claim (only if the evidence supports them).**
1. An open, reproducible simulation environment combining wind, temperature-aware battery with self-heating, air density and dynamic payload for delivery drones.
2. A battery model chain: electrochemical reference (PyBaMM) to a fast circuit surrogate usable in RL, with documented validation and limitations.
3. An evaluation of RL against fair baselines with proper statistics.

**Novelty caution.** The "this combination has not been done" claim comes from quick searches. The closest prior work found was a NASA paper on wind-only, energy-based RL path planning. Task S6 (Section 5.4) is a time-boxed literature check that must finish before the Introduction is written; the novelty sentence is worded from its results.

### 1.2 Mission definition

Origin (takeoff) to **pickup** (payload on board) to **delivery** (payload released; mass drops from 1.5 to 1.0 kg) to **home**. Waypoints and payload mass are randomised per episode within set ranges. A mission lasts about 2 to 4 minutes of simulated flight; the policy acts at about 10 Hz while the PID runs at the physics rate. Starting state of charge is randomised (roughly 25 to 100%) so that cold-weather sag and brownout risk can actually matter; with full batteries a short mission barely changes SoC (hover uses about 2% of the pack per minute).

Success means: all waypoints reached within tolerance, payload released at the delivery point, and no termination by crash, boundary exit, battery cutoff or over-temperature.

### 1.3 Factors considered

| Factor | Physical effect | Model (and source to cite) | Status | Owner |
|---|---|---|---|---|
| **Battery electrochemistry** | Voltage sag, usable energy and power ceiling depend on state of charge, temperature and load | PyBaMM SPMe with Chen2020 parameters (LG M50 21700, 5 Ah) generates pulse data; fitted to a 1-RC equivalent circuit lookup table (20 SoC by 17 temperature points, -20 to 60 degrees C), scaled to 6S1P | Done, validated against PyBaMM, limitations documented | A |
| **Battery self-heating** | Current through R0 and R1 heats the pack, which lowers resistance | Lumped single-node thermal model: heat = I squared times R0 plus V1 squared over R1; heat loss to ambient; pack temperature drives the lookup. Heat-balance basis: Bernardi et al. 1985 (irreversible part only) | Done, 9 tests pass, parameters assumed | A |
| **Ambient (environment) temperature** | Sets starting pack temperature and the temperature the pack cools toward; also changes air density | Zone temperature from real-data scenario cards; separate from self-heating (two different effects) | Interface defined; data pipeline not started | B (data), A (use) |
| **Air density / altitude** | Thinner or warmer air raises hover power | International Standard Atmosphere pressure model plus actual temperature, ideal-gas density | Done in `energy.py` (hover); to be wired to zone temperature | A |
| **Payload (dynamic)** | Mass drops mid-mission; hover power scales about mass to the power 1.5 (observed ratio 1.84 for 1.5 to 1.0 kg) | PyBullet mass and inertia change plus controller mass update | Controller side done and checked offline; PyBullet side not run | A |
| **Wind (spatial zones)** | Drag on relative airspeed; headwinds cost energy and time | Zone map of wind vectors, force on the drone from relative airspeed, drag area assumed 0.07 square metres; zone statistics calibrated to ERA5 | Not started | B (zones), A (force hook) |
| **Gusts / turbulence** | Variable wind forces constant corrections | First-order Gauss-Markov (Dryden-style) gust process added to the mean wind; intensity from ERA5 gust statistics, shape parameters from MIL-HDBK-1797 | Not started | B |
| **Forward-flight propulsion power** | Power changes with speed (blade profile, induced, parasite) | Rotary-wing propulsion power model of Zeng, Xu and Zhang, validated in trend against the CMU DJI Matrice 100 energy dataset | Hover-only version done; forward flight not started | A |
| *Out of scope (future work)* | Rain, icing, motor and propeller heating, battery aging, risk-aware planning | Named in the paper's future-work section | | |

### 1.4 Vehicle and controller ("HL1")

| Quantity | Value | Basis |
|---|---|---|
| Mass | 1.0 kg empty (battery about 0.41 kg of it) plus 0.5 kg payload | Team decision |
| Propellers and arm | 12 inch, arm 0.275 m, X layout | Design choice |
| Thrust and torque constants | kf 2.94e-7, km 5.52e-9 (per RPM squared) | Assumed thrust coefficient 0.10 |
| Hover RPM | 3,537 loaded, 2,888 empty; maximum 5,003 | Derived |
| Thrust-to-weight | 3.0 empty (2.0 loaded) | Design choice |
| Electrical hover power | about 133 to 135 W loaded, 75 to 76 W empty; about 5.9 A (1.17C) loaded | Energy model, assumed efficiency 0.65 times 0.80 plus 5 W avionics |
| Control | Library PID wrapped and rescaled (mass, thrust constants, PWM-to-RPM map, gains); position gains halved after the offline check showed flips at full scale | Offline closed-loop check |

Key lesson to keep: the controller must be told the new mass at the payload drop (otherwise a persistent altitude error of about 12 cm remains).

### 1.5 Battery subsystem in detail

**Chain.** Electrochemical reference (PyBaMM SPMe, Chen2020, LG M50) leads to pulse experiments at 20 SoC by 17 temperature points, which lead to fitted R0, R1, C1 and OCV tables (pack-scaled by 6), which lead to a cleaned table (34 saturated cells repaired) and then the step-based equivalent-circuit model driven by electrical power, with a thermal state on top.

**What the model does each step.** Takes electrical power (watts), pack surroundings temperature, and time step; solves for current; flags a brownout if the pack cannot deliver; updates SoC and polarization voltage; computes heat; updates pack temperature; reports voltage, current, SoC, heat, pack temperature, over-temperature flag, and a maximum-power ceiling.

**Verification and validation record.**

| Check | Result |
|---|---|
| Code correctness (7 tests) | Pass: analytic RC response, rest relaxation, coulomb counting, table orientation and clamping, power closure (300 random states), step-size independence (agreement within 2 mV), determinism |
| Physical behaviour (9 tests) | Pass: sag grows when cold; usable energy falls with cold and with power; efficiency about 99% (C/5), 96% (1C), 92% (2C); low-rate energy 111 Wh against 108 Wh nominal; voltage rebound; no NaN under fuzzing |
| Thermal (9 tests) | Pass: heat balance, steady state, exact cooling constant, stable numerics, smooth zone change, over-temperature flag, parameter recovery |
| Against PyBaMM (runtime to cutoff) | Within about 1% at 1C; up to +10.6% too long at 2C and -10 degrees C; +3.8 to +6.8% at 25 W per cell |
| Against PyBaMM (voltage) | Optimistic by 60 to 145 mV per cell RMS (single fast RC branch cannot show slow diffusion polarization) |
| Cold capacity at low rate | Underestimated (about 2% loss at C/2 and -20 degrees C); PyBaMM itself shows only 97% of room-temperature runtime at -10 degrees C and 1C |
| Self-heating effect | 10 min hover at -10 degrees C ambient: pack warms about 7 K, end voltage +80 mV; hard cold flight to cutoff: runtime +5 to 10% |

**Documented limitations (paper text).** Optimistic voltage; temperature acts mainly through R0; calibrated to a simulator and not to measured pack data; thermal model is single-node with assumed heat capacity (390 J/K) and heat-loss coefficient (0.75 W/K, sweep 0.3 to 1.5); irreversible heat only (our table's OCV has zero temperature dependence, so there is no entropic information to use); about 1.2C loaded hover is high for a high-energy 21700 cell, so check the cell's continuous-discharge rating on the datasheet.

**Optional upgrades (stretch).** (a) Cross-check cold behaviour and R0(T) against measured open cell data (McMaster datasets, Section 7.3). (b) Replay logged environment power traces through PyBaMM (with its lumped thermal option, name to be checked in its documentation) to validate SoC, voltage and temperature on flight-like loads. (c) Two-branch (2-RC) refit from long relaxation curves.

### 1.6 Weather data and wind plan

**Data.** ERA5 hourly reanalysis on single levels from the Copernicus Climate Data Store (free account and API key). Variables: 10 m wind components (u, v), 100 m wind components (for the wind-shear estimate at drone altitude), 2 m temperature, surface pressure, and the 10 m wind gust variable. Native resolution 0.25 degrees (about 28 to 31 km). A no-registration alternative is the Open-Meteo historical API (free for non-commercial use, up to 10,000 calls per day, CC BY 4.0 attribution required; its ERA5 layer has 0.1 to 0.25 degree grids).

**The scale problem (be honest in the paper).** ERA5 cells are about 30 km wide, but a delivery mission is a few hundred metres to a few kilometres. We therefore do **not** replay ERA5 cells as zones. META approach:
1. Pick 2 to 3 real regions and seasons (scenario cards): for example a **monsoon-affected coastal region** (warm, strong gusty wind), a **cold winter region ideally at altitude** (cold, low density), and a **calm temperate reference**.
2. From several years of hourly ERA5 data per region, extract: distribution of wind speed and direction, gust factor (gust over mean wind), temperature distribution, surface pressure (hence density), and the **contrast between neighbouring ERA5 cells** (how much mean wind and temperature differ across a 3 by 3 block of cells).
3. A zone-map generator draws spatially correlated wind and temperature fields for the simulation world whose marginal statistics and spatial contrast match those numbers. Each episode samples one realisation.
4. Gusts: a first-order Gauss-Markov process per axis on top of the mean wind, intensity from the gust factor, length scale from MIL-HDBK-1797 Dryden values.

**What the paper may say.** "Zone fields are synthetic realisations calibrated to ERA5 statistics (hourly, 0.25 degree) for regions X, Y, Z." It must not say "we simulated real weather at 100 m resolution". Note a known data quirk: the ERA5 gust variable is occasionally below the mean wind speed in a given hour, so compute gust factors robustly (use quantiles, discard ratios below 1).

### 1.7 Evaluation: metrics and protocol (the most important section)

**Principle.** A result counts only if it survives the fairness checks below. The energy figure alone is not enough, because an agent that finishes faster uses less energy even if it flies no more efficiently (hover power dominates a multirotor's energy).

**Primary metrics (per episode).**

| Metric | Definition |
|---|---|
| Mission energy (Wh) | Electrical energy drawn at the pack terminals, from takeoff to landing at home |
| Success rate | Fraction of episodes meeting the success criteria in 1.2 |
| Energy saving versus baseline (%) | Paired by scenario and evaluation seed; with a 95% bootstrap confidence interval |
| **Fairness pair** | Energy per metre flown (Wh/km) and mean electrical power (W) alongside total energy, so savings from simply finishing sooner are visible |
| Payload efficiency | Energy per kilogram-kilometre of payload delivered |

**Secondary metrics.**

| Group | Metrics |
|---|---|
| Mission | Completion time, path length, path efficiency (path length divided by sum of straight-line leg distances), waypoints reached, mean ground speed and airspeed |
| Battery safety | Final SoC, minimum SoC, minimum voltage margin to cutoff, brownout events, peak power and peak C-rate, maximum pack temperature, over-temperature events |
| Behaviour diagnostics | Speed against headwind component, fraction of flight time in tailwind, altitude profile, control effort (mean change in commanded target), behaviour around the payload drop |
| Learning | Learning curves per seed, steps to reach 90% of final performance, variance across seeds |
| Model verification | Battery runtime error against PyBaMM, voltage RMSE per cell, tests passing, hover and tracking error in PyBullet, altitude transient at payload drop, energy model trend against the DJI Matrice 100 dataset |

**Baselines (all see the same wind, temperature, payload, battery and seeds).**
- **B0:** straight-line cruise at fixed speed with PID (the naive reference).
- **B1:** A* planning over a wind-cost map with **energy-tuned cruise speed** (speed that minimises energy per distance from our power model). This is the strongest fair classical baseline.
- **B2:** time-optimal (fast) PID, to show whether PPO's advantage is only speed.
- **B3 (stretch):** wind-aware dynamic-programming planner.

**Experiments.**

| ID | Purpose | Needs RL? | Priority |
|---|---|---|---|
| E0 | Battery verification and validation (done; extend with cross-checks) | No | Must |
| E1 | Battery-fidelity ablation: endurance and usable energy predicted by (i) constant-capacity energy bucket, (ii) circuit model isothermal, (iii) circuit model with self-heating, for one scripted flight profile from -20 to 40 degrees C; plus heat-loss coefficient sweep | No | Must |
| E2 | Main comparison: PPO versus B0 to B2 across the grid: wind (calm, windy) by temperature (warm, cold) by payload (light, heavy) = 8 conditions | Yes | Must |
| E3 | Observation ablation: wind lookahead on/off; pack temperature and SoC on/off; payload mass versus yes/no flag | Yes | Should |
| E4 | Generalisation: train on a subset of conditions, test on unseen colder, windier or heavier ones | Yes | Should |
| E5 | Forecast noise on the observed wind (a cheap risk-aware experiment) | Yes | Stretch |
| E6 | Self-heating on/off at evaluation time | Yes | Stretch |

To save compute, train **one policy per seed on a randomised mixture of conditions** (conditions visible in the observation) and evaluate it on each of the 8 conditions, instead of 8 separate trainings.

**Statistical protocol (fixed before running).**
- 3 training seeds (5 if time allows); 100 evaluation episodes per condition per seed; the same evaluation episode seeds for every method (paired comparison).
- Report mean plus or minus standard deviation and 95% bootstrap confidence intervals; Welch's t-test (with Mann-Whitney as a non-parametric check); effect size (Cohen's d); Holm correction across the 8 conditions.
- Pre-stated success criterion for H2: PPO beats B1 by at least 5% mean energy, the interval excludes zero, success rate at least 95%, and the sign of the effect holds in at least 2 of 3 seeds. Negative or mixed results are reported as they are.
- No tuning on evaluation episodes; hyperparameters and the reward are frozen on 21 October (Section 6).

**Reward-hacking checks (run before any result is trusted).** Payload-drop exploit (does the agent gain by dropping early?), lookup-table edge behaviour (hugging the temperature or SoC limits), brownout and cutoff avoidance by stalling, hovering instead of finishing, energy per second identical to baseline (time-only saving).

---

## 2. Where we are: progress and completion

### 2.1 Subsystem status

| Subsystem | Done | Verified how | What is missing |
|---|---|---|---|
| Battery lookup table (PyBaMM, 20 by 17, 6S) | 95% | Cleaned (34 cells repaired); compared against PyBaMM | Optional 2-RC refit, optional measured-data cross-check |
| Battery circuit model with SoC and temperature | 90% | 25 of 25 tests; PyBaMM comparison | Known limitations (Section 1.5) |
| Self-heating (thermal) | 85% | 9 tests; demo | Parameters assumed; optional bench identification tool exists |
| Drone definition (URDF, parameters) | 75% | URDF parser check (10 of 10) | Not yet loaded in real PyBullet |
| Controller (PID wrapper) | 70% | Offline closed-loop check (hover, 3 m move, 8 m/s wind, payload drop) | Not yet run in PyBullet; one assumption about a library file we have not seen (`BaseControl.py`) |
| Energy model (hover, air density) | 60% | Hover numbers checked; trend sanity | Forward flight; wiring to zone temperature; validation against M100 trends |
| Environment class (PyBullet) | 0% | | Everything (Task A2) |
| Mission manager | 0% | | Task A5 |
| Weather pipeline, zone map, gusts | 5% | (B's earlier point-mass prototype has an early version, not yet reviewed) | Tasks B1, B2 |
| Baselines | 5% | (B's prototype PID, wind-blind) | Tasks B3 |
| RL training harness | 10% | (B's prototype scripts) | Task B4 |
| Evaluation harness and statistics | 5% | (B's prototype protocol) | Task B5 |
| Visualisation | 0% | | Task B6 |
| Paper | 5% | Overview document exists | Everything |
| Repository and reproducibility | 25% | Package, `pyproject.toml`, README, tests | GitHub repo, CI, configs, CITATION file |

### 2.2 Overall completion estimate

Method: effort-weighted. Work already done is estimated at about 55 hours-equivalent (battery chain about 35, drone/controller/energy about 15, repository and documents about 5). Remaining planned work is about 195 person-hours (Section 6.1). Completion = 55 divided by (55 plus 195) = **about 22%**. Counting the partial credit for B's earlier prototype gives **about 27%**. Use **"about 25%"** as the working figure. This is a rough estimate (plus or minus 30% on the remaining hours), but the direction matters: most of the research value (environment, training, experiments, writing) is still ahead, so the schedule has to be protected.

---

## 3. Systems and models plan (what the simulation consists of)

### 3.1 Architecture

```
                    +-----------------------------------------------+
                    |               Gymnasium environment            |
                    |   (HeavyLiftBatteryEnv, owner: A)              |
                    +-----------------------------------------------+
                      ^        ^         ^          ^          ^
  action (target)     |        |         |          |          |  observation, reward, info
                      v        v         v          v          v
        +-----------+  +-----------+ +-----------+ +---------+ +-----------+
        | Mission   |  | Vehicle + | | Weather   | | Energy  | | Battery   |
        | manager   |  | PID (A)   | | field (B) | | model   | | pack (A)  |
        | (A)       |  | PyBullet  | | wind,gust,| | (A)     | | ECM +     |
        | waypoints,|  | HL1 drone | | temp,     | | power,  | | thermal   |
        | payload,  |  | mass swap | | density   | | density | | lookup    |
        | success   |  |           | |           | |         | | table     |
        +-----------+  +-----------+ +-----------+ +---------+ +-----------+
                                          ^                         ^
                         scenario cards   |                         | lookup table
                         (ERA5 statistics)|                         | (PyBaMM, offline)
                              +-----------+--+                +-----+---------+
                              | data pipeline |                | PyBaMM SPMe   |
                              | (B, offline)  |                | Chen2020 (A,  |
                              +---------------+                | offline)      |
                                                               +---------------+

   Outside the environment:  baselines (B)  |  PPO training harness (B)  |  evaluation + statistics (B)  |  plots (B)
```

### 3.2 What happens in one control step (about 10 Hz policy, higher-rate PID)

1. The policy (or baseline) outputs a target (position or velocity command).
2. The mission manager reports the active waypoint and whether a payload event is due; if the payload drops, mass and inertia change in PyBullet and the controller is told the new mass.
3. The weather field returns wind, gust and ambient temperature at the drone position and time; air density follows from temperature and altitude.
4. The PID converts the target into rotor RPMs; PyBullet advances the physics several sub-steps, with the wind drag force applied at every sub-step on relative airspeed.
5. The energy model turns the rotor thrusts and air density into electrical power (watts).
6. The battery steps with that power, the pack temperature and the ambient temperature; it returns voltage, SoC, heat, pack temperature and flags (brownout, over-temperature, depleted).
7. The environment builds the observation, computes the reward (success minus energy, with a small shaping term), and ends the episode on success, crash, boundary exit, battery depletion or over-temperature.

### 3.3 Interface contract (agree on Wed 7 and Thu 8 October, then freeze)

All units SI unless stated. Changing a name or unit after 9 October needs both members' agreement and an entry in `docs/decisions.md`.

| Interface | Inputs | Outputs |
|---|---|---|
| **Weather field** (B provides) | position (x, y, z in metres), time (s), scenario id | mean wind vector (m/s, world frame), gust vector (m/s), ambient temperature (degrees C), air pressure (Pa); a second call returns a lookahead set of wind samples around the drone |
| **Scenario card** (B provides, data file) | scenario id | wind speed and direction distributions, gust factor, temperature distribution, pressure or altitude, spatial contrast statistics, source region and period |
| **Mission** (A provides) | seed, scenario id | waypoints, payload mass, start SoC, start pack temperature; per step: active waypoint, events (pickup, drop), done flags |
| **Battery** (A, done) | reset (SoC, optional pack temperature); step (power in W, time step in s, ambient in degrees C) | voltage, current, SoC, heat (W), pack temperature, brownout, depleted, over-temperature, maximum power |
| **Energy model** (A) | rotor thrusts or speed and mass, air density, wind-relative speed | electrical power (W) |
| **Info dictionary** (A, per step) | | energy (Wh), SoC, pack temperature, pack voltage, power (W), mass (kg), ambient temperature, air density, wind vector, active waypoint, success flag, termination reason |
| **Results table** (B defines, A writes during experiments) | | one row per episode: scenario id, seed, method, episode id, energy (Wh), success, time (s), distance (m), final SoC, minimum SoC, minimum voltage margin, maximum pack temperature, brownouts, mean power (W), mean speed (m/s) |

### 3.4 Fidelity plans

| | **Plan A (primary)** | **Plan B (fallback)** |
|---|---|---|
| Physics | PyBullet 6-DoF with HL1 drone and wrapped PID | B's point-mass environment (own integrator) |
| Battery, thermal, energy | Same modules | Same modules (they do not depend on PyBullet) |
| Trigger | Default | If G1 (12 October) fails, or PyBullet throughput is too low to finish three seeds by 22 October |
| What PyBullet is still used for | | Controller and energy-model validation section only |
| Paper wording | "6-DoF simulation with PID inner loop" | "Point-mass kinematic model with validated energy and battery models, validated against 6-DoF PyBullet hover and transit" |
| Credibility | Highest | Acceptable, common in UAV energy-optimisation work; disclose clearly |

Plan B costs roughly 1 to 2 days of re-integration, because B's environment already trains and the battery and energy modules are simulator-independent. **B therefore develops the training and evaluation harness on the point-mass environment first (Tasks B4 and B5) and switches the environment id when A's environment passes G1.** This also keeps B productive while A integrates PyBullet.

### 3.5 Files and models we will need (checklist)

| Item | Type | Owner | Exists? |
|---|---|---|---|
| Battery lookup table (original and cleaned) | data (.npz) | A | Yes |
| Battery model, thermal model, thermal identification | code | A | Yes |
| HL1 parameters, HL1 URDF | code, asset | A | Yes (untested in PyBullet) |
| PID wrapper | code | A | Yes (untested in PyBullet) |
| Energy model (hover) | code | A | Yes |
| Energy model (forward flight, density wiring) | code | A | No |
| Environment class | code | A | No |
| Mission manager, scenario randomisation | code | A | No |
| Scenario cards (ERA5 statistics) | data (small JSON or CSV) | B | No |
| Weather data download and processing | code | B | No |
| Zone-map and gust generator, weather field | code | B | No |
| Baselines B0 to B2 | code | B | No |
| PPO training configuration and harness | code, configs | B | Partly (prototype) |
| Evaluation harness, statistics, results tables | code | B | Partly (prototype protocol) |
| Plotting scripts | code | B | No |
| Unit tests for every module above | code | owner of module | Battery yes |
| Paper source and bibliography | LaTeX or Markdown | both | No |

---

## 4. GitHub repository plan

Use one repository (private until submission, then public with a licence so the work is reproducible). Keep the **package name `drone_energy`** and the existing structure; add folders as modules appear.

```
drone-energy-rl/                     <- repository root (currently drone_energy_rl on disk)
|-- README.md                        what it is, install, quick start, results link
|-- LICENSE                          choose an open licence (e.g. MIT) after checking dependency licences
|-- CITATION.cff                     how to cite our work
|-- pyproject.toml                   package definition and dependencies
|-- requirements-lock.txt            exact versions used for the reported runs
|-- .gitignore                       venv, results (large), data/raw, caches, checkpoints
|-- .github/workflows/ci.yml         runs the unit tests on every push (free for public repositories)
|
|-- drone_energy/                    the importable package
|   |-- battery/                     thevenin.py, thermal_fit.py            [A]  done
|   |-- drones/                      hl1.py                                  [A]  done
|   |-- assets/                      hl1.urdf                                [A]  done
|   |-- control/                     heavy_pid.py                            [A]  done
|   |-- physics/                     energy.py (power, density), drag.py     [A]  partly
|   |-- weather/                     field.py, zones.py, gusts.py, cards.py  [B]  to do
|   |-- mission/                     mission.py (waypoints, payload, success)[A]  to do
|   |-- envs/                        heavy_lift_env.py                       [A]  to do
|   |-- baselines/                   astar_cost_map.py, cruise.py, timeopt.py[B]  to do
|   |-- rl/                          train.py, callbacks.py, configs loader  [B]  to do
|   |-- eval/                        evaluate.py, stats.py, tables.py        [B]  to do
|   `-- viz/                         plots.py, trajectories.py               [B]  to do
|
|-- configs/                         plain-text experiment settings (one file per experiment)
|   |-- scenarios/                   calm_warm.yaml, windy_cold.yaml, ...    [B]
|   |-- ppo/                         ppo_default.yaml                        [B]
|   `-- experiments/                 e1.yaml, e2.yaml, ...                   [both]
|
|-- data/
|   |-- battery_luts/                original and clean .npz                 [A]  done
|   |-- scenario_cards/              small processed ERA5 statistics (committed) [B]
|   |-- raw/                         downloaded ERA5 and datasets (NOT committed; script re-downloads) 
|   `-- external/                    notes and download scripts for third-party datasets (M100, McMaster)
|
|-- scripts/                         command-line tools (keep existing battery scripts here)
|   |-- clean_battery_lut.py, validate_vs_pybamm.py, demo_self_heating.py,
|   |   fit_thermal_params.py, test_battery.py, check_urdf.py, check_pid_offline.py   [A]  done
|   |-- BatteryTemperatureMatrix.py  the PyBaMM table generator              [A]  move here
|   |-- weather_download.py, weather_make_cards.py                          [B]
|   |-- train.py, run_baselines.py, run_experiments.py, make_figures.py     [B]
|   `-- throughput_check.py          measures simulation steps per second   [A]
|
|-- tests/                           unit tests: test_battery_model.py (done), test_weather.py,
|                                    test_mission.py, test_env_smoke.py, test_baselines.py, test_stats.py
|-- notebooks/                       exploration only; nothing the paper depends on
|-- results/                         run outputs: logs, checkpoints, tables (gitignored except small summary tables)
|   `-- summary/                     committed CSV tables and final figures
|-- docs/
|   |-- decisions.md                 dated log of design decisions and who made them
|   |-- assumptions.md               the assumed-values registry (Appendix A of this file)
|   |-- model_cards/                 one page per model: purpose, source, validity range, limits
|   |-- literature_gap_table.md      output of Task S6
|   `-- TEAM_PLAN_TASKS_TIMELINE_RESOURCES.md   this file
`-- paper/
    |-- main.tex (or main.md), refs.bib
    |-- figures/                     generated by scripts only (never hand-edited)
    `-- tables/
```

**Rules for the repository.**
- `main` always passes the tests. Work happens on short-lived branches named `a/short-name` or `b/short-name` and is merged by pull request, reviewed by the other member within 24 hours.
- Large or regenerable files are never committed (raw weather data, checkpoints, logs). Everything that produces a number in the paper is a script plus a config file that is committed.
- Every figure and table in the paper comes from a script in `scripts/` and an experiment config; record the git commit hash in the paper's reproducibility statement.
- If you move existing scripts into sub-folders, remember that two of them (the cleanup and the PyBaMM comparison) find the project root from their own location and must be updated accordingly.
- Tag the code freeze as `v1.0-freeze` (24 October) and the submission as `v1.0-submission`.

---

## 5. Roles and tasks (who does what, where, how, when)

### 5.1 The META rules and the credibility test

**META tactics we use everywhere.**

| Tactic | Example in this project | Credibility guard that must accompany it |
|---|---|---|
| **Cite instead of invent** | Rotor power model from Zeng et al.; atmosphere from the International Standard Atmosphere; gust shape from MIL-HDBK-1797; battery chain from PyBaMM and Chen2020 | Name the source in the code header and model card; state which parameters are ours and assumed |
| **Calibrate to a reference, do not hand-tune** | Battery table from PyBaMM; weather statistics from ERA5 | Report the agreement numbers, including where the model is worse |
| **Surrogate for speed, reference for truth** | Circuit battery model in the loop (fast); PyBaMM only offline for validation | Keep the validation table in the paper |
| **Sensitivity sweep instead of measurement** | Heat-loss coefficient, drag area, efficiency chain | Sweep, report how much conclusions change; list in the assumptions registry |
| **Two fidelity levels with a fallback** | PyBullet primary, point-mass Plan B | Disclose which was used |
| **Domain randomisation instead of many specialist trainings** | One policy trained on a mixture, evaluated per condition | Report per-condition results and the training distribution |
| **Develop on the cheap simulator, finalise on the real one** | B builds harness on point-mass first | Re-run final numbers on the final environment |
| **Reuse proven libraries** | Stable-Baselines3 PPO with default settings, scipy statistics | Pin versions, record seeds |

**The credibility test (a META choice is allowed only if all five answers are yes).**
1. **Traceable:** is it cited, or derived in a way someone can follow?
2. **Reproducible:** are code, config, seed and data source committed?
3. **Disclosed:** is the limitation written down in the paper?
4. **Fair:** no tuning on evaluation episodes; baselines are not deliberately weak?
5. **Proportionate:** are we claiming no more than the evidence shows?

**Red lines (never, even under deadline pressure).** Inventing or "smoothing" data. Reporting only the best seeds. Changing the reward or hyperparameters after seeing evaluation results. Citing a paper we have not opened. Copying text or figures without permission or citation. Hiding a limitation. Using paid or licence-restricted data or software. Leaving out AI-assistance disclosure where the institution or venue requires it.

**Effort legend.** Hours are estimates for one person (plus or minus 30%). "Done when" is the acceptance test; the other member checks it.

---

### 5.2 MEMBER A card (battery, vehicle, PyBullet environment)

**A's one-line mission:** make the HL1 drone fly in PyBullet with the battery in the loop, then turn it into a Gymnasium environment the training code can use.

#### A1. PyBullet smoke test and Gate 1 (10 h, 7 to 12 October)
- **Where:** `scripts/check_urdf.py`, `scripts/check_pid_offline.py`, `scripts/test_battery.py`, `tests/test_battery_model.py` (run first on your machine); new `scripts/throughput_check.py`; library clone `gym-pybullet-drones` (unchanged).
- **How:** (1) Run all existing checks on your machine; fix any version differences. (2) Open `BaseControl.py` in the library and confirm it exposes the gravity, thrust and torque constants the PID wrapper overrides. (3) Load HL1 in PyBullet through the library, using a subclass or small enumeration so the library itself stays untouched. (4) Hover 60 s with the wrapped PID, stepping the battery each control tick. (5) Record physics steps per second (this decides the experiment budget). (6) Set PyBullet's default linear and angular damping to zero and note that the library's own drag mode depends on rotor speed, not wind, so we apply our own drag.
- **META:** start from the library's own hover example and change one thing at a time; do not write a loader from scratch.
- **Credibility guard:** record library version (2.2.0), Python and PyBullet versions in `docs/decisions.md`; if any library file must change, do it as a documented subclass, never an edit.
- **Done when (G1):** hover error under 5 cm over 60 s, battery SoC and temperature change as the offline estimate predicts (about 135 W), throughput number written down.
- **If stuck at the end of 11 October:** tell B immediately; Plan B starts 12 October.

#### A2. Environment class (16 h, 9 to 16 October)
- **Where:** `drone_energy/envs/heavy_lift_env.py`, `tests/test_env_smoke.py`, `docs/model_cards/env.md`.
- **How:** subclass the library's RL aviary; policy at about 10 Hz with the library's PID action type; observation: position, velocity, local wind, ambient temperature, pack temperature, SoC, payload mass (not a yes/no flag), direction and distance to the active waypoint, and (later, from B) wind lookahead; step: apply wind drag each physics sub-step, compute power, step battery, build info dictionary (Section 3.3), terminate on success, crash, boundary exit, depletion, over-temperature; reset: draw mission and weather realisation from a seed.
- **META:** minimal viable observation first; add lookahead only after G2. Use B's `WeatherField` through a stub (uniform wind) until B delivers.
- **Credibility guard:** deterministic given a seed; all constants imported from `hl1.py`; observation and action spaces documented in the model card.
- **Done when:** random-action and PID smoke tests run 1,000 steps with no exception; info fields correct; reset with the same seed reproduces the same episode.

#### A3. Payload mechanics inside PyBullet (5 h, 12 to 14 October)
- **Where:** environment and mission modules; `tests/test_env_smoke.py`.
- **How:** at the delivery event change the body's mass and inertia together and call the controller's mass update on the same step; compare altitude transient and hover power before and after with the offline check (expected: altitude held within about 1 cm; power 135 W to 76 W).
- **META:** reuse the exact test numbers from the offline check as the pass criteria.
- **Credibility guard:** also run the "controller not told" case once and keep it as a documented negative control.
- **Done when:** the numbers match the offline expectations within 25%.

#### A4. Energy model upgrade and validation (10 h, 13 to 16 October)
- **Where:** `drone_energy/physics/energy.py`, `docs/model_cards/energy.md`, `data/external/` notes for the M100 dataset.
- **How:** implement the rotary-wing propulsion model of Zeng, Xu and Zhang (blade profile, induced, parasite power as a function of airspeed) with HL1 geometry and density from the zone temperature; keep the existing hover model as a cross-check (the two must agree at zero speed). Validate **trends** against the CMU DJI Matrice 100 dataset (energy against speed, payload and wind): compare normalised quantities (for example power divided by mass to the power 1.5), because the M100 is a larger vehicle.
- **META:** cite the model and take its structure and typical constants; do not derive new aerodynamics. Experimental support for the model exists in the follow-up paper by Gao et al. (arXiv 2005.01305).
- **Credibility guard:** report where trends disagree; do not tune constants to match the dataset; list rotor constants not known for HL1 as assumptions with a sensitivity sweep.
- **Done when:** model reproduces the hover value, shows the expected U-shaped power-speed curve, and the validation plot with an honest discussion exists.

#### A5. Mission manager and randomisation (7 h, 11 to 14 October)
- **Where:** `drone_energy/mission/mission.py`, `tests/test_mission.py`.
- **How:** a plain Python module (no PyBullet dependency): waypoint sequence (origin, pickup, delivery, home), arrival tolerances, payload mass draw, start SoC draw, start pack temperature, success and failure definitions, event flags. Ranges are set in the Oct 14 design freeze.
- **META:** pure logic with unit tests, so it also works in Plan B.
- **Credibility guard:** every random draw uses the episode seed; ranges and tolerances are written in `configs/`.
- **Done when:** 1,000 seeded draws are reproducible and within the stated ranges.

#### A6. Battery deliverables for the paper (8 h, 19 to 23 October)
- **Where:** `scripts/` (new `battery_fidelity_ablation.py` and `battery_replay_validation.py`), `docs/model_cards/battery.md`, `results/summary/`.
- **How:** (1) **E1:** one scripted flight profile (hover, cruise, drop, hover, land) run from -20 to 40 degrees C with the three battery variants (energy bucket, circuit isothermal, circuit plus self-heating); plot endurance and usable energy. (2) Heat-loss coefficient sweep (0.3, 0.75, 1.5 W/K). (3) Replay validation: take five logged power traces from the environment, run them through PyBaMM and through our model, compare SoC, voltage and temperature. (4) Write the battery model card.
- **META:** E1 needs no reinforcement learning, so it can run while training occupies the machines.
- **Credibility guard:** state the model's known limits next to the results (optimistic voltage, cold low-rate underestimate).
- **Done when:** E1 figure and table, sweep, replay table and model card exist.

#### A7. STRETCH: cross-check against measured cell data (8 h, only if ahead on 22 October)
- **Where:** `data/external/`, `scripts/`, `docs/model_cards/battery.md`.
- **How:** use the free McMaster datasets (Turnigy Graphene 5000 mAh and Samsung INR21700 30T, tested at 40, 25, 10, 0, -10, -20 degrees C with HPPC pulses and C/20 discharge). Compare measured cold-capacity ratios and R0 against temperature trends with our model's trends. Trend-level only: different cells.
- **META:** a measured-data cross-check removes the biggest limitation ("calibrated only to a simulator") at low cost.
- **Credibility guard:** cite the dataset DOI and state it is a different cell.
- **Done when:** one figure comparing cold-capacity ratio and R0 trends.

---

### 5.3 MEMBER B card (weather and wind, baselines, training and evaluation)

**B's one-line mission:** supply a credible real-data-grounded weather world, fair baselines, and a training and evaluation machine that gives statistically sound results.

#### B1. Weather data pipeline and scenario cards (12 h, 7 to 13 October)
- **Where:** `scripts/weather_download.py`, `scripts/weather_make_cards.py`, `drone_energy/weather/cards.py`, `data/scenario_cards/` (committed), `data/raw/` (not committed).
- **How:** (1) Register for the free Copernicus CDS account today and submit ERA5 requests immediately (queues can take hours). (2) In parallel, use Open-Meteo historical data (no registration) for a quick first cut. (3) Choose 2 to 3 regions and seasons by 9 October (monsoon coast; cold winter region, ideally at altitude; calm temperate reference). (4) For each region extract 5 or more years of hourly: 10 m and 100 m wind components, 2 m temperature, surface pressure, 10 m gust; compute distributions, gust factor (robust), shear exponent from 10 m versus 100 m, and the contrast between neighbouring cells (3 by 3 block). (5) Save one small card per region.
- **META:** statistics from real data instead of replaying weather; two sources so a delay in one does not block.
- **Credibility guard:** write dataset name and version, access date, period, variables and licence text in each card; include the Copernicus and Open-Meteo attributions; state the sub-grid assumption (Section 1.6).
- **Done when:** three cards exist, a plot of each card's distributions is in `results/summary/`, and B has written a one-paragraph description for the paper.

#### B2. Zone map, gusts and the WeatherField interface (10 h, 9 to 15 October)
- **Where:** `drone_energy/weather/zones.py`, `gusts.py`, `field.py`, `tests/test_weather.py`.
- **How:** a spatially correlated random-field generator whose marginal statistics and neighbour contrast match a card; first-order Gauss-Markov gusts per axis; the interface in Section 3.3 (including the lookahead call); a stub version for A on 9 October that returns uniform wind and a card temperature.
- **META:** simple smoothed random fields plus a cited gust model; no turbulence solver.
- **Credibility guard:** a unit test shows generated mean, spread and gust factor match the card within a stated tolerance.
- **Done when:** A's environment runs with B's field in place of the stub; statistics test passes.

#### B3. Baselines B0 to B2 (12 h, 10 to 17 October)
- **Where:** `drone_energy/baselines/`, `tests/test_baselines.py`.
- **How:** B0 straight-line cruise; B1 A* on a wind-cost map with cruise speed tuned to minimise energy per distance using A's power model; B2 time-optimal fast PID. All use the same environment interface as the agent. Develop on the point-mass environment, then connect to A's environment after 12 October.
- **META:** grid A* with a cost equal to estimated energy of the leg (from the power model); no new planning research.
- **Credibility guard:** **A reviews baseline strength.** The baseline must be at least as informed about the wind map as the agent's observation; if the agent has lookahead, the planner gets the same map. No deliberately weak baseline.
- **Done when (G2):** each baseline finishes at least 95% of missions in all 8 conditions.

#### B4. PPO training harness (9 h, 8 to 16 October)
- **Where:** `drone_energy/rl/`, `configs/ppo/`, `scripts/train.py`.
- **How:** Stable-Baselines3 PPO with parallel environments; configuration files per run; observation normalisation with saved statistics; periodic evaluation on separate validation seeds; logging to TensorBoard and CSV; checkpoints. If the warm-to-cold curriculum is used, trigger the switch on "steps greater than or equal to" a threshold with a done flag (an exact-equality check can be skipped when several environments step together). Budget at most one day of hyperparameter tuning (learning rate, rollout length, entropy), done on validation seeds only.
- **META:** library defaults first; tune only three settings; one policy per seed over a randomised condition mixture.
- **Credibility guard:** tuning episodes are never evaluation episodes; all seeds and configs are committed; learning curves for every seed are kept.
- **Done when (G2):** PPO shows clear learning on one simple scenario on A's environment.

#### B5. Evaluation harness, statistics and tables (11 h, 12 to 19 October; re-run 22 to 24 October)
- **Where:** `drone_energy/eval/`, `scripts/run_experiments.py`, `configs/experiments/`, `tests/test_stats.py`, `results/summary/`.
- **How:** implement the results table (Section 3.3), paired comparisons by evaluation seed, bootstrap confidence intervals, Welch and Mann-Whitney tests, Cohen's d, Holm correction, and the pre-registered criteria file. Output CSV tables and LaTeX-ready tables.
- **META:** `scipy.stats` and NumPy; no custom statistics.
- **Credibility guard:** the pre-stated criteria file (Section 1.7) is committed **before** any final run; a unit test checks the statistics on synthetic data with a known answer.
- **Done when:** one command reproduces all tables from the saved episode logs.

#### B6. Visualisation (8 h, 19 to 25 October)
- **Where:** `drone_energy/viz/`, `scripts/make_figures.py`, `paper/figures/`.
- **How:** figures F1 to F11 listed in Section 8: 3-D trajectories (agent versus baselines), wind map with paths, energy by condition with confidence intervals, energy against payload mass, SoC and pack-temperature timelines, learning curves, ablation bars; optional short demo video.
- **META:** Matplotlib only, scripts generate every figure; start on baseline data from 19 October so only the PPO series needs adding at the end.
- **Credibility guard:** axes labelled with units, uncertainty shown, colour-blind-safe palette, no hand-edited images.
- **Done when:** all figures regenerate from one command.

---

### 5.4 SHARED tasks (both members)

| ID | Task | Where | How (and META) | When | Hours each |
|---|---|---|---|---|---|
| **S1** | Repository, CI, environment parity | GitHub, `pyproject.toml`, `requirements-lock.txt`, `.github/workflows/` | A pushes the current project; B clones and runs the battery tests; pin versions; add the CI test workflow; both confirm identical test results on both machines | 7 to 9 Oct | 3 |
| **S2** | Design meetings: (a) contract and scenario matrix, (b) design freeze 1 (observation, reward, mission ranges), (c) design freeze 2 (hyperparameters, last reward change) | `docs/decisions.md` | 45 to 90 minutes each, outcome written the same day; **reward guidance:** success minus energy with a small progress-shaping term, no large per-step time penalty (it makes the agent a time minimiser) | (a) 7 to 8 Oct, (b) 14 Oct, (c) 21 Oct | 4 total |
| **S3** | Experiment runs and monitoring | `results/`, `configs/experiments/` | A and B each run different seeds on their own machines; check learning curves daily; run the reward-hacking checks (Section 1.7) on one trained policy per seed | 18 to 23 Oct | 6 |
| **S4** | Paper writing | `paper/` | Draft each section as soon as its experiment is done; split per Section 8; swap sections for review on 27 October. META: write Methods now from the model cards, Results last | 14 to 29 Oct | 18 |
| **S5** | Final QA, reproducibility and submission | repo, `paper/` | Fresh-clone test on a clean machine; README check; reference verification (Appendix B); AI-use disclosure; licence; tag `v1.0-submission` | 28 to 30 Oct | 4 |
| **S6** | Literature check (time-boxed) | `docs/literature_gap_table.md` | 3 hours each. A searches battery and energy models plus drone energy datasets; B searches RL for UAV energy, wind-aware planning and the NASA wind-RL paper. Search terms: "reinforcement learning multirotor minimum energy wind field", "UAV energy temperature battery path planning", "delivery drone payload energy reinforcement learning". Sources: Google Scholar, Semantic Scholar, arXiv, NASA Technical Reports Server, IEEE Xplore (through your institution). Output: table of paper, what it does, what it does not do | 8 to 12 Oct | 3 |

### 5.5 Workload balance

| | A | B |
|---|---|---|
| Own tasks | 56 h (A1 10, A2 16, A3 5, A4 10, A5 7, A6 8) | 62 h (B1 12, B2 10, B3 12, B4 9, B5 11, B6 8) |
| Shared tasks | 38 h | 38 h |
| **Total** | **94 h** (102 h with stretch A7) | **100 h** |
| Capacity (Section 6.1) | about 121 h | about 121 h |
| Slack | about 22% | about 17% |

If B falls behind, A takes over **B6 (visualisation)** after 24 October; if A falls behind, B takes over **A5 (mission manager)** on 11 October at the latest (it is plain Python). Decide on the 14 October freeze meeting, not in a panic.

---

## 6. Predicted timeline and timetable

### 6.1 Assumptions and capacity

Calendar: Wednesday 7 October to Friday 30 October (submission), Saturday 31 October buffer. That is 18 weekdays and 7 weekend days before the buffer day. **Assumed availability per person: 4 hours on a weekday, 7 hours on a weekend day = about 121 hours each, about 242 in total.** Planned work is about 194 hours (Section 5.5), leaving about 20% slack. If your real availability is lower, say so on 8 October and apply the scope-cut ladder (6.5) immediately, not at the last minute.

### 6.2 Phases and gates

| Phase | Dates | Goal | Exit gate |
|---|---|---|---|
| **1. Foundations** | Wed 7 to Mon 12 Oct | Repository live; HL1 flying in PyBullet with the battery; ERA5 requested and first scenario cards; harness started on the point-mass environment; literature gap table | **G1** (Mon 12 Oct) |
| **2. Integration** | Tue 13 to Sat 17 Oct | Environment with wind, gusts, payload, mission; baselines; PPO smoke test; design freeze 1 | **G2** (Sat 17 Oct): environment tagged v1.0 |
| **3. Training and experiments** | Sun 18 to Sat 24 Oct | Three seeds trained; E1 and E2 (and E3, E4 if possible); reward-hacking checks; design freeze 2; code freeze | **G3** (Sat 24 Oct): `v1.0-freeze` |
| **4. Writing and submission** | Sun 25 to Fri 30 Oct | Draft, review, polish, reproducibility, submit | Submission Fri 30 Oct (buffer Sat 31) |

### 6.3 Day-by-day timetable

| Date | Member A (you) | Member B (friend) | Joint, gates and decisions |
|---|---|---|---|
| **Wed 7 Oct** | S1: push the repository to GitHub. A1: run all existing checks on your machine; read `BaseControl.py` | S1: clone and set up; register for CDS today and submit ERA5 requests; start Open-Meteo pulls | S2a kickoff (45 min): roles, interface draft |
| **Thu 8 Oct** | A1: load HL1 in PyBullet. S6: literature (1.5 h) | B1: first scenario cards from Open-Meteo. B4: start the harness on the point-mass environment. S6 (1.5 h) | S2a finalise contract and scenario matrix (60 min) |
| **Fri 9 Oct** | A1: hover with the wrapped PID; measure steps per second. S1: CI working. Provide hook for B's weather stub | B1: choose regions. B2: deliver a stub WeatherField to A. S1 done. S6 | Decide: regions, mission geometry, hours per person |
| **Sat 10 Oct** | A1: battery stepping in the hover loop. Start A2 skeleton | B2: zone generator v0 and gusts v0. B3: start baselines B0 and B1 on the point-mass environment | |
| **Sun 11 Oct** | A2: reset, step, info. A5: mission manager. **End-of-day self-check: will G1 pass tomorrow?** | B3: B0 and B1 running. B1: ERA5 data arrives, cards v1. S6: finish gap table | If A says no: Plan B starts Monday |
| **Mon 12 Oct** | **G1 delivery.** A3: start payload | B2: integrate field with A's environment. B5: evaluation skeleton | **G1 review (20 min); Plan B decision** |
| **Tue 13 Oct** | A2: wind drag hook, observation. A5: finish mission | B1: final cards. B2: gusts finished. B5: results table schema | |
| **Wed 14 Oct** | A3: finish payload test. A4: start forward-flight energy model | B3: B2 baseline (time-optimal). B4: PPO smoke on point-mass. Related-work outline | **S2b design freeze 1** (observation, reward, mission ranges) |
| **Thu 15 Oct** | A4: model and trend check against M100. A2: integrate real WeatherField with lookahead | B2: statistics test done. B3: A* on wind-cost map | |
| **Fri 16 Oct** | A2 finish (environment v1). A4 finish | B4: switch harness to A's environment; first PPO smoke run (about 200k steps, simple scenario) | |
| **Sat 17 Oct** | Tag environment v1.0. Fix bugs found by B. Draft Methods (battery, vehicle) | Baselines across all 8 conditions. Check PPO learning curve. Draft Related Work | **G2 review; go or no-go for full training** |
| **Sun 18 Oct** | Launch training seed 3 on your machine | Launch seeds 1 and 2 on your machine | S3 starts. No physics changes after tomorrow |
| **Mon 19 Oct** | A6: E1 battery-fidelity ablation (no RL needed) | B5: evaluation harness complete. B6: figures from baseline data | Daily learning-curve check |
| **Tue 20 Oct** | A6: heat-loss sweep and replay validation | B6: figures | Reward-hacking checks on interim checkpoints |
| **Wed 21 Oct** | Evaluate interim policies; start E3 on spare capacity | Same | **S2c design freeze 2**: last reward or hyperparameter change |
| **Thu 22 Oct** | A6: battery model card. E3 and E4 runs. A7 stretch only if ahead | Finalise training; B5 tables; start final evaluations | |
| **Fri 23 Oct** | Final E2 evaluations (A's seed) | Final E2 evaluations (B's seeds); statistics | Check the pre-stated criteria |
| **Sat 24 Oct** | Validation figures final | Main figures final | **G3 code freeze**, tag `v1.0-freeze` |
| **Sun 25 Oct** | Writing sprint 1: Methods polish, Limitations | Writing sprint 1: Results and Discussion | |
| **Mon 26 Oct** | Abstract, battery limitations, references | Introduction, Related Work | |
| **Tue 27 Oct** | Draft v0.9 complete; review B's half | Draft v0.9 complete; review A's half | Swap and review |
| **Wed 28 Oct** | Revisions; S5 fresh-clone test | Revisions; reference verification (Appendix B) | |
| **Thu 29 Oct** | Final proofread; submission package; AI-use disclosure | Final proofread; supplementary; figures check | Lockdown at 18:00 (typos only afterwards) |
| **Fri 30 Oct** | **Submit** (morning). Tag `v1.0-submission`; archive repository | Confirm submission; backup | |
| **Sat 31 Oct** | Buffer, emergencies only | Buffer, emergencies only | |

**Weekly view.** Week 1 (7 to 11 Oct): foundations. Week 2 (12 to 18 Oct): integration. Week 3 (19 to 25 Oct): training, experiments, freeze. Week 4 (26 to 31 Oct): writing and submission.

### 6.4 Critical path and slack

The chain that decides the finish date is: A1 (PyBullet works) then A2 (environment) then G2 then training (18 to 22 Oct) then final evaluation (23 Oct) then freeze (24 Oct) then writing. B's tasks (weather, baselines, harness) run in parallel and are protected by the point-mass development path, so **the slack on A's chain is only about 1 to 2 days**; that is why G1 exists on 12 October. Writing begins on 14 October for the sections whose content is already settled (Methods, Related Work), so only Results and Discussion wait for the freeze.

### 6.5 Scope-cut ladder and Plan B

Apply in order whenever a gate is missed by a day or more.

| Step | Cut | Saves |
|---|---|---|
| 1 | Drop stretch items: E5, E6, A7, B3 dynamic-programming baseline | about 20 h |
| 2 | Drop E4 (generalisation test) | about 8 h |
| 3 | Evaluation episodes from 100 to 50 per condition (keep 3 seeds) | compute time |
| 4 | Scenario grid from 8 conditions to 4 corner cases (calm-warm-light, windy-warm-light, windy-cold-light, windy-cold-heavy) | compute and writing time |
| 5 | Replace the forward-flight energy model with hover plus drag (disclose) | about 8 h |
| 6 | Switch to Plan B (point-mass environment) if not already | PyBullet risk |
| 7 | Keep only the lookahead ablation from E3 | about 4 h |

**Never cut:** fair baselines (B1, B2), the pre-stated statistics, the battery validation record, the limitations section, reproducibility, the reward-hacking checks.

### 6.6 Compute plan

Measure physics steps per second on 9 October and write the number into `docs/decisions.md`. One policy step equals about 24 physics steps (240 Hz physics, 10 Hz policy). Training time per run is total policy steps times 24, divided by (steps per second per core times cores used); evaluation cost is episodes times physics steps per episode divided by the same rate. Target: each seed in 8 hours or less of wall-clock time on one machine. If the measured rate is too low: reduce to 1 to 2 million policy steps, shorten missions, lower the physics rate to a divisor the library accepts, or move to Plan B. Run training overnight with laptops plugged in, save checkpoints, and keep CPU thermal throttling in mind. Free cloud notebooks (Colab, Kaggle) can serve as emergency CPU capacity for evaluation runs, but session limits make them unsuitable as the main compute.

---

## 7. Resources (all free; credibility notes included)

### 7.1 Software

| Tool | Used for | Licence (check the repository before redistributing) |
|---|---|---|
| Python 3.10 or newer; NumPy, SciPy, pandas, Matplotlib | Everything | Open source |
| PyBullet | Physics | zlib-style open licence |
| gym-pybullet-drones (v2.2.0) | Drone environments, PID | MIT |
| Gymnasium | Environment API | MIT |
| Stable-Baselines3 and PyTorch | PPO | MIT and BSD-style |
| PyBaMM | Electrochemical reference model | BSD-3-Clause |
| xarray, netCDF4, cdsapi | Reading and downloading ERA5 | Open source |
| pytest | Unit tests | MIT |
| TensorBoard | Learning curves | Apache-2.0 |
| Git and GitHub (free) | Version control, issues, project board, CI (free for public repositories) | n/a |
| Zotero (free) | Reference manager and BibTeX export | Open source |
| LaTeX (TeX Live) and Overleaf free plan | Writing. Free Overleaf allows the owner plus one collaborator, which is enough for two people; its 10-second compile limit can be a problem with many figures, so include pre-rendered PDF figures. Fallback: local TeX Live with Git | Open source / free tier |
| VS Code, draw.io (diagrams) | Editing, figures | Free |

### 7.2 Data and standards

| Resource | What for | Access and notes |
|---|---|---|
| **ERA5 hourly single levels** (Copernicus Climate Data Store) | Wind (10 m and 100 m), 2 m temperature, pressure, gust; 0.25 degrees (about 28 to 31 km), hourly, 1940 to present | Free CDS account and API key; licence "License to use Copernicus Products" (include the Copernicus acknowledgement text required by the licence page); queued requests, so submit on day one |
| **Open-Meteo historical API** | Quick no-signup access to reanalysis data | Free for non-commercial use up to 10,000 calls per day; data under CC BY 4.0, attribution required ("Weather data by Open-Meteo.com"); re-check the terms page on the day you use it |
| **CMU DJI Matrice 100 in-flight energy dataset** (Rodrigues et al., Scientific Data 2021) | Validate energy-model trends against payload, speed, altitude and wind; 209 flights, about 10.75 hours | Open; DOI 10.1038/s41597-021-00930-x; arXiv 2103.13313 |
| **McMaster University battery datasets** (Kollmeyer et al.) | Measured cold behaviour at 40, 25, 10, 0, -10, -20 degrees C: *Turnigy Graphene 5000 mAh* (DOI 10.17632/4fx8cjprxm.1), *Samsung INR21700 30T 3 Ah* (DOI 10.17632/9xyvy2njj3.2), *LG 18650HG2 3 Ah* (DOI 10.17632/b5mj79w5w9.2) | Free on Mendeley Data; the authors ask that the data be referenced; check the licence on each page |
| LG M50 cell datasheet | Capacity, continuous-discharge rating, mass, temperature limits | Manufacturer's website |
| Other open battery data: Battery Archive, CALCE (University of Maryland), NASA Prognostics Data Repository | Optional extra cross-checks | Recalled from memory: verify availability and licences before relying on them |
| US Standard Atmosphere 1976 | Air density as a function of altitude | Free from NASA's technical reports server |
| MIL-HDBK-1797 (flying qualities, Dryden turbulence) | Gust scale lengths and intensities | Free public standard |

### 7.3 Papers and sources to cite (status as of 7 Oct 2026)

"Verified" means I checked the title, authors and identifier online today. Everything else is recalled from memory and **must be checked** (Appendix B) before it enters the bibliography.

| Source | Status | What we take from it |
|---|---|---|
| Rodrigues, Patrikar, Choudhry, Feldgoise, Arcot, Gahlaut, Lau, Moon, Wagner, Matthews, Scherer, Samaras. In-flight positional and energy use data set of a DJI Matrice 100 quadcopter for small package delivery. *Scientific Data* 8 (2021). DOI 10.1038/s41597-021-00930-x | Verified | Real delivery-drone energy data for validation and motivation |
| Zeng, Xu, Zhang. Energy minimization for wireless communication with rotary-wing UAV. *IEEE Trans. Wireless Communications* (2019); arXiv 1804.02238 | Verified (title, authors, content); confirm volume and pages | Rotary-wing propulsion power model: blade profile, induced and parasite power |
| Gao, Zeng, et al. Energy model for UAV communications: experimental validation and model generalization. arXiv 2005.01305 | Verified that the arXiv entry exists; confirm exact title and venue | Experimental support for the Zeng model |
| Abeywickrama, Jayawickrama, He, Dutkiewicz. Comprehensive energy consumption model for unmanned aerial vehicles, based on empirical studies of battery performance. *IEEE Access* 6:58383-58394 (2018). DOI 10.1109/ACCESS.2018.2875040 | Verified | Empirical UAV energy and battery behaviour; related work |
| Thibbotuwawa, Nielsen, Zbigniew, Bocewicz. Energy consumption in unmanned aerial vehicles: a review of energy consumption models and their relation to the UAV routing. *Advances in Intelligent Systems and Computing* 853:173-184 (2019). DOI 10.1007/978-3-319-99996-8_16 | Verified | Review for related work |
| McMaster datasets (Kollmeyer et al.), DOIs above | Verified | Measured cold battery behaviour |
| Sulzer et al. Python Battery Mathematical Modelling (PyBaMM). *J. Open Research Software* 9(1) (2021) | From memory | Cite the simulator |
| Chen, Brosa Planella, O'Regan, Gastol, Widanage, Kendrick. Development of experimental techniques for parameterization of multi-scale lithium-ion battery models. *J. Electrochem. Soc.* 167 (2020) | From memory | Source of the LG M50 parameter set |
| Marquis, Sulzer, Timms, Please, Chapman. An asymptotic derivation of a single particle model with electrolyte. *J. Electrochem. Soc.* 166 (2019) | From memory | The SPMe model |
| Hu, Li, Peng. A comparative study of equivalent circuit models for Li-ion batteries. *J. Power Sources* 198 (2012) | From memory | Justify the 1-RC circuit structure |
| Bernardi, Pawlikowski, Newman. A general energy balance for battery systems. *J. Electrochem. Soc.* 132 (1985) | From memory | Heat generation (irreversible and reversible parts) |
| Panerati, Zheng, Zhou, Xu, Prorok, Schoellig. Learning to fly: a Gym environment with PyBullet physics for reinforcement learning of multi-agent quadcopter control. IROS 2021 | From memory | Cite the simulator |
| Schulman et al. Proximal policy optimization algorithms. arXiv 1707.06347 (2017) | From memory | PPO |
| Raffin et al. Stable-Baselines3: reliable reinforcement learning implementations. *JMLR* 22 (2021) | From memory | Implementation |
| Hersbach et al. The ERA5 global reanalysis. *Quarterly J. Royal Meteorological Society* 146 (2020) | From memory | Cite ERA5 (plus the CDS dataset citation) |
| Stolaroff et al. Energy use and life cycle greenhouse gas emissions of drones for commercial package delivery. *Nature Communications* 9 (2018) | From memory | Delivery-drone energy context; energy per payload-distance |
| US Standard Atmosphere 1976; MIL-HDBK-1797 | From memory | Atmosphere and gust model |
| The NASA wind-field reinforcement-learning energy-path paper | **To be found in S6** (NASA Technical Reports Server) | Closest prior work; the novelty gap rests on it |

### 7.4 "Cite instead of build" quick reference

| We need | We do not build | We cite and adopt |
|---|---|---|
| Electrochemical battery truth | A cell model from scratch | PyBaMM SPMe with Chen2020 |
| Fast battery for RL | A new circuit theory | 1-RC equivalent circuit (Hu et al.) fitted to PyBaMM |
| Pack heating | New thermal theory | Energy balance of Bernardi et al., lumped node |
| Rotor power | New aerodynamics | Zeng et al. propulsion model |
| Air density | An atmosphere model | International Standard Atmosphere |
| Gusts | A turbulence solver | Dryden-style Gauss-Markov process (MIL-HDBK-1797 parameters) |
| Weather realism | Weather generator from nothing | ERA5 statistics |
| Real-flight sanity check | Our own flight tests | CMU Matrice 100 dataset |
| Drone physics and PID | A simulator | gym-pybullet-drones |
| RL algorithm | PPO from scratch | Stable-Baselines3 |

### 7.5 Not used

Paid tools (MATLAB or Simulink, commercial battery software, paid datasets, cloud GPUs), data without a clear licence, and code copied without a licence.

### 7.6 Citation hygiene

PyBaMM includes a built-in way to print the list of papers to cite for the models used in a run; use it and add the result to the bibliography. Keep everything in Zotero from day one, verify each DOI resolves, cite software versions (Python, PyBullet, gym-pybullet-drones 2.2.0, Stable-Baselines3, PyBaMM), and give each dataset's DOI and access date.

---

## 8. Paper plan

| Section | Content | Lead | Source material |
|---|---|---|---|
| Abstract | Problem, method, headline numbers with intervals, limitation in one sentence | A (draft), B (edit) | Written last |
| 1. Introduction | Delivery drones and energy; gap (from S6); contributions | B (draft), A (edit) | S6 gap table |
| 2. Related work | Energy models, battery-aware planning, RL for UAVs, wind and weather, datasets | B | S6; Section 7.3 |
| 3. Methods | 3.1 Mission; 3.2 Vehicle and controller; 3.3 Energy model; 3.4 Battery (electrochemical, circuit, thermal) with validation; 3.5 Weather, wind, gusts, temperature, density; 3.6 RL formulation; 3.7 Baselines | A: 3.1 to 3.4; B: 3.5 to 3.7 | Model cards; this file |
| 4. Experiments | Scenarios, metrics, statistics, E1 to E6 | B (A reviews) | Section 1.7 |
| 5. Results | E1 to E4 (E5, E6 if done) with tables and figures | B (E2 to E6), A (E0, E1) | `results/summary/` |
| 6. Discussion | What the agent does differently; whether savings are speed or efficiency; battery effects | Both | Behaviour diagnostics |
| 7. Limitations and future work | Battery limits (Section 1.5), assumed parameters, simulation-only, weather scale, excluded factors (rain, motor heating, aging, risk-aware planning, two-branch battery, hardware) | A (battery, physics), B (data, RL) | Appendix A |
| 8. Reproducibility, data and code availability, AI-use disclosure | Commit hash, configs, licence | Both | S5 |

**Figures.** F1 system architecture; F2 mission and scenario map; F3 battery lookup heatmaps; F4 battery against PyBaMM (runtime and voltage); F5 self-heating illustration; F6 energy model against M100 trends; F7 learning curves (all seeds); F8 energy per condition with intervals; F9 trajectories; F10 wind map with paths; F11 ablations. **Tables.** T1 factors and models; T2 parameters and sources (assumptions flagged); T3 main results with intervals; T4 statistical tests; T5 verification record.

**Venue.** Check the target venue's page limit, template and anonymity rules today (7 October), so the figure budget and writing format are fixed early.

---

## 9. Risks and contingencies

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| First PyBullet run fails (loading, controller, library assumption) | Medium | High | G1 on 12 Oct; offline checks narrow the cause; Plan B ready | A |
| Training too slow | Medium | High | Measure throughput on 9 Oct; mixture training; budget cuts (6.6, 6.5) | A and B |
| Agent only "wins" by being faster | Medium | High | Fairness pair of metrics; time-optimal baseline B2; energy-dominant reward; reward-hacking checks | B |
| Reward hacking at factor boundaries | Medium | Medium | Checks before trusting any result | Both |
| ERA5 request delayed | Medium | Low | Open-Meteo fallback; submit on day one | B |
| ERA5 resolution too coarse for zones | Certain | Medium | Statistical grounding with disclosed synthetic realisations (1.6) | B |
| Battery effects too small in short missions | Medium | Medium | Randomised starting SoC; mission length 2 to 4 minutes | A |
| Assumed parameters challenged | High | Medium | Registry (Appendix A), sweeps, honest wording | Both |
| Merge conflicts or lost work | Low | High | Small branches, push daily, CI | Both |
| Time shortage or illness | Medium | High | 20% slack; scope-cut ladder (6.5); shared back-up tasks (5.5) | Both |
| Novelty claim challenged | Medium | Medium | Complete S6 before writing the Introduction; cautious wording | Both |
| Unverified references | Medium | High | Appendix B check on 28 Oct (earlier for key ones) | Both |

---

## 10. Working agreements

- **Daily 15-minute check-in** at a fixed time. Three questions each: What did I finish? What am I doing next? What blocks me? Update the status table at the end of the day.
- **Blocked for more than 2 hours: say so** (do not wait until the next check-in).
- **Definition of done for a task:** code merged to `main`, unit test passing, one paragraph in the module's model card, an entry in `docs/decisions.md` if a design decision was made.
- **Pull requests:** small, reviewed by the other member within 24 hours; the other member runs the tests.
- **Push every evening** (backup).
- **After 24 October only fixes:** no new features, no reward changes, no new conditions.
- **Disagreements** on a design choice: write both options in `docs/decisions.md`, pick the cheaper one that passes the credibility test, move on.

---

## Appendix A. Assumed-values registry (copy into `docs/assumptions.md` and keep updated)

| Quantity | Value | Basis | How to confirm or test |
|---|---|---|---|
| Thermal heat capacity | 390 J/K | 6 cells, about 69 g each, about 950 J/(kg K) | Datasheet mass; bench log with `fit_thermal_params.py` |
| Heat-loss coefficient | 0.75 W/K (sweep 0.3 to 1.5) | Plausible range | Bench cool-down; otherwise sweep |
| Propeller thrust coefficient | 0.10 | Typical 12-inch propeller | Manufacturer thrust tables |
| Rotor figure of merit; motor and ESC efficiency | 0.65; 0.80 | Typical | Thrust-stand data |
| Avionics power | 5 W | Estimate | Datasheet |
| Wind drag area | 0.07 square metres | Estimate | Fly in measured wind, record tilt; sweep |
| Inertia and arm length | rough mass breakdown, 0.275 m | Design choice | CAD |
| Rotor constants for the Zeng model (solidity, blade profile power, fuselage drag ratio) | taken from the paper's example, scaled | Not known for HL1 | Sweep; report sensitivity |
| Zone correlation length, gust scale length | set on 14 Oct | Dryden-style defaults | Sweep |
| Mission ranges (leg lengths, payload mass, start SoC, start temperature) | set on 14 Oct | Design choice | Document in `configs/` |
| Cell mass, cutoff voltage, continuous-discharge rating | about 69 g, 3.0 V per cell, to be read | Datasheet recall | Check the LG M50 datasheet |
| Humidity effect on air density | ignored | Small (about 1 to 2%) | State in the paper |

## Appendix B. Reference verification checklist (28 October, earlier for key sources)

For every entry in the bibliography: (1) open the source and read at least its abstract and the part we rely on; (2) confirm title, authors, year, venue, volume and pages against the publisher or DOI page; (3) confirm the DOI resolves; (4) confirm that what we say it contains is actually in it; (5) check the licence of any data or code taken from it. Sources marked "From memory" in 7.3 get the full treatment; sources marked "Verified" still need volume and page confirmation.

## Appendix C. Glossary

**SoC** state of charge. **OCV** open-circuit voltage. **R0, R1, C1** the series resistance and the resistor-capacitor branch of the equivalent circuit. **ECM** equivalent-circuit model. **SPMe** single-particle model with electrolyte. **PyBaMM** open-source battery simulator. **ERA5** global weather reanalysis from ECMWF. **PPO** proximal policy optimisation. **PID** proportional-integral-derivative controller. **C-rate** current divided by capacity (1C = 5 A for this pack). **HPPC** hybrid pulse power characterisation. **Brownout** the pack cannot deliver the requested power. **META** most effective tactic available (this project's shorthand). **Plan B** the point-mass fallback environment.

---

*End of document. Update the log below when the plan changes.*

| Date | Change | By |
|---|---|---|
| 7 Oct 2026 | Version 1.0 created | |
