# Multi-Factor Drone RL — Research Project

**Paper topic**: Multi-factor energy-aware drone delivery with PPO — combining wind zones, temperature-aware battery, and dynamic payload in a single RL environment.

## Project Structure

```
src/
├── run_all.py                    ← Master pipeline (train + eval + plot)
├── requirements.txt
├── utils/
│   ├── energy_model.py           ← Thrust-to-power, battery, air density
│   └── wind_zones.py             ← Spatial wind/temperature grid
├── envs/
│   └── multi_factor_drone_env.py ← Gymnasium environment
├── models/
│   └── pid_baseline.py           ← PID waypoint-following baseline
├── experiments/
│   ├── sanity_check.py           ← Run this FIRST to verify models
│   ├── train_ppo.py              ← PPO training
│   ├── evaluate.py               ← PID vs PPO metrics
│   └── plot_results.py           ← All paper figures
└── results/                      ← Generated models, logs, figures
```

## Quick Start

### 1. Set up virtual environment

```bash
cd /home/sudeee/research
python3 -m venv venv
source venv/bin/activate
pip install -r src/requirements.txt
```

### 2. Run sanity checks first (always do this)

```bash
cd src
python experiments/sanity_check.py
```

You should see all `[PASS]` — this verifies your physics models are correct before touching RL.

### 3. Full pipeline (train + evaluate + plot)

```bash
python run_all.py --timesteps 500000
```

### 4. Quick test run (shorter training)

```bash
python run_all.py --timesteps 100000 --n_envs 2
```

### 5. Skip training (use existing models)

```bash
python run_all.py --skip_train
```

---

## Training individual scenarios

```bash
python experiments/train_ppo.py --scenario calm  --timesteps 500000 --seed 42
python experiments/train_ppo.py --scenario windy --timesteps 500000 --seed 42
python experiments/train_ppo.py --scenario cold  --timesteps 500000 --seed 42
```

## Evaluating agents

```bash
python experiments/evaluate.py --model_dir results/ --episodes 50
```

## Generating figures only

```bash
python experiments/plot_results.py --results_dir results/
```

---

## Environment Details

**Observation space**: 22-dimensional vector
| Indices | Content |
|---------|---------|
| [0:3]   | Position (normalised) |
| [3:6]   | Velocity (normalised) |
| [6:9]   | Current zone wind vector (m/s) |
| [9]     | Zone temperature (normalised) |
| [10]    | Battery state-of-charge [0,1] |
| [11]    | Payload carried (0 or 1) |
| [12:15] | Direction to next waypoint |
| [15]    | Distance to next waypoint |
| [16]    | Altitude (normalised) |
| [17:20] | Previous action |
| [20]    | Time remaining |
| [21]    | Step fraction |

**Action space**: 3-D continuous acceleration [-1, 1] per axis

**Mission**: Start → Pickup (mass added) → Delivery (mass dropped) → Home

---

## Paper Figures Generated

| Figure | Content | Source data |
|--------|---------|-------------|
| Fig 1  | Energy comparison PID vs PPO across scenarios | evaluation_results.json |
| Fig 2  | Mission success rate comparison | evaluation_results.json |
| Fig 3  | Battery capacity vs temperature (model validation) | energy_model.py |
| Fig 4  | Hover power vs altitude | energy_model.py |
| Fig 5  | ISA air density vs altitude | energy_model.py |
| Fig 6  | Payload effect on hover power | energy_model.py |
| Fig 7  | 4-panel combined summary | Both |
| Fig 8  | PPO training curves by scenario | SB3 eval logs |

All figures saved as both `.pdf` (for LaTeX) and `.png` (for quick preview) in `results/figures/`.

---

## Key Research Values to Report

### Model Validation (from sanity_check.py output)
- Hover power at 0 m, 500 m, 1000 m, 2000 m, 3000 m
- Battery capacity at -20°C, -10°C, 0°C, 10°C, 25°C
- Air density at key altitudes (compare with ISA tables)

### Comparison Metrics (from evaluate.py)
- Mean energy consumed (Wh) — PID vs PPO — per scenario
- Mission success rate (%) — PID vs PPO — per scenario  
- Mean episode reward — PID vs PPO — per scenario
- Battery SoC at end of episode
- Steps to completion

### Statistical Significance
- Run with multiple seeds (e.g., `--seed 42`, `43`, `44`) and report mean ± std

---

## Extending the Environment

### Add a new scenario
Edit `utils/wind_zones.py` — add a new `load_<name>()` method on `WindZoneMap`
and register it in `make_scenario()`.

### Change drone parameters
```python
from utils.energy_model import DroneParams
drone = DroneParams(mass=1.5, battery_capacity_wh=75.0)
env = MultiFactorDroneEnv(drone_params=drone)
```

### Use real ERA5 data
Replace `load_calm()` / `load_windy()` / `load_cold()` values in `wind_zones.py`
with values extracted from ERA5 reanalysis (NOAA or Copernicus CDS API).
