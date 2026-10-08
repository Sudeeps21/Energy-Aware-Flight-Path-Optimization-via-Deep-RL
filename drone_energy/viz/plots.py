"""
plot_results.py — Generate all paper-quality figures from evaluation results.

Figures produced
----------------
  Fig 1 — Energy comparison (PID vs PPO across scenarios)  [bar chart]
  Fig 2 — Mission success rate comparison                   [bar chart]
  Fig 3 — Battery capacity vs temperature                   [line chart]
  Fig 4 — Hover power vs altitude                           [line chart]
  Fig 5 — Battery capacity factor vs temperature (model validation)
  Fig 6 — Training curve (reward over timesteps)            [line chart]
  Fig 7 — Energy efficiency gain (PPO vs PID) per scenario  [bar chart]
  Fig 8 — Payload effect on hover power                     [grouped bar]

Usage
-----
  python plot_results.py --results_dir results/

All figures are saved to results/figures/ as high-resolution PDFs and PNGs
suitable for direct inclusion in LaTeX.
"""

import os
import sys
import json
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from drone_energy.physics.energy import (
    DroneParams,
    air_density,
    battery_capacity_factor,
    hover_power,
)

# ── Aesthetics ──────────────────────────────────────────────────────────────

PALETTE = {
    "pid"  : "#E07A5F",   # terracotta
    "ppo"  : "#3D405B",   # dark navy
    "calm" : "#81B29A",   # sage green
    "windy": "#F2CC8F",   # warm yellow
    "cold" : "#7AAFBD",   # arctic blue
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "figure.dpi"      : 150,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "grid.linestyle"  : "--",
})

SCENARIOS = ["calm", "windy", "cold"]
SCENARIO_LABELS = {"calm": "Calm", "windy": "Windy", "cold": "Cold"}


def load_results(results_dir: str) -> dict:
    path = os.path.join(results_dir, "evaluation_results.json")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Generating model-only figures.")
        return {}
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Energy Comparison ─────────────────────────────────────────────

def plot_energy_comparison(summary: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    x       = np.arange(len(SCENARIOS))
    width   = 0.35

    pid_energy = [
        summary.get(sc, {}).get("pid", {}).get("total_energy_wh_mean", 0)
        for sc in SCENARIOS
    ]
    pid_std    = [
        summary.get(sc, {}).get("pid", {}).get("total_energy_wh_std", 0)
        for sc in SCENARIOS
    ]
    ppo_energy = [
        summary.get(sc, {}).get("ppo", {}) or {}
        for sc in SCENARIOS
    ]
    ppo_vals = [p.get("total_energy_wh_mean", None) for p in ppo_energy]
    ppo_std  = [p.get("total_energy_wh_std", 0) for p in ppo_energy]

    bars1 = ax.bar(x - width/2, pid_energy, width, label="PID Baseline",
                   color=PALETTE["pid"], alpha=0.85, yerr=pid_std,
                   error_kw=dict(elinewidth=1.2, capsize=4))

    # Only plot PPO if data is available
    valid_ppo = [v is not None for v in ppo_vals]
    if any(valid_ppo):
        ppo_plot = [v if v is not None else 0 for v in ppo_vals]
        ppo_std_plot = [s if v is not None else 0 for v, s in zip(ppo_vals, ppo_std)]
        bars2 = ax.bar(x + width/2, ppo_plot, width, label="PPO Agent",
                       color=PALETTE["ppo"], alpha=0.85, yerr=ppo_std_plot,
                       error_kw=dict(elinewidth=1.2, capsize=4))

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Energy Consumed (Wh)")
    ax.set_title("Fig 1 — Mission Energy: PID Baseline vs. PPO Agent")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig1_energy_comparison")
    print("  Saved: fig1_energy_comparison")


# ── Figure 2: Mission Success Rate ───────────────────────────────────────────

def plot_success_rate(summary: dict, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x, width = np.arange(len(SCENARIOS)), 0.35

    pid_sr = [
        summary.get(sc, {}).get("pid", {}).get("mission_complete_mean", 0) * 100
        for sc in SCENARIOS
    ]
    ppo_data = [summary.get(sc, {}).get("ppo") for sc in SCENARIOS]
    ppo_sr   = [
        d.get("mission_complete_mean", 0) * 100 if d else None
        for d in ppo_data
    ]

    ax.bar(x - width/2, pid_sr, width, label="PID Baseline",
           color=PALETTE["pid"], alpha=0.85)

    if any(v is not None for v in ppo_sr):
        ax.bar(x + width/2, [v or 0 for v in ppo_sr], width, label="PPO Agent",
               color=PALETTE["ppo"], alpha=0.85)

    ax.set_ylim(0, 115)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Mission Success Rate (%)")
    ax.set_title("Fig 2 — Mission Success Rate: PID vs. PPO")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS])
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig2_success_rate")
    print("  Saved: fig2_success_rate")


# ── Figure 3: Battery Capacity vs Temperature (model validation) ─────────────

def plot_battery_temperature(out_dir: str):
    temps  = np.linspace(-20, 40, 200)
    factors = [battery_capacity_factor(t) for t in temps]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temps, [f * 100 for f in factors], color=PALETTE["ppo"], linewidth=2.5)

    # Reference points
    ref_points = [(-10, battery_capacity_factor(-10) * 100),
                  (0,   battery_capacity_factor(0)   * 100),
                  (25,  100.0)]
    for t, c in ref_points:
        ax.scatter([t], [c], color=PALETTE["pid"], zorder=5, s=60)
        ax.annotate(f"{c:.0f}% @ {t}°C", (t, c),
                    textcoords="offset points", xytext=(8, -12), fontsize=9)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(25,  color="gray", linestyle=":",  linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Ambient Temperature (°C)")
    ax.set_ylabel("Available Battery Capacity (%)")
    ax.set_title("Fig 3 — Temperature Derating: Li-Po Battery Capacity Model")
    ax.set_xlim(-22, 42)
    ax.set_ylim(20, 110)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_battery_temperature")
    print("  Saved: fig3_battery_temperature")


# ── Figure 4: Hover Power vs Altitude ────────────────────────────────────────

def plot_hover_power_altitude(out_dir: str):
    drone      = DroneParams()
    altitudes  = np.linspace(0, 3000, 300)   # 0 – 3000 m ASL

    # Three payload conditions
    conditions = [
        (0.0,   "No payload",    PALETTE["calm"]),
        (0.3,   "0.3 kg payload", PALETTE["windy"]),
        (0.6,   "0.6 kg payload", PALETTE["cold"]),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    for payload, label, color in conditions:
        powers = [hover_power(drone, alt, 25.0, payload) for alt in altitudes]
        ax.plot(altitudes, powers, label=label, color=color, linewidth=2)

    ax.set_xlabel("Altitude (m above sea level)")
    ax.set_ylabel("Hover Power (W)")
    ax.set_title("Fig 4 — Altitude Effect on Hover Power (ISA atmosphere)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig4_hover_power_altitude")
    print("  Saved: fig4_hover_power_altitude")


# ── Figure 5: Air Density vs Altitude ────────────────────────────────────────

def plot_air_density(out_dir: str):
    altitudes = np.linspace(0, 5000, 300)
    densities = [air_density(a) for a in altitudes]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(altitudes, densities, color=PALETTE["ppo"], linewidth=2.5)
    ax.fill_between(altitudes, densities, alpha=0.15, color=PALETTE["ppo"])
    ax.axhline(1.225, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Sea-level (1.225 kg/m³)")
    ax.set_xlabel("Altitude (m above sea level)")
    ax.set_ylabel("Air Density (kg/m³)")
    ax.set_title("Fig 5 — ISA Air Density vs. Altitude")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig5_air_density")
    print("  Saved: fig5_air_density")


# ── Figure 6: Payload Effect on Hover Power ───────────────────────────────────

def plot_payload_effect(out_dir: str):
    drone    = DroneParams()
    payloads = np.linspace(0, 1.0, 50)  # 0 – 1 kg payload

    # Two altitudes
    fig, ax = plt.subplots(figsize=(7, 4))
    for alt, label, color in [(0, "Sea level (0 m)", PALETTE["calm"]),
                               (500, "500 m ASL",     PALETTE["cold"])]:
        powers = [hover_power(drone, alt, 25.0, p) for p in payloads]
        ax.plot(payloads, powers, label=label, color=color, linewidth=2)

    # Mark the pickup event
    ax.axvline(0.3, color=PALETTE["pid"], linestyle="--", linewidth=1.2,
               label="Pickup (0.3 kg)")
    ax.axvline(0.0, color=PALETTE["pid"], linestyle=":",  linewidth=1.2,
               label="Delivery (drop)")

    ax.set_xlabel("Payload Mass (kg)")
    ax.set_ylabel("Hover Power (W)")
    ax.set_title("Fig 6 — Dynamic Payload: Effect on Hover Power")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig6_payload_effect")
    print("  Saved: fig6_payload_effect")


# ── Figure 7: Combined scenario summary (4-panel) ─────────────────────────────

def plot_combined_summary(summary: dict, out_dir: str):
    """4-panel overview figure for the paper's results section."""
    fig = plt.figure(figsize=(12, 8))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    x      = np.arange(len(SCENARIOS))
    labels = [SCENARIO_LABELS[s] for s in SCENARIOS]
    width  = 0.35

    def _extract(metric, agent_type="pid"):
        vals = [
            summary.get(sc, {}).get(agent_type, {}).get(metric, 0) or 0
            for sc in SCENARIOS
        ]
        stds = [
            summary.get(sc, {}).get(agent_type, {}).get(metric.replace("mean", "std"), 0) or 0
            for sc in SCENARIOS
        ]
        return vals, stds

    # ── Panel A: Energy ──────────────────────────────────────────────────
    ax = axes[0]
    pid_e, pid_e_s = _extract("total_energy_wh_mean", "pid")
    ppo_e, ppo_e_s = _extract("total_energy_wh_mean", "ppo")
    ax.bar(x - width/2, pid_e, width, label="PID", color=PALETTE["pid"], alpha=0.85)
    if any(v > 0 for v in ppo_e):
        ax.bar(x + width/2, ppo_e, width, label="PPO", color=PALETTE["ppo"], alpha=0.85)
    ax.set_title("(a) Energy Consumed (Wh)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    # ── Panel B: Success Rate ────────────────────────────────────────────
    ax = axes[1]
    pid_s, _ = _extract("mission_complete_mean", "pid")
    ppo_s, _ = _extract("mission_complete_mean", "ppo")
    ax.bar(x - width/2, [v*100 for v in pid_s], width, label="PID",
           color=PALETTE["pid"], alpha=0.85)
    if any(v > 0 for v in ppo_s):
        ax.bar(x + width/2, [v*100 for v in ppo_s], width, label="PPO",
               color=PALETTE["ppo"], alpha=0.85)
    ax.set_ylim(0, 115)
    ax.set_title("(b) Mission Success Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend(fontsize=8)

    # ── Panel C: Battery temp model ──────────────────────────────────────
    ax = axes[2]
    temps  = np.linspace(-20, 40, 200)
    caps   = [battery_capacity_factor(t) * 100 for t in temps]
    ax.plot(temps, caps, color=PALETTE["ppo"], linewidth=2)
    ax.set_title("(c) Battery Capacity vs Temperature")
    ax.set_xlabel("Temp (°C)")
    ax.set_ylabel("Capacity (%)")

    # ── Panel D: Payload/altitude hover power ────────────────────────────
    ax = axes[3]
    drone  = DroneParams()
    payloads = np.linspace(0, 1.0, 50)
    for alt, label, color in [(0, "0 m", PALETTE["calm"]), (500, "500 m", PALETTE["cold"])]:
        pows = [hover_power(drone, alt, 25.0, p) for p in payloads]
        ax.plot(payloads, pows, label=label, color=color, linewidth=2)
    ax.axvline(0.3, color=PALETTE["pid"], linestyle="--", linewidth=1.2, label="Pickup")
    ax.set_title("(d) Payload vs Hover Power (W)")
    ax.set_xlabel("Payload (kg)")
    ax.legend(fontsize=8)

    fig.suptitle("Multi-Factor Drone RL — Results Overview", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "fig7_combined_summary")
    print("  Saved: fig7_combined_summary")


# ── Training curve helper ──────────────────────────────────────────────────────

def plot_training_curve_from_monitor(results_dir: str, out_dir: str):
    """
    Plot training curves from SB3 monitor logs.
    Looks for eval_ppo_<scenario>_s42/evaluations.npz files.
    """
    found_any = False
    fig, ax   = plt.subplots(figsize=(9, 5))

    colors = {"calm": PALETTE["calm"], "windy": PALETTE["windy"], "cold": PALETTE["cold"]}

    for scenario in SCENARIOS:
        npz_path = os.path.join(
            results_dir, f"eval_ppo_{scenario}_s42", "evaluations.npz"
        )
        if not os.path.exists(npz_path):
            continue
        data    = np.load(npz_path)
        steps   = data["timesteps"]
        rewards = data["results"].mean(axis=1)
        std     = data["results"].std(axis=1)

        ax.plot(steps, rewards, label=SCENARIO_LABELS[scenario],
                color=colors[scenario], linewidth=2)
        ax.fill_between(steps, rewards - std, rewards + std,
                        color=colors[scenario], alpha=0.15)
        found_any = True

    if not found_any:
        print("  No training curves found — run train_ppo.py first.")
        plt.close(fig)
        return

    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Fig 8 — PPO Training Curves by Scenario")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "fig8_training_curves")
    print("  Saved: fig8_training_curves")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _save(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(results_dir: str = "results"):
    fig_dir = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n=== Generating paper figures → {fig_dir} ===\n")

    # Load evaluation results (may be empty if not yet generated)
    data    = load_results(results_dir)
    summary = data.get("summary_by_scenario", {})

    # Model-only figures (always possible)
    plot_battery_temperature(fig_dir)
    plot_hover_power_altitude(fig_dir)
    plot_air_density(fig_dir)
    plot_payload_effect(fig_dir)

    # Results-dependent figures
    plot_energy_comparison(summary, fig_dir)
    plot_success_rate(summary, fig_dir)
    plot_combined_summary(summary, fig_dir)

    # Training curves (if SB3 eval logs exist)
    plot_training_curve_from_monitor(results_dir, fig_dir)

    print(f"\n=== Done. All figures saved to {fig_dir}/ ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    main(args.results_dir)
