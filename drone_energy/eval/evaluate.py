"""
evaluate.py — Evaluate a trained PPO agent vs. the PID baseline across
all experimental conditions.

This script generates the structured results tables required for the paper:
  - Per-scenario metric tables (energy, mission success, waypoints)
  - Cross-condition comparison (RL vs. PID)

Usage
-----
  python evaluate.py --model_dir results/ --episodes 50

Results are saved to results/evaluation_results.json for downstream plotting.
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv
from drone_energy.baselines.pid_baseline import WaypointPIDAgent


SCENARIOS = ["calm", "windy", "cold"]


def run_episodes(
    agent,
    scenario: str,
    n_episodes: int = 50,
    seed_start: int = 1000,
    agent_type: str = "ppo",
) -> List[Dict[str, Any]]:
    """
    Roll out `agent` for `n_episodes` on `scenario` and collect per-episode stats.

    Returns
    -------
    list of dicts, each containing:
        episode, total_energy_wh, mission_complete, waypoints_reached, steps,
        final_soc, scenario, agent_type
    """
    records = []
    for ep in range(n_episodes):
        env  = MultiFactorDroneEnv(scenario=scenario, seed=seed_start + ep)
        obs, _ = env.reset(seed=seed_start + ep)

        if agent_type == "pid":
            agent.reset()

        done       = False
        total_r    = 0.0

        while not done:
            if agent_type == "ppo":
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action    = agent.act(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done     = terminated or truncated

        stats = env.episode_stats if env.episode_stats else {
            "total_energy_wh"  : info.get("total_energy_wh", 0.0),
            "mission_complete" : info.get("mission_complete", False),
            "waypoints_reached": info.get("waypoint_idx", 0),
            "steps"            : info.get("steps", env._step_count),
            "final_soc"        : info.get("battery_soc", 0.0),
        }

        records.append({
            "episode"          : ep,
            "scenario"         : scenario,
            "agent_type"       : agent_type,
            "total_reward"     : float(total_r),
            **{k: (float(v) if not isinstance(v, bool) else v)
               for k, v in stats.items()},
        })
        env.close()

    return records


def summarise(records: List[Dict]) -> Dict:
    """Compute mean +/- std for key metrics over a list of episode records."""
    metrics = ["total_energy_wh", "mission_complete", "waypoints_reached",
               "steps", "final_soc", "total_reward"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in records if m in r]
        if vals:
            arr = np.array(vals, dtype=float)
            summary[m + "_mean"] = float(arr.mean())
            summary[m + "_std"]  = float(arr.std())
    return summary


def evaluate(
    model_dir: str = "results",
    n_episodes: int = 50,
    seed_start: int = 1000,
):
    """Main evaluation routine."""

    all_records = []
    results_by_scenario = {}

    pid_agent = WaypointPIDAgent()

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  Evaluating scenario: {scenario.upper()}")
        print(f"{'='*60}")

        # ── PID baseline ──────────────────────────────────────────────
        print("  Running PID baseline...")
        pid_records = run_episodes(
            pid_agent, scenario, n_episodes, seed_start, agent_type="pid"
        )
        all_records.extend(pid_records)

        pid_summary = summarise(pid_records)
        print(f"  PID  | energy={pid_summary.get('total_energy_wh_mean', 0):.4f} Wh "
              f"| success={pid_summary.get('mission_complete_mean', 0)*100:.1f}% "
              f"| reward={pid_summary.get('total_reward_mean', 0):.1f}")

        # ── PPO agent ─────────────────────────────────────────────────
        model_path = os.path.join(model_dir, f"ppo_{scenario}_s42.zip")
        ppo_summary = None

        if SB3_AVAILABLE and os.path.exists(model_path):
            print(f"  Loading PPO model: {model_path}")
            ppo_agent = PPO.load(model_path)
            ppo_records = run_episodes(
                ppo_agent, scenario, n_episodes, seed_start, agent_type="ppo"
            )
            all_records.extend(ppo_records)
            ppo_summary = summarise(ppo_records)
            print(f"  PPO  | energy={ppo_summary.get('total_energy_wh_mean', 0):.4f} Wh "
                  f"| success={ppo_summary.get('mission_complete_mean', 0)*100:.1f}% "
                  f"| reward={ppo_summary.get('total_reward_mean', 0):.1f}")
        else:
            if not SB3_AVAILABLE:
                print("  PPO model skipped — stable-baselines3 not installed.")
            else:
                print(f"  PPO model not found at {model_path}. Train first with train_ppo.py")

        results_by_scenario[scenario] = {
            "pid"  : pid_summary,
            "ppo"  : ppo_summary,
        }

    # ── Save results ──────────────────────────────────────────────────────
    out = {
        "summary_by_scenario": results_by_scenario,
        "all_records"        : all_records,
        "n_episodes"         : n_episodes,
    }

    out_path = os.path.join(model_dir, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved → {out_path}")
    _print_table(results_by_scenario)
    return out


def _print_table(results_by_scenario: Dict):
    """Print a compact ASCII comparison table."""
    print("\n" + "="*70)
    print(f"{'SCENARIO':<10} {'AGENT':<6} {'ENERGY(Wh)':<14} {'SUCCESS%':<12} {'AVG REWARD':<12}")
    print("="*70)
    for sc, agents in results_by_scenario.items():
        for agent_type, s in agents.items():
            if s is None:
                continue
            energy  = s.get("total_energy_wh_mean", 0)
            success = s.get("mission_complete_mean", 0) * 100
            reward  = s.get("total_reward_mean", 0)
            print(f"{sc:<10} {agent_type.upper():<6} {energy:<14.4f} {success:<12.1f} {reward:<12.1f}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  default="results")
    parser.add_argument("--episodes",   default=50, type=int)
    parser.add_argument("--seed_start", default=1000, type=int)
    args = parser.parse_args()

    evaluate(
        model_dir=args.model_dir,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
    )
