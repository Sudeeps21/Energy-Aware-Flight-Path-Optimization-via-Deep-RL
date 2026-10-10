"""
evaluate.py — Evaluate a trained PPO agent vs. the PID and A* baselines across
all experimental conditions, with full statistical tests required by the paper.

Statistical protocol (from TEAM_PLAN §1.7):
  - 100 evaluation episodes per condition per seed (same seeds across all methods)
  - Mean ± std + 95% bootstrap confidence intervals on energy savings
  - Welch's t-test (primary) + Mann-Whitney U (non-parametric check)
  - Cohen's d effect size
  - FAIRNESS CHECK: Energy per km (Wh/km) and mean speed reported alongside total
    energy so that savings from simply flying faster are visible.

Usage
-----
  python evaluate.py --model_dir results/ --episodes 100
"""

import argparse
import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as scipy_stats

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv
from drone_energy.baselines.pid_baseline import WaypointPIDAgent
from drone_energy.baselines.astar_cost_map import AStarEnergyAgent


SCENARIOS = ["calm", "windy", "cold"]

# Metrics collected per episode (must exist in episode_stats after env fix)
_METRIC_KEYS = [
    "total_energy_wh",
    "mission_complete",
    "waypoints_reached",
    "steps",
    "final_soc",
    "path_length_m",
    "energy_per_km",
    "mean_speed_ms",
    "payload_mass_kg",
    "start_soc",
]


# ─── Episode rollout ──────────────────────────────────────────────────────────

def run_episodes(
    agent,
    scenario: str,
    n_episodes: int = 100,
    seed_start: int = 1000,
    agent_type: str = "ppo",
) -> List[Dict[str, Any]]:
    """
    Roll out `agent` for `n_episodes` on `scenario` and collect per-episode stats.
    The same seed sequence is used for every agent so comparisons are paired.
    """
    records = []
    for ep in range(n_episodes):
        ep_seed = seed_start + ep
        env = MultiFactorDroneEnv(scenario=scenario, seed=ep_seed)
        obs, _ = env.reset(seed=ep_seed)

        if agent_type in ["pid", "astar"]:
            agent.reset()

        done    = False
        total_r = 0.0

        while not done:
            if agent_type.startswith("ppo"):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                action = agent.act(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward
            done = terminated or truncated

        # Prefer episode_stats (set by env at termination); fall back to info
        stats = env.episode_stats if env.episode_stats else {
            k: info.get(k, 0.0) for k in _METRIC_KEYS
        }

        record = {
            "episode"    : ep,
            "scenario"   : scenario,
            "agent_type" : agent_type,
            "total_reward": float(total_r),
        }
        for k in _METRIC_KEYS:
            v = stats.get(k, info.get(k, 0.0))
            record[k] = float(v) if not isinstance(v, bool) else bool(v)

        records.append(record)
        env.close()

    return records


# ─── Statistics helpers ───────────────────────────────────────────────────────

def bootstrap_mean(data: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via percentile bootstrap."""
    rng = np.random.default_rng(0)
    boot_means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (
        float(data.mean()),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d (signed: positive means a < b, i.e. a is better for energy)."""
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def energy_saving_pct(baseline: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Per-episode energy saving % (positive = model uses less energy)."""
    return (baseline - model) / (baseline + 1e-12) * 100.0


def summarise(records: List[Dict]) -> Dict:
    """Mean ± std + bootstrap CIs for all collected metrics."""
    metrics = _METRIC_KEYS + ["total_reward"]
    summary: Dict[str, Any] = {}
    for m in metrics:
        vals = np.array([r[m] for r in records if m in r], dtype=float)
        if len(vals) == 0:
            continue
        mean, lo, hi = bootstrap_mean(vals)
        summary[f"{m}_mean"] = mean
        summary[f"{m}_std"]  = float(vals.std())
        summary[f"{m}_ci95_lo"] = lo
        summary[f"{m}_ci95_hi"] = hi
    return summary


def run_stat_tests(
    baseline_records: List[Dict],
    model_records: List[Dict],
    metric: str = "total_energy_wh",
) -> Dict[str, Any]:
    """
    Run Welch's t-test + Mann-Whitney U + Cohen's d comparing model vs baseline on metric.
    Positive saving% = model uses less energy (better).
    """
    base_vals  = np.array([r[metric] for r in baseline_records], dtype=float)
    model_vals = np.array([r[metric] for r in model_records],    dtype=float)

    # Paired saving %
    min_len   = min(len(base_vals), len(model_vals))
    savings   = energy_saving_pct(base_vals[:min_len], model_vals[:min_len])
    mean_sav, sav_lo, sav_hi = bootstrap_mean(savings)

    # Welch t-test (two-sided)
    t_stat, p_welch = scipy_stats.ttest_ind(model_vals, base_vals, equal_var=False)

    # Mann-Whitney U (non-parametric)
    u_stat, p_mwu = scipy_stats.mannwhitneyu(model_vals, base_vals, alternative="two-sided")

    # Effect size
    d = cohens_d(model_vals, base_vals)

    return {
        "mean_saving_pct"    : round(mean_sav, 3),
        "saving_ci95_lo"     : round(sav_lo, 3),
        "saving_ci95_hi"     : round(sav_hi, 3),
        "significant_p05"    : bool(p_welch < 0.05),
        "welch_t"            : round(float(t_stat), 4),
        "welch_p"            : round(float(p_welch), 6),
        "mannwhitney_u"      : round(float(u_stat), 1),
        "mannwhitney_p"      : round(float(p_mwu), 6),
        "cohens_d"           : round(d, 4),
        "n_baseline"         : len(base_vals),
        "n_model"            : len(model_vals),
    }


# ─── Main evaluation ─────────────────────────────────────────────────────────

def evaluate(
    model_dir: str = "results",
    n_episodes: int = 100,
    seed_start: int = 1000,
):
    """Main evaluation routine."""

    all_records         : List[Dict]       = []
    results_by_scenario : Dict[str, Any]   = {}
    stats_tests         : Dict[str, Any]   = {}

    pid_agent = WaypointPIDAgent()

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"  Evaluating scenario: {scenario.upper()}")
        print(f"{'='*60}")

        # ── PID baseline ──────────────────────────────────────────────
        print("  Running PID baseline...")
        pid_records = run_episodes(pid_agent, scenario, n_episodes, seed_start, agent_type="pid")
        all_records.extend(pid_records)
        pid_summary = summarise(pid_records)
        print(
            f"  PID     | energy={pid_summary.get('total_energy_wh_mean', 0):.4f} Wh"
            f" | {pid_summary.get('total_energy_wh_mean',0)/max(pid_summary.get('path_length_m_mean',1)/1000,1e-9):.3f} Wh/km"
            f" | speed={pid_summary.get('mean_speed_ms_mean', 0):.2f} m/s"
            f" | success={pid_summary.get('mission_complete_mean', 0)*100:.1f}%"
        )

        # ── A* Energy baseline (B1) ───────────────────────────────────
        print("  Running A* baseline...")
        astar_agent = AStarEnergyAgent()
        astar_records = run_episodes(astar_agent, scenario, n_episodes, seed_start, agent_type="astar")
        all_records.extend(astar_records)
        astar_summary = summarise(astar_records)
        print(
            f"  A*(B1)  | energy={astar_summary.get('total_energy_wh_mean', 0):.4f} Wh"
            f" | {astar_summary.get('total_energy_wh_mean',0)/max(astar_summary.get('path_length_m_mean',1)/1000,1e-9):.3f} Wh/km"
            f" | speed={astar_summary.get('mean_speed_ms_mean', 0):.2f} m/s"
            f" | success={astar_summary.get('mission_complete_mean', 0)*100:.1f}%"
        )

        # ── PPO agent (multi-seed) ─────────────────────────────────────
        ppo_records_all : List[Dict] = []
        seeds_found     : List[int]  = []

        for seed in [42, 43, 44]:
            best_path  = os.path.join(model_dir, f"best_ppo_{scenario}_s{seed}", "best_model.zip")
            final_path = os.path.join(model_dir, f"ppo_{scenario}_s{seed}.zip")
            model_path = best_path if os.path.exists(best_path) else final_path
            if SB3_AVAILABLE and os.path.exists(model_path):
                print(f"  Loading PPO s{seed}: {model_path}")
                ppo_agent = PPO.load(model_path)
                seed_records = run_episodes(
                    ppo_agent, scenario, n_episodes, seed_start, agent_type=f"ppo_s{seed}"
                )
                ppo_records_all.extend(seed_records)
                seeds_found.append(seed)

        ppo_summary  : Optional[Dict] = None
        ppo_vs_pid   : Optional[Dict] = None
        ppo_vs_astar : Optional[Dict] = None

        if ppo_records_all:
            all_records.extend(ppo_records_all)
            ppo_summary = summarise(ppo_records_all)
            ppo_summary["seeds_evaluated"] = seeds_found

            # Statistical tests vs PID
            ppo_vs_pid   = run_stat_tests(pid_records,   ppo_records_all)
            # Statistical tests vs A*
            ppo_vs_astar = run_stat_tests(astar_records, ppo_records_all)

            print(
                f"  PPO ({len(seeds_found)}s) | energy={ppo_summary.get('total_energy_wh_mean', 0):.4f}"
                f" ± {ppo_summary.get('total_energy_wh_std', 0):.4f} Wh"
                f" | {ppo_summary.get('total_energy_wh_mean',0)/max(ppo_summary.get('path_length_m_mean',1)/1000,1e-9):.3f} Wh/km"
                f" | speed={ppo_summary.get('mean_speed_ms_mean', 0):.2f} m/s"
                f" | success={ppo_summary.get('mission_complete_mean', 0)*100:.1f}%"
            )
            print(
                f"         | saving vs PID: {ppo_vs_pid['mean_saving_pct']:+.1f}%"
                f" [{ppo_vs_pid['saving_ci95_lo']:+.1f}%, {ppo_vs_pid['saving_ci95_hi']:+.1f}%] 95% CI"
                f" | p={ppo_vs_pid['welch_p']:.4f} | d={ppo_vs_pid['cohens_d']:.2f}"
            )
        else:
            if not SB3_AVAILABLE:
                print("  PPO model skipped — stable-baselines3 not installed.")
            else:
                print(f"  No PPO models found for scenario '{scenario}' in {model_dir}")

        results_by_scenario[scenario] = {
            "pid"          : pid_summary,
            "astar"        : astar_summary,
            "ppo"          : ppo_summary,
            "ppo_vs_pid"   : ppo_vs_pid,
            "ppo_vs_astar" : ppo_vs_astar,
        }

    # ── Save results ──────────────────────────────────────────────────────────
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
    _print_stat_table(results_by_scenario)
    return out


# ─── Pretty-print tables ──────────────────────────────────────────────────────

def _print_table(results_by_scenario: Dict):
    """Print compact per-scenario comparison table with fairness metrics."""
    W = 95
    print("\n" + "=" * W)
    print(
        f"{'SCENARIO':<10} {'AGENT':<8} {'ENERGY(Wh)':<17}"
        f" {'Wh/km':<10} {'SPEED(m/s)':<11} {'SUCCESS%':<10} {'AVG REWARD':<12}"
    )
    print("=" * W)
    for sc, agents in results_by_scenario.items():
        for agent_key, s in agents.items():
            if s is None or not isinstance(s, dict):
                continue
            if agent_key in ("ppo_vs_pid", "ppo_vs_astar"):
                continue
            energy    = s.get("total_energy_wh_mean", 0.0)
            energy_sd = s.get("total_energy_wh_std",  0.0)
            ekm       = s.get("energy_per_km_mean",    0.0)
            speed     = s.get("mean_speed_ms_mean",    0.0)
            success   = s.get("mission_complete_mean", 0.0) * 100
            reward    = s.get("total_reward_mean",     0.0)
            label     = agent_key.upper()
            print(
                f"{sc:<10} {label:<8} {energy:.4f} ± {energy_sd:.4f}  "
                f" {ekm:<10.3f} {speed:<11.2f} {success:<10.1f} {reward:<12.1f}"
            )
    print("=" * W)


def _print_stat_table(results_by_scenario: Dict):
    """Print statistical test results comparing PPO vs PID."""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS: PPO vs PID (energy_wh, Welch t-test + bootstrap 95% CI)")
    print("=" * 80)
    print(f"{'SCENARIO':<10} {'SAVING%':<12} {'95% CI':<22} {'p-value':<12} {'Cohen d':<10} {'SIG?'}")
    print("-" * 80)
    for sc, d in results_by_scenario.items():
        t = d.get("ppo_vs_pid")
        if t is None:
            continue
        ci_str = f"[{t['saving_ci95_lo']:+.1f}%, {t['saving_ci95_hi']:+.1f}%]"
        sig    = "YES ✓" if t["significant_p05"] else "NO"
        print(
            f"{sc:<10} {t['mean_saving_pct']:+.2f}%      {ci_str:<22}"
            f" {t['welch_p']:<12.4f} {t['cohens_d']:<10.3f} {sig}"
        )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",  default="results")
    parser.add_argument("--episodes",   default=100, type=int)
    parser.add_argument("--seed_start", default=1000, type=int)
    args = parser.parse_args()

    evaluate(
        model_dir  = args.model_dir,
        n_episodes = args.episodes,
        seed_start = args.seed_start,
    )
