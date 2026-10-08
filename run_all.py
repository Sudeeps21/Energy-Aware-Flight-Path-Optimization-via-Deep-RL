"""
run_all.py — One-command pipeline to run the full research workflow.

  Step 1: Sanity checks (always runs)
  Step 2: Generate model-only figures (energy model validation, etc.)
  Step 3: Train PPO on all scenarios (skippable with --skip_train)
  Step 4: Evaluate all agents (PID + PPO) across scenarios
  Step 5: Generate all paper figures

Usage
-----
  python run_all.py                           # full pipeline
  python run_all.py --skip_train              # skip training (use existing models)
  python run_all.py --timesteps 200000        # quicker training run
  python run_all.py --scenarios calm windy    # subset of scenarios
"""

import argparse
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = ["calm", "windy", "cold"]


def run_step(label: str, cmd: list, cwd: str = None):
    print(f"\n{'='*65}")
    print(f"  STEP: {label}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, cwd=cwd or os.getcwd())
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed (returncode={result.returncode}).")
        print("  Fix the error above before proceeding.\n")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip training, use existing models in results/")
    parser.add_argument("--timesteps",   default=500_000,  type=int)
    parser.add_argument("--seed",        default=42,        type=int)
    parser.add_argument("--n_envs",      default=4,         type=int)
    parser.add_argument("--episodes",    default=50,        type=int,
                        help="Evaluation episodes per agent per scenario")
    parser.add_argument("--scenarios",   nargs="+",
                        default=SCENARIOS,
                        choices=SCENARIOS)
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    src = os.path.dirname(os.path.abspath(__file__))

    # ── Step 1: Sanity checks ────────────────────────────────────────────
    run_step("Sanity checks",
             [sys.executable, os.path.join(src, "scripts", "sanity_check.py")])

    # ── Step 2: Model-only figures ───────────────────────────────────────
    run_step("Generate model figures (pre-training)",
             [sys.executable, os.path.join(src, "drone_energy", "viz", "plots.py"),
              "--results_dir", args.results_dir])

    # ── Step 3: Training ─────────────────────────────────────────────────
    if not args.skip_train:
        for scenario in args.scenarios:
            run_step(f"Train PPO — {scenario}",
                     [sys.executable, os.path.join(src, "drone_energy", "rl", "train.py"),
                      "--scenario",  scenario,
                      "--timesteps", str(args.timesteps),
                      "--seed",      str(args.seed),
                      "--n_envs",    str(args.n_envs),
                      "--save_dir",  args.results_dir])
    else:
        print("\n[SKIP] Training skipped (--skip_train). Using existing models.")

    # ── Step 4: Evaluation ───────────────────────────────────────────────
    run_step("Evaluate agents (PID + PPO)",
             [sys.executable, os.path.join(src, "drone_energy", "eval", "evaluate.py"),
              "--model_dir", args.results_dir,
              "--episodes",  str(args.episodes)])

    # ── Step 5: All figures ───────────────────────────────────────────────
    run_step("Generate all paper figures",
             [sys.executable, os.path.join(src, "drone_energy", "viz", "plots.py"),
              "--results_dir", args.results_dir])

    print(f"\n{'='*65}")
    print("  ALL STEPS COMPLETE")
    print(f"  Results  → {args.results_dir}/")
    print(f"  Figures  → {args.results_dir}/figures/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
