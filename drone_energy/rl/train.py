"""
train_ppo.py — Train a PPO agent on MultiFactorDroneEnv.

Usage
-----
  python train_ppo.py --scenario calm --timesteps 500000 --seed 42

The trained model is saved to results/ppo_<scenario>_<seed>.zip.
Training curves are logged for later plotting.
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Lazy SB3 import with friendly error ───────────────────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from drone_energy.envs.multi_factor_drone_env import MultiFactorDroneEnv


def make_env(scenario: str, seed: int, randomise: bool = False):
    """Factory for a single monitored environment instance."""
    def _init():
        env = MultiFactorDroneEnv(scenario=scenario, seed=seed, randomise_scenario=randomise)
        env = Monitor(env)
        return env
    return _init


def train(
    scenario: str = "calm",
    timesteps: int = 500_000,
    seed: int = 42,
    n_envs: int = 4,
    save_dir: str = "results",
):
    """Run PPO training and save the model + training log."""

    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 is not installed.")
        print("Run:  pip install stable-baselines3")
        return

    os.makedirs(save_dir, exist_ok=True)
    run_name   = f"ppo_{scenario}_s{seed}"
    model_path = os.path.join(save_dir, run_name)
    log_path   = os.path.join(save_dir, f"{run_name}_log.json")

    print(f"\n=== Training PPO | scenario={scenario} | timesteps={timesteps} | seed={seed} ===\n")

    # ── Vectorised training envs ───────────────────────────────────────────
    train_env = make_vec_env(
        make_env(scenario, seed, randomise=False),
        n_envs=n_envs,
        seed=seed,
    )

    # ── Evaluation env ─────────────────────────────────────────────────────
    eval_env = Monitor(MultiFactorDroneEnv(scenario=scenario, seed=seed + 1))

    # ── Callbacks ──────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_" + run_name),
        log_path=os.path.join(save_dir, "eval_" + run_name),
        eval_freq=max(timesteps // 20, 5000),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(timesteps // 10, 10000),
        save_path=os.path.join(save_dir, "checkpoints_" + run_name),
        name_prefix="ckpt",
    )

    # ── Model ──────────────────────────────────────────────────────────────
    tb_log = os.path.join(save_dir, "tensorboard")
    try:
        import tensorboard
    except ImportError:
        tb_log = None

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=tb_log,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    try:
        import tqdm, rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, checkpoint_cb],
        progress_bar=use_progress_bar,
    )

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(model_path)
    print(f"\nModel saved → {model_path}.zip")

    # Log metadata
    meta = {
        "scenario": scenario,
        "timesteps": timesteps,
        "seed": seed,
        "model_path": model_path + ".zip",
    }
    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)

    train_env.close()
    eval_env.close()
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on MultiFactorDroneEnv")
    parser.add_argument("--scenario",   default="calm",    choices=["calm", "windy", "cold", "random"])
    parser.add_argument("--timesteps",  default=500_000,   type=int)
    parser.add_argument("--seed",       default=42,        type=int)
    parser.add_argument("--n_envs",     default=4,         type=int)
    parser.add_argument("--save_dir",   default="results")
    args = parser.parse_args()

    train(
        scenario=args.scenario,
        timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
    )
