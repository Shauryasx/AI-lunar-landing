"""Train a PPO agent for Gymnasium's LunarLander-v3 environment."""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_wrapper import QuickHoverWrapper, TimePenaltyWrapper

ROOT = Path(__file__).resolve().parent


def make_env(reward_mode: str):
    env = gym.make("LunarLander-v3")
    if reward_mode == "time_penalty":
        # A small per-frame cost discourages unnecessary hovering. The saved
        # policy must be retrained with this option to learn the behavior.
        env = TimePenaltyWrapper(env, penalty=0.03)
    elif reward_mode == "quick_hover":
        env = QuickHoverWrapper(env, penalty=0.05, hover_bonus=1.0)
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward", choices=("built_in", "time_penalty", "quick_hover"), default="quick_hover")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", help="Optional model name (without .zip).")
    args = parser.parse_args()
    if args.timesteps <= 0:
        parser.error("--timesteps must be greater than zero")

    model_name = args.name or f"lunar_lander_{args.reward}"
    model_path = ROOT / "models" / model_name
    log_path = ROOT / "logs" / args.reward
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    env = make_env(args.reward)
    try:
        model = PPO(
            "MlpPolicy", env, verbose=1, seed=args.seed,
            tensorboard_log=str(log_path), policy_kwargs={"net_arch": [64, 64]},
            learning_rate=3e-4, n_steps=1024, batch_size=64,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        )
        print(f"Training {args.reward} for {args.timesteps:,} steps; saving to {model_path}.zip")
        model.learn(total_timesteps=args.timesteps, progress_bar=True)
        model.save(str(model_path))
        print(f"Saved model: {model_path}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
