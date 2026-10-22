"""Evaluate a saved model quickly without opening a graphics window."""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/lunar_lander_built_in")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = PPO.load(str(model_path))
    env = gym.make("LunarLander-v3")
    scores, lengths = [], []
    started = time.perf_counter()
    try:
        for i in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + i)
            score, steps, done = 0.0, 0, False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                steps += 1
                done = terminated or truncated
            scores.append(score)
            lengths.append(steps)
            print(f"Episode {i + 1:02}: reward={score:7.2f}, steps={steps}")
    finally:
        env.close()
    successes = sum(score >= 200 for score in scores)
    print(f"\nMean reward: {sum(scores) / len(scores):.2f}")
    print(f"Mean episode length: {sum(lengths) / len(lengths):.1f} steps")
    print(f"Successful episodes (reward >= 200): {successes}/{len(scores)}")
    print(f"Evaluation wall time: {time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
