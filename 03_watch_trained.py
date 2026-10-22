"""Watch or quickly run episodes with a trained LunarLander policy."""

import argparse
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/lunar_lander_built_in", help="Model path with or without .zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fast", action="store_true", help="Disable graphics for faster simulation")
    parser.add_argument("--continuous", action="store_true", help="Keep launching episodes; timer resets after each landing or crash")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    model = PPO.load(str(model_path))
    env = gym.make("LunarLander-v3", render_mode=None if args.fast else "human")
    scores = []
    try:
        episode = 0
        while args.continuous or episode < args.episodes:
            episode += 1
            observation, _ = env.reset(seed=1000 + episode)
            score, steps, done = 0.0, 0, False
            started = time.perf_counter()
            last_clock_update = started
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                steps += 1
                done = terminated or truncated
                now = time.perf_counter()
                if now - last_clock_update >= 1.0:
                    print(f"Episode {episode} | flight timer: {now - started:5.1f}s", end="\r", flush=True)
                    last_clock_update = now
            elapsed = time.perf_counter() - started
            if not args.continuous:
                scores.append(score)
            result = "LANDED" if score >= 100 else "CRASHED/UNSUCCESSFUL"
            print(f"Episode {episode}: {result}, reward={score:.1f}, steps={steps}, flight time={elapsed:.2f}s")
    finally:
        env.close()
    if scores:
        print(f"Average reward: {sum(scores) / len(scores):.1f}")


if __name__ == "__main__":
    main()
