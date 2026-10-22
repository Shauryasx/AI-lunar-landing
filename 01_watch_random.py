"""Watch an untrained LunarLander agent taking random actions."""

import gymnasium as gym


def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    try:
        for episode in range(1, 6):
            observation, _ = env.reset(seed=episode)
            score, done = 0.0, False
            while not done:
                observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                score += float(reward)
                done = terminated or truncated
            print(f"Episode {episode}: reward={score:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
