"""Reward shaping helpers for LunarLander-v3."""

import gymnasium as gym


class TimePenaltyWrapper(gym.Wrapper):
    """Apply a small per-step cost to encourage a prompt, controlled landing."""

    def __init__(self, env, penalty: float = 0.03):
        super().__init__(env)
        if penalty < 0:
            raise ValueError("penalty must be non-negative")
        self.penalty = float(penalty)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["base_reward"] = reward
        info["time_penalty"] = -self.penalty
        return observation, reward - self.penalty, terminated, truncated, info


class QuickHoverWrapper(TimePenaltyWrapper):
    """Reward a single brief, controlled hover near the ground, then landing."""

    def __init__(self, env, penalty: float = 0.05, hover_bonus: float = 1.0):
        super().__init__(env, penalty=penalty)
        self.hover_bonus = float(hover_bonus)
        self.hover_rewarded = False

    def reset(self, **kwargs):
        self.hover_rewarded = False
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["base_reward"] = reward
        info["time_penalty"] = -self.penalty
        height, vx, vy, angle, angular_velocity = map(float, observation[1:6])
        no_leg_contact = not (observation[6] > 0.5 or observation[7] > 0.5)
        controlled_hover = (
            0.08 < height < 0.25 and abs(vx) < 0.12 and abs(vy) < 0.12
            and abs(angle) < 0.12 and abs(angular_velocity) < 0.12 and no_leg_contact
        )
        bonus = 0.0
        if controlled_hover and not self.hover_rewarded:
            bonus = self.hover_bonus
            self.hover_rewarded = True
        info["hover_bonus"] = bonus
        return observation, reward - self.penalty + bonus, terminated, truncated, info
