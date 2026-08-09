"""
SAC (Soft Actor-Critic) - RL for optimizing position size.

unlike PPO (on-policy), SAC is an off-policy algorithm with a replay buffer and maximizes reward + policy
entropy (better exploration of the action space, typically higher
sample efficiency). Just like PPO/DDPG in this project, the risk of overfitting
to the specific backtest is not eliminated (see the table: "signal generation
is almost always overfit to the backtest") - this is an educational implementation here.

Simplifications relative to the original Haarnoja et al. (2018) paper:
  * entropy temperature alpha - a fixed hyperparameter (no auto-tuning);
  * a single Q-critic instead of a double one (typical clipped double-Q) - for simplicity;
  * the log-probability of the action after tanh squashing is computed approximately,
    without a strict tanh Jacobian correction (using the standard formula
    log_prob(raw_action) - log(1 - tanh(raw_action)^2 + eps), as in the paper,
    but without additional numerical stabilizations).

The agent trades a SINGLE asset in core.trading_env.TradingEnv (see ppo_agent.py
for the base environment description: state - a window of returns + position,
action - target position [-1, 1], reward - pnl minus costs).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.trading_env import TradingEnv


class _GaussianTanhPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h = self.body(state)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), -5.0, 2.0)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw_action = mean + std * eps
        action = torch.tanh(raw_action)
        # approximate log-probability of the action after tanh squashing (see file docstring)
        log_prob = (-0.5 * ((raw_action - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def deterministic_action(self, state: torch.Tensor):
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class _QCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class SACAgent(SingleAssetAlgorithm):
    name = "SAC Position Sizer"
    category = AlgorithmCategory.REINFORCEMENT_LEARNING
    description = (
        "An off-policy RL agent (Soft Actor-Critic) learns to choose a continuous position size "
        "[-1, 1], maximizing return net of costs plus an entropy exploration bonus."
    )

    def __init__(
        self,
        window: int = 20,
        hidden: int = 64,
        episodes: int = 5,
        updates_per_step: int = 1,
        batch_size: int = 64,
        buffer_size: int = 2000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        warmup_steps: int = 200,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            window=window, hidden=hidden, episodes=episodes, updates_per_step=updates_per_step,
            batch_size=batch_size, buffer_size=buffer_size, lr=lr, gamma=gamma, tau=tau,
            alpha=alpha, warmup_steps=warmup_steps, seed=seed, **kwargs,
        )
        self.window = window
        self.hidden = hidden
        self.episodes = episodes
        self.updates_per_step = updates_per_step
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)

        self.policy: _GaussianTanhPolicy | None = None
        self.critic: _QCritic | None = None
        self.critic_target: _QCritic | None = None
        self.replay: deque = deque(maxlen=buffer_size)

    def fit(self, train_data: pd.DataFrame) -> "SACAgent":
        if len(train_data) <= self.window + 5:
            self.is_fitted = False
            return self

        env = TradingEnv(train_data["close"], window=self.window)
        state_dim = env.state_dim
        self.policy = _GaussianTanhPolicy(state_dim, self.hidden)
        self.critic = _QCritic(state_dim, self.hidden)
        self.critic_target = _QCritic(state_dim, self.hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        policy_optim = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        total_steps = 0
        for _ in range(self.episodes):
            state = env.reset()
            done = False
            while not done:
                if total_steps < self.warmup_steps:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    with torch.no_grad():
                        action_t, _ = self.policy.sample(torch.as_tensor(state, dtype=torch.float32))
                    action = float(action_t.item())

                next_state, reward, done, _ = env.step(action)
                self.replay.append((state, action, reward, next_state, float(done)))
                state = next_state
                total_steps += 1

                if len(self.replay) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        self._update(policy_optim, critic_optim)

        self.is_fitted = True
        return self

    def _update(self, policy_optim, critic_optim):
        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32)

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_states_t)
            target_q = self.critic_target(next_states_t, next_action)
            target_value = rewards_t + (1 - dones_t) * self.gamma * (target_q - self.alpha * next_log_prob)

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_value)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        new_action, log_prob = self.policy.sample(states_t)
        q_new = self.critic(states_t, new_action)
        policy_loss = (self.alpha * log_prob - q_new).mean()
        policy_optim.zero_grad()
        policy_loss.backward()
        policy_optim.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * param)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.policy is None or len(data) <= self.window + 1:
            return pd.Series(0.0, index=data.index)

        env = TradingEnv(data["close"], window=self.window)
        state = env.reset()
        positions = [0.0] * env.window
        with torch.no_grad():
            for _ in range(env.n_steps - env.window):
                state_t = torch.as_tensor(state, dtype=torch.float32)
                action = float(self.policy.deterministic_action(state_t).item())
                positions.append(action)
                state, _, done, _ = env.step(action)
                if done:
                    break

        positions = positions[: len(data)]
        positions += [0.0] * (len(data) - len(positions))
        return pd.Series(positions, index=data.index).clip(-1.0, 1.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    price = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    dummy = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1_000_000}, index=idx)

    train_df, test_df = dummy.iloc[:210], dummy.iloc[210:]
    algo = SACAgent(episodes=2, warmup_steps=50, buffer_size=500)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
