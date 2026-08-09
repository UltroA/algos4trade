"""
DDPG (Deep Deterministic Policy Gradient) - RL for optimizing position size.

DDPG(PPO/SAC/DDPG) is one of the first successful
off-policy algorithms for continuous control, historically used in
early work on order execution and position sizing. Unlike PPO
(on-policy) and SAC (stochastic policy + entropy bonus), DDPG learns a
DETERMINISTIC policy via the action gradient of the Q-function and
explores the action space by adding Gaussian noise during
experience collection. As with PPO/SAC in this project, the risk of overfitting
to a specific backtest is not eliminated - this is an educational implementation.

The agent trades a SINGLE asset in core.trading_env.TradingEnv (see ppo_agent.py
for a description of the environment: state is a window of returns + position, action is
the target position [-1, 1], reward is pnl minus the cost of changing position).
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


class _DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


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


class DDPGAgent(SingleAssetAlgorithm):
    name = "DDPG Position Sizer"
    category = AlgorithmCategory.REINFORCEMENT_LEARNING
    description = (
        "Off-policy RL agent (Deep Deterministic Policy Gradient) learns to choose a continuous "
        "position size [-1, 1] via deterministic policy gradient through a critic Q(s, a)."
    )

    def __init__(
        self,
        window: int = 20,
        hidden: int = 64,
        episodes: int = 5,
        updates_per_step: int = 1,
        batch_size: int = 64,
        buffer_size: int = 2000,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        exploration_std: float = 0.1,
        warmup_steps: int = 200,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            window=window, hidden=hidden, episodes=episodes, updates_per_step=updates_per_step,
            batch_size=batch_size, buffer_size=buffer_size, actor_lr=actor_lr, critic_lr=critic_lr,
            gamma=gamma, tau=tau, exploration_std=exploration_std, warmup_steps=warmup_steps,
            seed=seed, **kwargs,
        )
        self.window = window
        self.hidden = hidden
        self.episodes = episodes
        self.updates_per_step = updates_per_step
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.exploration_std = exploration_std
        self.warmup_steps = warmup_steps
        self.seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.actor: _DeterministicActor | None = None
        self.actor_target: _DeterministicActor | None = None
        self.critic: _QCritic | None = None
        self.critic_target: _QCritic | None = None
        self.replay: deque = deque(maxlen=buffer_size)

    def fit(self, train_data: pd.DataFrame) -> "DDPGAgent":
        if len(train_data) <= self.window + 5:
            self.is_fitted = False
            return self

        env = TradingEnv(train_data["close"], window=self.window)
        state_dim = env.state_dim
        self.actor = _DeterministicActor(state_dim, self.hidden)
        self.actor_target = _DeterministicActor(state_dim, self.hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = _QCritic(state_dim, self.hidden)
        self.critic_target = _QCritic(state_dim, self.hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())

        actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        total_steps = 0
        for _ in range(self.episodes):
            state = env.reset()
            done = False
            while not done:
                if total_steps < self.warmup_steps:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    with torch.no_grad():
                        raw_action = self.actor(torch.as_tensor(state, dtype=torch.float32)).item()
                    noise = np.random.normal(0.0, self.exploration_std)
                    action = float(np.clip(raw_action + noise, -1.0, 1.0))

                next_state, reward, done, _ = env.step(action)
                self.replay.append((state, action, reward, next_state, float(done)))
                state = next_state
                total_steps += 1

                if len(self.replay) >= self.batch_size:
                    for _ in range(self.updates_per_step):
                        self._update(actor_optim, critic_optim)

        self.is_fitted = True
        return self

    def _update(self, actor_optim, critic_optim):
        batch = random.sample(self.replay, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
        rewards_t = torch.as_tensor(np.array(rewards), dtype=torch.float32)
        next_states_t = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.as_tensor(np.array(dones), dtype=torch.float32)

        with torch.no_grad():
            next_action = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_action)
            target_value = rewards_t + (1 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_value)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

        actor_loss = -self.critic(states_t, self.actor(states_t)).mean()
        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * param)
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * param)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.actor is None or len(data) <= self.window + 1:
            return pd.Series(0.0, index=data.index)

        env = TradingEnv(data["close"], window=self.window)
        state = env.reset()
        positions = [0.0] * env.window
        with torch.no_grad():
            for _ in range(env.n_steps - env.window):
                state_t = torch.as_tensor(state, dtype=torch.float32)
                action = float(self.actor(state_t).item())
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
    algo = DDPGAgent(episodes=2, warmup_steps=50, buffer_size=500)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
