"""
PPO (Proximal Policy Optimization) - RL для оптимизации размера позиции.

RL реально работает в задачах исполнения ордеров/сайзинга позиции, но в задаче генерации сигналов почти всегда
переобучается на бэктесте - этот риск здесь не устранён (задача учебная),
только частично смягчён небольшим числом эпох и энтропийным бонусом.

Агент торгует ОДНИМ активом в core.trading_env.TradingEnv: состояние - окно
последних доходностей + текущая позиция, действие - целевая позиция [-1, 1]
(непрерывная, через tanh на выходе политики), награда - доходность портфеля
за шаг минус издержки на изменение позиции.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.base import AlgorithmCategory, SingleAssetAlgorithm
from core.trading_env import TradingEnv


class _GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1) - 0.5)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h = self.body(state)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(h).squeeze(-1)
        return mean, std, value


class PPOAgent(SingleAssetAlgorithm):
    name = "PPO Position Sizer"
    category = AlgorithmCategory.REINFORCEMENT_LEARNING
    description = (
        "RL-агент (Proximal Policy Optimization) учится выбирать непрерывный размер позиции "
        "[-1, 1], максимизируя доходность за вычетом издержек на сделку."
    )

    def __init__(
        self,
        window: int = 20,
        hidden: int = 64,
        epochs: int = 8,
        rollout_steps: int = 256,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            window=window, hidden=hidden, epochs=epochs, rollout_steps=rollout_steps,
            clip_ratio=clip_ratio, lr=lr, gamma=gamma, entropy_coef=entropy_coef, seed=seed, **kwargs,
        )
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.rollout_steps = rollout_steps
        self.clip_ratio = clip_ratio
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        torch.manual_seed(seed)
        self.policy: _GaussianPolicy | None = None

    def fit(self, train_data: pd.DataFrame) -> "PPOAgent":
        if len(train_data) <= self.window + 5:
            self.is_fitted = False
            return self

        env = TradingEnv(train_data["close"], window=self.window)
        self.policy = _GaussianPolicy(env.state_dim, self.hidden)
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            states, actions, log_probs, rewards, values = self._collect_rollout(env)
            returns, advantages = self._compute_gae(rewards, values)
            self._ppo_update(optimizer, states, actions, log_probs, returns, advantages)

        self.is_fitted = True
        return self

    def _collect_rollout(self, env: TradingEnv):
        state = env.reset()
        states, actions, log_probs, rewards, values = [], [], [], [], []
        for _ in range(min(self.rollout_steps, env.n_steps - env.window - 1)):
            state_t = torch.as_tensor(state, dtype=torch.float32)
            mean, std, value = self.policy(state_t)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum()

            next_state, reward, done, _ = env.step(raw_action.item())
            states.append(state_t)
            actions.append(raw_action.detach())
            log_probs.append(log_prob.detach())
            rewards.append(reward)
            values.append(value.detach())

            state = next_state
            if done:
                state = env.reset()
        return states, actions, log_probs, rewards, values

    def _compute_gae(self, rewards: list[float], values: list[torch.Tensor], lam: float = 0.95):
        values = [v.item() for v in values] + [0.0]
        advantages, gae = [], 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * lam * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _ppo_update(self, optimizer, states, actions, old_log_probs, returns, advantages):
        states_t = torch.stack(states)
        actions_t = torch.stack(actions)
        old_log_probs_t = torch.stack(old_log_probs)

        mean, std, values = self.policy(states_t)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions_t).sum(-1)
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - old_log_probs_t).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if not self.is_fitted or self.policy is None or len(data) <= self.window + 1:
            return pd.Series(0.0, index=data.index)

        env = TradingEnv(data["close"], window=self.window)
        state = env.reset()
        positions = [0.0] * env.window
        with torch.no_grad():
            for _ in range(env.n_steps - env.window):
                state_t = torch.as_tensor(state, dtype=torch.float32)
                mean, _, _ = self.policy(state_t)
                action = float(mean.item())
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
    algo = PPOAgent(epochs=3, rollout_steps=64)
    algo.fit(train_df)
    signals = algo.generate_signals(test_df)
    print(signals.describe())
