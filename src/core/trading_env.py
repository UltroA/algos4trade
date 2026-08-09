"""
Lightweight trading environment for RL algorithms (PPO/SAC/DDPG), with no
dependency on gymnasium - numpy only. Shared by all RL files in src/algorithms/.

State: sliding window of the last `window` returns + current position.
Action: target position in [-1, 1] (short..long), continuous.
Reward: portfolio return per step minus the cost of changing position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TradingEnv:
    def __init__(self, close: pd.Series, window: int = 20, transaction_cost_bps: float = 5.0):
        self.returns = close.pct_change().fillna(0.0).to_numpy(dtype=np.float32)
        self.window = window
        self.cost = transaction_cost_bps / 10_000.0
        self.n_steps = len(self.returns)
        self.reset()

    @property
    def state_dim(self) -> int:
        return self.window + 1

    def reset(self) -> np.ndarray:
        self.t = self.window
        self.position = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        window_returns = self.returns[self.t - self.window : self.t]
        return np.concatenate([window_returns, [self.position]]).astype(np.float32)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, -1.0, 1.0))
        step_return = self.returns[self.t]
        cost = abs(action - self.position) * self.cost
        reward = self.position * step_return - cost
        self.position = action
        self.t += 1
        done = self.t >= self.n_steps
        state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return state, float(reward), done, {}
