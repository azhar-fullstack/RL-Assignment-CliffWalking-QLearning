"""Tabular Q-Learning agent implemented from scratch."""

from __future__ import annotations

from typing import Optional

import numpy as np


class QLearningAgent:
    """ε-greedy Q-Learning agent for discrete Gymnasium environments.

    Update rule:
        Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int, explore: bool = True) -> int:
        """Select an action using ε-greedy (or greedy if explore=False)."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Apply one Q-Learning TD update. Returns the TD error."""
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        """Exponentially decay exploration rate toward epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Persist Q-table and hyperparameters."""
        np.savez(
            path,
            q_table=self.q_table,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
        )

    def load(self, path: str) -> None:
        """Load Q-table and hyperparameters from disk."""
        data = np.load(path)
        self.q_table = data["q_table"]
        self.alpha = float(data["alpha"])
        self.gamma = float(data["gamma"])
        self.epsilon = float(data["epsilon"])
        self.epsilon_min = float(data["epsilon_min"])
        self.epsilon_decay = float(data["epsilon_decay"])
