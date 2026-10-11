"""Training helpers, plotting, and GIF rendering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import QLearningAgent


def make_env(env_id: str = "CliffWalking-v1", render_mode: Optional[str] = None) -> gym.Env:
    """Create and return a Gymnasium environment."""
    return gym.make(env_id, render_mode=render_mode)


def train_agent(
    env: gym.Env,
    agent: QLearningAgent,
    n_episodes: int = 5000,
    max_steps: int = 200,
    log_every: int = 100,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train a Q-Learning agent and return episode metrics."""
    rewards: List[float] = []
    lengths: List[int] = []
    epsilons: List[float] = []
    successes: List[float] = []

    for ep in tqdm(range(n_episodes), desc="Training"):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        steps = 0
        reached_goal = False

        for _ in range(max_steps):
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, float(reward), next_state, done)
            state = next_state
            total_reward += float(reward)
            steps += 1
            if done:
                # CliffWalking: goal reward is -1 and terminates without falling
                reached_goal = terminated and reward > -100
                break

        agent.decay_epsilon()
        rewards.append(total_reward)
        lengths.append(steps)
        epsilons.append(agent.epsilon)
        successes.append(1.0 if reached_goal else 0.0)

        if (ep + 1) % log_every == 0:
            avg_r = np.mean(rewards[-log_every:])
            avg_s = np.mean(successes[-log_every:]) * 100
            tqdm.write(
                f"Episode {ep + 1:5d} | avg reward {avg_r:7.2f} | "
                f"success {avg_s:5.1f}% | eps={agent.epsilon:.3f}"
            )

    return {
        "rewards": rewards,
        "lengths": lengths,
        "epsilons": epsilons,
        "successes": successes,
    }


def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """Compute a simple moving average for smoother curves."""
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def evaluate_agent(
    env: gym.Env,
    agent: QLearningAgent,
    n_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 10_000,
) -> Dict[str, float]:
    """Evaluate a greedy policy without exploration."""
    rewards = []
    successes = 0

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            if terminated or truncated:
                if terminated and reward > -100:
                    successes += 1
                break
        rewards.append(total)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "success_rate": successes / n_episodes,
    }


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    window: int = 100,
    title: str = "Q-Learning Training Progress",
) -> None:
    """Save reward, success-rate, and epsilon plots."""
    rewards = moving_average(history["rewards"], window)
    successes = moving_average(history["successes"], window)
    epsilons = history["epsilons"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    axes[0].plot(rewards, color="#1f77b4")
    axes[0].set_title(f"Episode Reward (MA-{window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(successes, color="#2ca02c")
    axes[1].set_title(f"Success Rate (MA-{window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epsilons, color="#d62728")
    axes[2].set_title("Exploration Rate (ε)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("ε")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hyperparameter_comparison(
    results: Dict[str, List[float]],
    save_path: str,
    ylabel: str = "Mean Episode Reward (last 500)",
    title: str = "Hyperparameter Sensitivity",
) -> None:
    """Bar chart comparing hyperparameter configurations."""
    labels = list(results.keys())
    values = [float(np.mean(v)) for v in results.values()]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_q_heatmap(q_table: np.ndarray, save_path: str, n_rows: int = 4, n_cols: int = 12) -> None:
    """Visualize max Q-value per CliffWalking grid cell."""
    max_q = q_table.max(axis=1).reshape(n_rows, n_cols)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(max_q, cmap="viridis")
    ax.set_title("Learned State Values (max_a Q(s,a))")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_policy_gif(
    env_id: str,
    agent: QLearningAgent,
    save_path: str,
    max_steps: int = 100,
    seed: int = 0,
    fps: int = 4,
) -> Tuple[float, int, bool]:
    """Roll out the greedy policy and save a GIF of the trajectory."""
    env = make_env(env_id, render_mode="rgb_array")
    state, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(max_steps):
        action = agent.select_action(state, explore=False)
        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            success = terminated and reward > -100
            break

    env.close()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return total_reward, steps, success


def run_hyperparameter_sweep(
    env_id: str,
    configs: Dict[str, Dict[str, float]],
    n_episodes: int = 3000,
    max_steps: int = 200,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Train multiple agents and return last-window rewards for comparison."""
    summary: Dict[str, List[float]] = {}

    for name, params in configs.items():
        env = make_env(env_id)
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            seed=seed,
            **params,
        )
        history = train_agent(env, agent, n_episodes=n_episodes, max_steps=max_steps, seed=seed)
        env.close()
        summary[name] = history["rewards"][-500:]
        print(f"[Sweep] {name}: mean last-500 reward = {np.mean(summary[name]):.2f}")

    return summary
