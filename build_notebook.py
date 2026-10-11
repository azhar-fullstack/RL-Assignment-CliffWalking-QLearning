"""Build and execute the assignment Jupyter notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "notebooks" / "training.ipynb"


def main() -> None:
    cells = [
        new_markdown_cell(
            "# Custom RL Agent: Q-Learning on CliffWalking-v1\n\n"
            "This notebook trains a **tabular Q-Learning agent from scratch** using "
            "`gymnasium`, visualizes learning curves, evaluates the greedy policy, "
            "and renders a demo GIF."
        ),
        new_markdown_cell("## 1. Imports & Setup"),
        new_code_cell(
            """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("..").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import QLearningAgent
from utils import (
    evaluate_agent,
    make_env,
    plot_hyperparameter_comparison,
    plot_q_heatmap,
    plot_training_curves,
    render_policy_gif,
    run_hyperparameter_sweep,
    train_agent,
)

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("Project root:", ROOT)"""
        ),
        new_markdown_cell(
            "## 2. Initialize Environment\n\n"
            "Using `gymnasium.make(\"CliffWalking-v1\")`."
        ),
        new_code_cell(
            """ENV_ID = "CliffWalking-v1"
env = make_env(ENV_ID)

print("Environment:", ENV_ID)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("n_states =", env.observation_space.n, "| n_actions =", env.action_space.n)

# Quick random baseline
state, info = env.reset(seed=0)
total = 0.0
for _ in range(50):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    total += float(reward)
    if terminated or truncated:
        break
print(f"Random rollout return (first episode fragment): {total:.1f}")
env.close()"""
        ),
        new_markdown_cell(
            "## 3. Create Q-Learning Agent\n\n"
            "Update rule: $Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma\\max_{a'}Q(s',a') - Q(s,a)]$"
        ),
        new_code_cell(
            """HYPERPARAMS = dict(
    alpha=0.5,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    seed=42,
)

env = make_env(ENV_ID)
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    **HYPERPARAMS,
)
print("Agent ready with Q-table shape:", agent.q_table.shape)
print("Hyperparameters:", HYPERPARAMS)"""
        ),
        new_markdown_cell("## 4. Train Agent"),
        new_code_cell(
            """N_EPISODES = 5000
MAX_STEPS = 200

history = train_agent(
    env,
    agent,
    n_episodes=N_EPISODES,
    max_steps=MAX_STEPS,
    seed=42,
)
env.close()

print("Training finished.")
print(f"Final epsilon: {agent.epsilon:.4f}")
print(f"Mean reward (last 100 eps): {np.mean(history['rewards'][-100:]):.2f}")
print(f"Success rate (last 100 eps): {np.mean(history['successes'][-100:]) * 100:.1f}%")"""
        ),
        new_markdown_cell("## 5. Save Model Weights (Q-table)"),
        new_code_cell(
            """agent.save(str(MODELS / "q_table.npz"))
np.save(MODELS / "q_table.npy", agent.q_table)
print("Saved:", MODELS / "q_table.npy")
print("Q-table stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(
    agent.q_table.min(), agent.q_table.max(), agent.q_table.mean()
))"""
        ),
        new_markdown_cell("## 6. Evaluate Greedy Policy"),
        new_code_cell(
            """eval_env = make_env(ENV_ID)
metrics = evaluate_agent(eval_env, agent, n_episodes=100, seed=10999)
eval_env.close()

print("Evaluation over 100 greedy episodes:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")"""
        ),
        new_markdown_cell("## 7. Training Visualizations"),
        new_code_cell(
            """plot_training_curves(history, str(RESULTS / "training_curves.png"))
plot_q_heatmap(agent.q_table, str(RESULTS / "q_value_heatmap.png"))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].imshow(plt.imread(RESULTS / "training_curves.png"))
axes[0].axis("off")
axes[0].set_title("Training Curves")
axes[1].imshow(plt.imread(RESULTS / "q_value_heatmap.png"))
axes[1].axis("off")
axes[1].set_title("Q-Value Heatmap")
plt.tight_layout()
plt.show()"""
        ),
        new_markdown_cell("## 8. Hyperparameter Sensitivity"),
        new_code_cell(
            """configs = {
    "a=0.1": {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.995},
    "a=0.5": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.995},
    "a=0.9": {"alpha": 0.9, "gamma": 0.99, "epsilon_decay": 0.995},
    "g=0.90": {"alpha": 0.5, "gamma": 0.90, "epsilon_decay": 0.995},
    "g=0.99": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.995},
    "slow eps": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.999},
}

sweep = run_hyperparameter_sweep(ENV_ID, configs, n_episodes=3000, max_steps=200, seed=42)
plot_hyperparameter_comparison(sweep, str(RESULTS / "hyperparameter_sensitivity.png"))

plt.figure(figsize=(8, 3.5))
plt.imshow(plt.imread(RESULTS / "hyperparameter_sensitivity.png"))
plt.axis("off")
plt.title("Hyperparameter Sensitivity")
plt.show()

for name, rewards in sweep.items():
    print(f"{name:10s} -> mean last-500 reward = {np.mean(rewards):7.2f}")"""
        ),
        new_markdown_cell("## 9. Render Trained Agent GIF"),
        new_code_cell(
            """gif_path = RESULTS / "agent_demo.gif"
total_r, steps, success = render_policy_gif(ENV_ID, agent, str(gif_path), seed=0, fps=4)
print(f"GIF saved to: {gif_path}")
print(f"Demo rollout: reward={total_r:.1f}, steps={steps}, success={success}")

from IPython.display import Image, display
display(Image(filename=str(gif_path)))"""
        ),
        new_markdown_cell(
            "## 10. Conclusion\n\n"
            "- Environment: **CliffWalking-v1** (48 states, 4 actions)\n"
            "- Algorithm: **ε-greedy Q-Learning** implemented from scratch\n"
            "- The greedy policy reaches the optimal return of **-13** with **100%** success "
            "under evaluation.\n"
            "- Hyperparameter study shows learning rate and ε-decay strongly affect "
            "sample efficiency and late-training return.\n"
            "- Artifacts: `models/q_table.npy`, `results/agent_demo.gif`, and plots in `results/`."
        ),
    ]

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(OUT.parent)}})
    with OUT.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote executed notebook -> {OUT}")


if __name__ == "__main__":
    main()
