"""Train, evaluate, and visualize a Q-Learning agent on CliffWalking-v1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom RL agent — CliffWalking Q-Learning")
    parser.add_argument("--env-id", type=str, default="CliffWalking-v1")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sweep", action="store_true", help="Skip hyperparameter sweep")
    parser.add_argument("--eval-episodes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Custom RL Agent - Q-Learning on", args.env_id)
    print("=" * 60)

    # 1) Initialize environment
    env = make_env(args.env_id)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")

    # 2) Initialize agent
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    # 3) Train
    history = train_agent(
        env,
        agent,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    env.close()

    # 4) Save model
    model_path = MODELS_DIR / "q_table.npz"
    agent.save(str(model_path))
    # Also save plain .npy for assignment checklist compatibility
    np.save(MODELS_DIR / "q_table.npy", agent.q_table)
    print(f"Saved Q-table → {model_path}")

    # 5) Evaluate
    eval_env = make_env(args.env_id)
    metrics = evaluate_agent(eval_env, agent, n_episodes=args.eval_episodes, seed=args.seed + 999)
    eval_env.close()
    print(
        f"Evaluation ({args.eval_episodes} eps): "
        f"mean={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}, "
        f"success={metrics['success_rate'] * 100:.1f}%"
    )

    # 6) Plots
    plot_training_curves(history, str(RESULTS_DIR / "training_curves.png"))
    plot_q_heatmap(agent.q_table, str(RESULTS_DIR / "q_value_heatmap.png"))
    print(f"Saved plots → {RESULTS_DIR}")

    # 7) GIF of trained policy
    gif_path = RESULTS_DIR / "agent_demo.gif"
    total_r, steps, success = render_policy_gif(args.env_id, agent, str(gif_path), seed=0)
    print(f"Saved GIF → {gif_path} | reward={total_r:.1f}, steps={steps}, success={success}")

    # 8) Hyperparameter sensitivity study
    if not args.skip_sweep:
        print("\nRunning hyperparameter sensitivity sweep...")
        configs = {
            "a=0.1": {"alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.995},
            "a=0.5": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.995},
            "a=0.9": {"alpha": 0.9, "gamma": 0.99, "epsilon_decay": 0.995},
            "g=0.90": {"alpha": 0.5, "gamma": 0.90, "epsilon_decay": 0.995},
            "g=0.99": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.995},
            "slow eps": {"alpha": 0.5, "gamma": 0.99, "epsilon_decay": 0.999},
        }
        sweep = run_hyperparameter_sweep(
            args.env_id,
            configs,
            n_episodes=min(3000, args.episodes),
            max_steps=args.max_steps,
            seed=args.seed,
        )
        plot_hyperparameter_comparison(
            sweep,
            str(RESULTS_DIR / "hyperparameter_sensitivity.png"),
        )
        print(f"Saved sensitivity plot → {RESULTS_DIR / 'hyperparameter_sensitivity.png'}")

    print("\nDone. All deliverables written under models/ and results/.")


if __name__ == "__main__":
    main()
