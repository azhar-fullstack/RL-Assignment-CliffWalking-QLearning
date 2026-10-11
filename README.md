# Custom Reinforcement Learning Agent — CliffWalking-v1 (Q-Learning)

Hands-on implementation of **tabular Q-Learning from scratch** using [Farama Gymnasium](https://gymnasium.farama.org/).

## Environment

- **CliffWalking-v1**: 4×12 grid world
- **States**: 48 discrete cells
- **Actions**: Up, Right, Down, Left
- **Reward**: −1 per step, −100 for falling off the cliff, episode ends at the goal

## Project Structure

```
RL_Assignment/
├── main.py                 # CLI entry point (train / eval / GIF / plots)
├── agent.py                # QLearningAgent (from scratch)
├── utils.py                # env helpers, training, plotting, GIF
├── requirements.txt
├── notebooks/
│   └── training.ipynb      # Executed notebook with visualizations
├── models/
│   ├── q_table.npy         # Saved Q-table
│   └── q_table.npz         # Q-table + hyperparameters
├── results/
│   ├── training_curves.png
│   ├── q_value_heatmap.png
│   ├── hyperparameter_sensitivity.png
│   └── agent_demo.gif
└── report/
    └── Technical_Report.pdf
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train & Evaluate

```bash
# Full pipeline (training + eval + GIF + hyperparameter sweep)
python main.py

# Faster run (skip sensitivity sweep)
python main.py --skip-sweep --episodes 5000
```

## Notebook

Open `notebooks/training.ipynb` and run all cells to reproduce training curves and the demo GIF.

## Algorithm

ε-greedy **Q-Learning** update:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]
\]

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| α (learning rate) | 0.5 |
| γ (discount) | 0.99 |
| ε start | 1.0 |
| ε min | 0.01 |
| ε decay | 0.995 |
| Episodes | 5000 |
| Max steps | 200 |
| Seed | 42 |
