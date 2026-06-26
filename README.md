# CSCN8020 — Assignment 2: Q-Learning on the Taxi Gymnasium Environment

## Student Information

| Field | Value |
|---|---|
| **Full Name** | *Anthony Izevbokun* |
| **Student ID** | *9016626* |
| **Course** | CSCN8020 — Reinforcement Learning Programming |

---

## Project Summary

This project implements the **Q-Learning** algorithm (Sutton & Barto, 2018) applied to the **Taxi-v4** Gymnasium environment. A taxi agent learns to navigate a 5×5 grid, pick up a passenger from one of four locations, and deliver them to the correct destination. The implementation uses a full object-oriented architecture, trains the agent over 10,000 episodes, reports training metrics with plots, runs hyperparameter experiments (varying learning rate α and exploration factor ε), and identifies the best-performing hyperparameter combination.

---

## Repository Link

```
https://github.com/AntUse/CSCN8020_Assignment2
```

**Cloneable URL:**
```
https://github.com/AntUse/CSCN8020_Assignment2.git
```

---

## How to Run the Notebook

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AntUse/CSCN8020_Assignment2.git
   cd CSCN8020_Assignment2
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter and open the notebook:**
   ```bash
   jupyter notebook CSCN8020_Assignment2.ipynb
   ```

5. **Run all cells top to bottom** using **Kernel → Restart & Run All**.

> The notebook runs in approximately 2–5 minutes depending on hardware.

---

## Repository File Structure

```
CSCN8020_Assignment2/
├── CSCN8020_Assignment2.ipynb   # Main Jupyter Notebook (full solution)
├── training.log                  # Log file generated during training
├── plots_baseline.png            # Baseline training metrics plots
├── plots_lr_comparison.png       # Learning rate experiment comparison
├── plots_eps_comparison.png      # Exploration factor experiment comparison
├── plots_best_combination.png    # Best combination training metrics
├── plots_best_vs_baseline.png    # Best combination vs baseline comparison
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

---

## Q-Learning Implementation Overview

The implementation follows the **Tabular Q-Learning** algorithm (Sutton & Barto, 2018, p. 131) using six OOP classes:

| Class | Responsibility |
|---|---|
| `TaxiEnvironmentManager` | Wraps Gymnasium Taxi-v4; handles reset, step, state decoding |
| `QLearningAgent` | Owns Q-table; implements ε-greedy selection and TD update rule |
| `MetricsLogger` | Records per-episode rewards/steps; writes `training.log` |
| `PlotManager` | Generates and saves all training metric plots |
| `QLearningTrainer` | Runs the episode training loop and greedy policy evaluation |
| `ExperimentRunner` | Orchestrates multiple experiments; produces comparison tables |

**Q-Learning update rule:**

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

**Baseline hyperparameters:** α = 0.1, ε = 0.1, γ = 0.9, 10,000 episodes.

---

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Gymnasium Documentation. https://gymnasium.farama.org/
