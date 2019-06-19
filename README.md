# Maximum Entropy Inverse Reinforcement Learning

This is a python implementation of the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) algorithm based on the similarly named paper by Ziebart et al. and the Maximum Causal Entropy Inverse Reinforcement Learning (MaxCausalEnt IRL) algorithm based on his PhD thesis.
Project for the Advanced Seminar in Imitation Learning, summer term 2019, University of Stuttgart.

For an example demonstrating how the Maximum (non-causal) Entropy IRL algorithm works, see the corresponding Jupyter notebook (`notebooks/maxent.ipynb`).
Note that the provided python files (`src/`) contain a slightly more optimized implementation of the algorithms.

To run a demonstration without the notebook, you can directly run `src/main.py`.
Also have a look at this file on how to use the provided framework.
The framework contains:
- Two GridWorld implementations for demonstration (`src/gridworld.py`)
- The algorithm implementations (`src/maxent.py`)
- A gradient based optimizer framework (`src/optimizer.py`)
- Plotting helper functions (`src/plot.py`)
- A MDP solver framework, i.e. value iteration and corresponding utilities (`src/solver.py`)
- A trajectory/trajectory generation framework (`src/trajectory.py`)

This project solely relies on the following dependencies: `numpy`, `matplotlib`, `itertools`, and `pytest`.
