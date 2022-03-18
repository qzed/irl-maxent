# Maximum Entropy Inverse Reinforcement Learning

This is a python implementation of the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) algorithm based on the similarly named paper by Ziebart et al. and the Maximum Causal Entropy Inverse Reinforcement Learning (MaxCausalEnt IRL) algorithm based on his PhD thesis.
Project for the Advanced Seminar in Imitation Learning, summer term 2019, University of Stuttgart.

This implementation is available as python package at https://pypi.org/project/irl-maxent/ and can be installed via `pip install irl-maxent`.
You may also want to have a look at the accompanying [presentation][presentation].

For an example demonstrating how the Maximum (non-causal) Entropy IRL algorithm works, see the corresponding Jupyter notebook ([`notebooks/maxent.ipynb`][nb-viewer]).
Note that the provided python files (`src/`) contain a slightly more optimized implementation of the algorithms.

To run a demonstration without the notebook, you can directly run `./src/example.py`.
Also have a look at this file on how to use the provided framework.
The framework contains:
- Two GridWorld implementations for demonstration (`irl_maxent.gridworld`)
- The algorithm implementations (`irl_maxent.maxent`)
- A gradient based optimizer framework (`irl_maxent.optimizer`)
- Plotting helper functions (`irl_maxent.plot`)
- A MDP solver framework, i.e. value iteration and corresponding utilities (`src/solver.py`)
- A trajectory/trajectory generation framework (`irl_maxent.trajectory`)

This project solely relies on the following dependencies: `numpy`, `matplotlib`, `itertools`, and `pytest`.

[nb-viewer]: https://nbviewer.jupyter.org/github/qzed/irl-maxent/blob/master/notebooks/maxent.ipynb
[presentation]: https://nbviewer.jupyter.org/github/qzed/irl-maxent/blob/master/Presentation.pdf
