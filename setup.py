import setuptools

long_description = """
# Maximum Entropy Inverse Reinforcement Learning

This is a python implementation of the Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) algorithm based on the similarly named paper by Ziebart et al. and the Maximum Causal Entropy Inverse Reinforcement Learning (MaxCausalEnt IRL) algorithm based on his PhD thesis.

You may also want to have a look at the accompanying [presentation][presentation].

For an example demonstrating how the Maximum (non-causal) Entropy IRL algorithm works, see the corresponding Jupyter notebook ([`notebooks/maxent.ipynb`][nb-viewer]).
Note that this python package contains a slightly more optimized implementation of the algorithms.

For an example on how to use this framework, have a look at the [`example.py`][example] file.
The framework contains:
- Two GridWorld implementations for demonstration (`irl_maxent.gridworld`)
- The algorithm implementations (`irl_maxent.maxent`)
- A gradient based optimizer framework (`irl_maxent.optimizer`)
- Plotting helper functions (`irl_maxent.plot`)
- A MDP solver framework, i.e. value iteration and corresponding utilities (`irl_maxent.solver`)
- A trajectory/trajectory generation framework (`irl_maxent.trajectory`)

[nb-viewer]: https://nbviewer.jupyter.org/github/qzed/irl-maxent/blob/master/notebooks/maxent.ipynb
[presentation]: https://nbviewer.jupyter.org/github/qzed/irl-maxent/blob/master/Presentation.pdf
[example]: https://github.com/qzed/irl-maxent/blob/master/src/example.py

"""

setuptools.setup(
    name="irl-maxent",
    version="0.1.0",
    author="Maximilian Luz",
    author_email="luzmaximilian@gmail.com",
    description="A small package for Maximum Entropy Inverse Reinforcement Learning on simple MDPs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qzed/irl-maxent",
    project_urls={
        "Bug Tracker": "https://github.com/qzed/irl-maxent/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
