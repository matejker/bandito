# Bandito :slot_machine:
A package playing with Stochastic Multi-Armed Bandits (MAB) :slot_machine: In last decade bandits became widely used 
machine learning algorithm.

## Examples
In following [jupyter notebooks](./notebooks) we tried demonstrate different types of arms, bandit policies and types of 
Stochastic Multi-Armed Bandits.
 - [Non-adaptive policies](./notebooks/01_simple_non_adaptive_algorithms.ipynb) - Uniform exploration and Epsilon-Greedy policy
 - [Adaptive policies](./notebooks/02_adaptive_algorithms.ipynb) - Successive Elimination and UCB1 policy 


## Install
The package was developed on `Typed Python 3.8.0` and the required packages can be find in [requirements](./requirements) folder.
In order to install `bandito` from GitHub, run:

```shell
pip install git+https://github.com/matejker/bandito.git@master  # install the latest [maybe not stable] version
pip install git+https://github.com/matejker/bandito.git@v0.1.0  # install specific version
```

## Lint, tests and typechecking
In this repo we use a few tools to keep the code clean, styled and properly tested:
```shell
make lint  # runs flake8 and Black check
make autoformat  # runs Black formating
make typecheck  # runs mypy
make test  # runs unit [py]tests
```

## References
[1] Slivkins A. (2019), *Introduction to Multi-Armed Bandits*, arXiv:1904.07272, https://arxiv.org/abs/1904.07272
