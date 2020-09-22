# Bandito :slot_machine:
A fairly simple package playing with Stochastic Multi-Armed Bandits :slot_machine:


## Install
The package was developed on `Python 3.8.0` and the required packages can be find in [requirements](./requirements) folder.
In order to install `bandito` from GitHub run:

```shell
pip install git+https://github.com/matejker/bandito.git@master  # install the latest [maybe not stable] version
pip install git+https://github.com/matejker/bandito.git@v0.1.0  # install specific version
```

## Lint, tests and typechecking
In this repo we use a few tools to keep code clean, styled and properly tested:
```shell
make lint  # runs flake8 and Black check
make autoformat  # runs Black formating
make typecheck  # runs mypy
make test  # runs unit [py]tests
```
