# Boax: A Bayesian Optimization library for JAX.

![tests](https://github.com/Lando-L/boax/actions/workflows/tests.yml/badge.svg?branch=main)
![pypi](https://img.shields.io/pypi/v/boax)

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**.

It comes with high-level interfaces for:
* **Experiments** (`boax.experiments`):
  * Bayesian Optimization Setups
  * Bandit Optimization Setups
  * Search Spaces

And with low-level interfaces for:
* **Constructing acquisition functions** (`boax.acquisition`):
  * Acquisition Functions
  * Surrogate Models
* **Constructing policy functions** (`boax.policies`):
  * Policy Functions
  * Believes
* **Core capabilities** (`boax.core`):
  * Common Distributions
  * Gaussian Process Models
  * Objective Functions
  * Quasi-Newton Optimizers
  * Monte-Carlo Samplers

## Installation

You can install the latest released version of Boax from PyPI via:

```sh
pip install boax
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/Lando-L/boax.git
```

## Basic Usage

Here is a basic example of using the Boax for hyperparamter tuning.
For more details check out the [docs](https://boax.readthedocs.io/en/latest/).

1. Setting up classification task:

```python
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.svm import SVC

  iris = load_iris()
  X = iris.data
  y = iris.target

  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  def evaluate(C, gamma):
    svc = SVC(C=C, gamma=gamma, kernel='rbf')
    svc.fit(X_train, y_train)
    return svc.score(X_test, y_test)
```

2. Setting up a bayesian optimization experiment.

```python
  from jax import config
  config.update("jax_enable_x64", True)
  from boax.experiments import optimization

  experiment = optimization(
    parameters=[
      {
        'name': 'C',
        'type': 'log_range',
        'bounds': [1, 1_000],
      },
      {
        'name': 'gamma',
        'type': 'log_range',
        'bounds': [1e-4, 1e-3],
      },
    ],
    batch_size=4,
  )
```

3. Running the trial for N = 25 steps.

```python
  step, results = None, []

  for _ in range(25):
    # Retrieve next parameterizations to evaluate
    step, parameterizations = experiment.next(step, results)

    # Evaluate parameterizations
    evaluations = [
      evaluate(**parameterization)
      for parameterization in parameterizations
    ]
    
    results = list(
        zip(parameterizations, evaluations)
    )

  # Predicted best
  experiment.best(step)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.2.0},
  year = {2023},
}
```

In the above bibtex entry, the version number is intended to be the latest, and the year corresponds to the project's open-source release.
