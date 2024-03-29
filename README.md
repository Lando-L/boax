# Boax: A Bayesian Optimization library for JAX.

![tests](https://github.com/Lando-L/boax/actions/workflows/tests.yml/badge.svg?branch=main)
![pypi](https://img.shields.io/pypi/v/boax)

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization that is **designed for flexibility**. It comes with low-level interfaces for:

* **Core capabilities** (`boax.core`):
  * Common Distributions
  * Monte-Carlo Samplers
* **Fitting a surrogate model to data** (`boax.prediction`):
  * Kernels Functions
  * Likelihood Functions
  * Mean Functions
  * Model Functions
  * Objective Functions
* **Constructing and optimizing acquisition functions** (`boax.optimization`):
  * Acquisition Functions
  * Constraint Functions
  * Optimizer Functions

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

Here is a basic example of using the Boax API for Bayesian Optimization. For more details check out the [docs](https://boax.readthedocs.io/en/latest/).

1. Creation of a prediction model.

```python
def model(params):
  return models.outcome_transformed(
    models.gaussian_process_regression(
      means.zero(),
      kernels.rbf(params['length_scale']),
    )(
      x_train,
      y_train,
    ),
    likelihoods.gaussian(params['noise']),
  )
```

2. Construction of an acquisition function.

```python
def construct(model):
  return optimization.construct(
    models.outcome_transformed(
      model,
      distributions.multivariate_normal.as_normal,
    ),
    acquisitions.upper_confidence_bound(
      beta=2.0
    ),
  )
```

3. Selection of the next candidate to query.

```python
def select(key, acqf):
    bfgs = optimizers.bfgs(acqf, bounds, x0, 10)
    candidates = bfgs.init(key)
    next_candidates, values = bfgs.update(candidates)

    next_x = next_candidates[jnp.argmax(values)]
    next_y = objective(next_x)

    return next_x, next_y
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.1.1},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
