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

Here is a basic example of using the Boax API for defining a Gaussian Process model, constructing an Acquisition function, and combining the two for optimzation. For more details check out the [docs](https://boax.readthedocs.io/en/latest/).

1. Defining a Gaussian Process model:

```python
from boax.prediction import models

model = models.gaussian_process(
  models.means.zero(),
  models.kernels.scaled(
    models.kernels.rbf(1.0),
    0.5
  ),
  models.likelihoods.gaussian(1e-4),
  x_train,
  y_train,
)
```

2. Constructing an Acquisition function.

```python
from boax.optimization import acquisitions

acquisition = acquisitions.upper_confidence_bound(
  beta=2.0
)
```

3. Combining the two for optimization

```python
from jax import jit, random, vmap
from jax import numpy as jnp
from boax.optimization import optimizers

def acqf(x):
  return acquisition(vmap(model)(x))

key1, key2 = random.split(random.key(0))
bounds = jnp.array([[-1.0, 1.0]])
x0 = random.uniform(key1, shape=(100, 1, 1))
bfgs = optimizers.bfgs(jit(acqf), bounds, x0, 10)
candidates = bfgs.init(key2)
next_candidates, values = bfgs.update(candidates)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.1.2},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
