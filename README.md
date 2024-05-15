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
  * Model Functions
  * Objective Functions
* **Constructing and optimizing acquisition functions** (`boax.optimization`):
  * Acquisition Functions
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

Here is a basic example of using the Boax API for defining a Gaussian Process model, constructing an Acquisition function, and generating the next batch of data points to query. For more details check out the [docs](https://boax.readthedocs.io/en/latest/).

1. Defining a Gaussian Process model:

```python
from boax.prediction import models

model = models.gaussian_process.exact(
  models.means.zero(),
  models.kernels.scaled(
    models.kernels.rbf(1.0), 0.5
  ),
  models.likelihoods.gaussian(1e-4),
  x_train,
  y_train,
)
```

2. Constructing an Acquisition function.

```python
from jax import vmap
from boax.optimization import acquisitions

acqf = models.outcome_transformed(
  vmap(model),
  acquisitions.upper_confidence_bound(2.0)
)
```

3. Generating the next batch of data points to query.

```python
from jax import numpy as jnp
from jax import random
from boax.core import distributions, samplers
from boax.optimization import optimizers

key = random.key(0)

batch_size, num_results, num_restarts = 1, 100, 10
bounds = jnp.array([[-1.0, 1.0]])

sampler = samplers.halton_uniform(
  distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
)

optimizer = optimizers.batch(
  optimizers.initializers.q_batch(
    acqf, sampler, batch_size, num_results, num_restarts,
  ),
  optimizers.solvers.scipy(
    acqf, bounds,  
  ),
)

next_x, value = optimizer(key)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.1.3},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
