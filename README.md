# Boax: A Bayesian Optimization library for JAX.

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**. It comes with a low-level interfaces for:

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

## Getting Started

Here is a quick start example of the two main compoments that form the Bayesian optimization loop.
For more details check out the [docs](https://boax.readthedocs.io/en/latest/).

1. Create a synthetic dataset.

```python
from jax import config

# Double precision is highly recommended.
config.update("jax_enable_x64", True)

from jax import jit
from jax import lax
from jax import nn
from jax import numpy as jnp
from jax import random
from jax import value_and_grad

import optax

from boax import prediction, optimization
from boax.core import distributions, samplers
from boax.prediction import kernels, likelihoods, means, models, objectives
from boax.optimization import acquisitions, optimizers

bounds = jnp.array([[0.0, 1.0]])

def objective(x):
  return 1 - jnp.linalg.norm(x - 0.5)

data_key, sampler_key, optimizer_key = random.split(random.key(0), 3)

x_train = random.uniform(
  random.fold_in(data_key, 0),
  minval=bounds[:, 0],
  maxval=bounds[:, 1],
  shape=(10, 1)
)

y_train = objective(x_train) + 0.1 * random.normal(
  random.fold_in(data_key, 1),
  shape=(10,)
)
```

2. Fit a Gaussian Process surrogate model to the training dataset.

```python
params = {
  'amplitude': jnp.zeros(()),
  'length_scale': jnp.zeros(()),
  'noise': jnp.zeros(()),
}

adam = optax.adam(0.01)

def fit(x_train, y_train):
  def model(params):
    return models.outcome_transformed(
      models.gaussian_process(
        means.zero(),
        kernels.scaled(
          kernels.rbf(params['amplitude']),
          params['length_scale'],
        ),
      ),
      likelihoods.gaussian(params['noise']),
    )

  def objective(params):
    return objectives.negative_log_likelihood(
      distributions.multivariate_normal.logpdf
    )

  def projection(params):
    return {
      'amplitude': nn.softplus(params['amplitude']),
      'length_scale': nn.softplus(params['length_scale']),
      'noise': nn.softplus(params['noise']) + 1e-4,
    }

  def step(state, iteration):
    loss_fn = prediction.construct(model, objective, projection)
    loss, grads = value_and_grad(loss_fn)(state[0], x_train, y_train)
    updates, opt_state = adam.update(grads, state[1])
    params = optax.apply_updates(state[0], updates)
    
    return (params, opt_state), loss
  
  (next_params, _), _ = lax.scan(
    jit(step),
    (params, adam.init(params)),
    jnp.arange(500)
  )

  return projection(next_params)
```

3. Construct and optimize an UCB acquisition function.
```python
x0 = jnp.reshape(
  samplers.halton_uniform(
    distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(
    sampler_key,
    100,
  ),
  (100, 1, -1)
)

def optimize(x_train, y_train):
  def model(params):
    return models.outcome_transformed(
      models.gaussian_process_regression(
        means.zero(),
        kernels.scaled(
          kernels.rbf(params['amplitude']),
          params['length_scale']
        )
      )(
        x_train,
        y_train,
      ),
      likelihoods.gaussian(params['noise']),
      distributions.multivariate_normal.as_normal,
    )

  for i in range(10):
    params = fit(x_train, y_train)

    acqf = optimization.construct(
        model(params),
        acquisitions.upper_confidence_bound(2.0),
    )
    
    bfgs = optimizers.bfgs(acqf, bounds, x0, 10)
    candidates = bfgs.init(random.fold_in(optimizer_key, i))
    next_candidates, values = bfgs.update(candidates)

    next_x = next_candidates[jnp.argmax(values)]
    next_y = objective(next_x)
    
    x_train = jnp.vstack([x_train, next_x])
    y_train = jnp.hstack([y_train, next_y])

  return x_train, y_train

next_x_train, next_y_train = optimize(x_train, y_train)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.0.4},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
