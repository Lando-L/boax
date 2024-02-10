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
  * Surrogate Models
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

1. Create a dataset from a noisy sinusoid.

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

from boax.core import distributions, samplers
from boax.prediction import kernels, likelihoods, means, models
from boax.optimization import acquisitions, optimizers

bounds = jnp.array([[0.0, 1.0]])

def objective(x):
  return 1 - jnp.linalg.norm(x - 0.5)

data_key, sampler_key, maximizer_key = random.split(random.key(0), 3)
x_train = random.uniform(random.fold_in(data_key, 0), minval=bounds[:, 0], maxval=bounds[:, 1], shape=(10, 1))
y_train = nn.standardize(objective(x_train) + 0.1 * random.normal(random.fold_in(data_key, 1), shape=(10,)))
```

2. Fit a Gaussian Process surrogate model to the training dataset.

```python
params = {
  'amplitude': jnp.zeros(()),
  'length_scale': jnp.zeros(()),
  'noise': jnp.array(-5.),
}

adam = optax.adam(0.01)

def fit(x_train, y_train):
  def model(amplitude, length_scale, noise):
    return models.predictive(
      models.gaussian_process(
        means.zero(),
        kernels.scaled(
          kernels.rbf(nn.softplus(length_scale)),
          nn.softplus(amplitude)
        ),
      ),
      likelihoods.gaussian(nn.softplus(noise))
    )
  
  def target_log_prob(params):
    mvn = model(**params)(x_train)
    return -jnp.sum(distributions.multivariate_normal.logpdf(mvn, y_train))

  def train_step(state, iteration):
    loss, grads = value_and_grad(target_log_prob)(state[0])
    updates, opt_state = adam.update(grads, state[1])
    params = optax.apply_updates(state[0], updates)
    
    return (params, opt_state), loss
  
  return lax.scan(
    jit(train_step),
    (params, adam.init(params)),
    jnp.arange(500)
  )

(next_params, next_opt_state), history = fit(x_train, y_train)

surrogate = models.predictive(
  models.gaussian_process_regression(
    means.zero(),
    kernels.scaled(
      kernels.rbf(nn.softplus(next_params['length_scale'])),
      nn.softplus(next_params['amplitude'])
    ),
  )(
    x_train,
    y_train,
  ),
  likelihoods.gaussian(nn.softplus(next_params['noise']))
)
```

3. Construct and optimize an UCB acquisition function.
```python
x0 = jnp.reshape(
  samplers.halton_uniform(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))(
    sampler_key,
    100,
  ),
  (100, 1, -1)
)

acqf = optimizers.construct(
  models.outcome_transformed(
    surrogate,
    distributions.multivariate_normal.as_normal
  ),
  acquisitions.upper_confidence_bound(2.0),
)

bfgs = optimizers.bfgs(acqf, bounds, x0, 10)
candidates = bfgs.init(maximizer_key)
next_candidates, values = bfgs.update(candidates)
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
