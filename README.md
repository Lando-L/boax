# Boax: A Bayesian Optimization library for JAX.

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**. It comes with a low-level interfaces for:

* **Fitting a surrogate model to data** (`boax.prediction`):
  * Kernels
  * Mean Functions
  * Surrogate Models
  * Objectives
* **Constructing and optimizing acquisition functions** (`boax.optimization`):
  * Acquisition Functions
  * Constraint Functions
  * Maximizers

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

from boax.prediction import kernels, means, models, objectives
from boax.optimization import acquisitions, maximizers

bounds = jnp.array([[-3.0, 3.0]])

def objective(x):
  return jnp.sin(4 * x[..., 0]) + jnp.cos(2 * x[..., 0])

sample_key, noise_key, maximizer_key = random.split(random.key(0), 3)
x_train = random.uniform(sample_key, minval=bounds[:, 0], maxval=bounds[:, 1], shape=(10, 1))
y_train = objective(x_train) + 0.3 * random.normal(noise_key, shape=(10,))
```

2. Fit a Gaussian Process surrogate model to the training dataset.

```python
def prior(amplitude, length_scale, noise):
  return models.gaussian_process(
    means.zero(),
    kernels.scaled(kernels.rbf(nn.softplus(length_scale)), nn.softplus(amplitude)),
    nn.softplus(noise),
  )

def target_log_prob(params):
  y_hat = prior(**params)(x_train)
  return -objectives.exact_marginal_log_likelihood()(y_hat, y_train)

params = {
  'amplitude': jnp.zeros(()),
  'length_scale': jnp.zeros(()),
  'noise': jnp.array(-5.),
}

optimizer = optax.adam(0.01)

def train_step(state, iteration):
  loss, grads = value_and_grad(target_log_prob)(state[0])
  updates, opt_state = optimizer.update(grads, state[1])
  params = optax.apply_updates(state[0], updates)

  return (params, opt_state), loss

(next_params, next_opt_state), history = lax.scan(
  jit(train_step),
  (params, optimizer.init(params)),
  jnp.arange(500)
)
```

3. Construct and optimize an UCB acquisition function.
```python
def posterior(amplitude, length_scale, noise):
  return models.gaussian_process_regression(
    x_train,
    y_train,
    means.zero(),
    kernels.scaled(kernels.rbf(nn.softplus(length_scale)), nn.softplus(amplitude)),
    nn.softplus(noise),
  )

surrogate = posterior(**next_params)
acqf = acquisitions.upper_confidence_bound(surrogate, beta=2.0)
maximizer = maximizers.bfgs(acqf, bounds, q=1, num_restarts=25, num_raw_samples=500)

candidates, values = maximizer(maximizer_key)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.0.3},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
