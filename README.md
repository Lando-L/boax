# Boax: A Bayesian Optimization library for JAX.

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://github.com/Lando-L/boax)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**. It comes with a low-level interfaces for:

* **Fitting a Gaussian Process model to data** (`boax.prediction`): Bijectors, Kernels, Mean Functions, Gaussian Processes
* **Constructing and optimizing acquisition functions** (`boax.optimization`) Acquisition Functions, Maximizers, Search Spaces

## Installation

You can install the latest released version of Boax from PyPI via:

```sh
pip install bayesian-optimization-jax
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/Lando-L/boax.git
```

## Getting Started

Here is a quick start example of the two main compoments that form the Bayesian optimization loop.
For more details check out the [docs](https://github.com/Lando-L/boax) and our [tutorials](https://github.com/Lando-L/boax/blob/main/examples).

1. Create a dataset from a noisy sinusoid.

```python
from jax import config

# Double precision is highly recommended.
config.update("jax_enable_x64", True)

from jax import numpy as jnp
from jax import random

sample_key, noise_key = random.split(random.key(0))

def objective(x):
  return jnp.sin(4 * x[..., 0]) + jnp.cos(2 * x[..., 0])

x_train = random.uniform(sample_key, minval=-3, maxval=3, shape=(10, 1))
y_train = objective(x_train) + 0.3 * random.normal(noise_key, shape=(10,))
```

2. Fit a Gaussian Process model to the training dataset.

```python
from jax import jit
from jax import lax
from jax import scipy
from jax import value_and_grad
from jax import vmap

import optax

from boax.prediction import bijectors, kernels, means
from boax.prediction.processes import gaussian

bijector = bijectors.softplus()

def prior(params):
  return gaussian.prior(
    vmap(means.zero()),
    vmap(vmap(kernels.scale(bijector.forward(params['amplitude']), kernels.rbf(bijector.forward(params['length_scale']))), in_axes=(None, 0)), in_axes=(0, None)),
    bijector.forward(params['noise']),
  )

params = {
  'amplitude': jnp.zeros(()),
  'length_scale': jnp.zeros(()),
  'noise': jnp.array(-5.),
}

optimizer = optax.adam(0.01)
opt_state = optimizer.init(params)

def train_step(state, iteration):
  def loss_fn(params):            
    loc, scale = prior(params)(x_train)
    return -scipy.stats.multivariate_normal.logpdf(y_train, loc, scale)

  loss, grads = value_and_grad(loss_fn)(state[0])
  updates, opt_state = optimizer.update(grads, state[1])
  params = optax.apply_updates(state[0], updates)

  return (params, opt_state), loss

(next_params, next_opt_state), history = lax.scan(jit(train_step), (params, opt_state), jnp.arange(500))
```

3. Construct and optimize an acquisition function
```python
from boax.optimization import acquisitions, maximizers, spaces

posterior = gaussian.posterior(
    x_train,
    y_train,
    vmap(means.zero()),
    vmap(vmap(kernels.scale(bijector.forward(next_params['amplitude']), kernels.rbf(bijector.forward(next_params['length_scale']))), in_axes=(None, 0)), in_axes=(0, None)),
    bijector.forward(next_params['noise']),
)

candidates, scores = maximizers.bfgs(50)(
  jit(acquisitions.upper_confidence_bound(2, posterior)),
  space.continous(jnp.array([[-3., 3.]]))
)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}ojax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.0.1},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
