# Boax: A Bayesian Optimization library for JAX.

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**. It comes with a low-level interfaces for:

* **Fitting a Gaussian Process model to data** (`boax.prediction`): Bijectors, Kernels, Mean Functions, Gaussian Processes
* **Constructing and optimizing acquisition functions** (`boax.optimization`) Acquisition Functions, Maximizers, Search Spaces

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
For more details check out the [docs](https://boax.readthedocs.io/en/latest/) and our [tutorials](https://github.com/Lando-L/boax/blob/main/examples).

1. Create a dataset from a noisy sinusoid.

```python
from jax import config

# Double precision is highly recommended.
config.update("jax_enable_x64", True)

from functools import partial

from jax import jit
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import scipy
from jax import value_and_grad
from jax import vmap

import optax
import matplotlib.pyplot as plt

from boax.prediction import bijectors, kernels, means, processes
from boax.optimization import acquisitions, maximizers

bounds = jnp.array([[-3, 3]])

def objective(x):
  return jnp.sin(4 * x[..., 0]) + jnp.cos(2 * x[..., 0])

sample_key, noise_key = random.split(random.key(0))
x_train = random.uniform(sample_key, minval=bounds[0, 0], maxval=bounds[0, 1], shape=(10, 1))
y_train = objective(x_train) + 0.3 * random.normal(noise_key, shape=(10,))
```

2. Fit a Gaussian Process model to the training dataset.

```python
bijector = bijectors.softplus

def process(params):
  return processes.gaussian(
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
    loc, scale = process(params).prior(x_train)
    return -scipy.stats.multivariate_normal.logpdf(y_train, loc, scale)

  loss, grads = value_and_grad(loss_fn)(state[0])
  updates, opt_state = optimizer.update(grads, state[1])
  params = optax.apply_updates(state[0], updates)

  return (params, opt_state), loss

(next_params, next_opt_state), history = lax.scan(
  jit(train_step),
  (params, opt_state),
  jnp.arange(500)
)
```

3. Construct and optimize an acquisition function
```python
acqusition = acquisitions.upper_confidence_bound(
    2,
    partial(process(next_params).posterior, observation_index_points=x_train, observations=y_train)
)

candidates, scores = maximizers.bfgs(50, bounds)(acqusition)
```

## Citing Boax

To cite Boax please use the citation:

```bibtex
@software{boax2023github,
  author = {Lando L{\"o}per},
  title = {{B}oax: A Bayesian Optimization library for {JAX}},
  url = {https://github.com/Lando-L/boax},
  version = {0.0.1},
  year = {2023},
}
```

In the above bibtex entry, the version number
is intended to be that from [boax/version.py](https://github.com/Lando-L/boax/blob/main/boax/version.py), and the year corresponds to the project's open-source release.
