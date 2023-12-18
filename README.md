# Boax: A Bayesian Optimization library for JAX.

[**Overview**](#overview)
| [**Installation**](#installation)
| [**Getting Started**](#getting-started)
| [**Documentation**](https://boax.readthedocs.io/en/latest/)

*Boax is currently in early alpha and under active development!*

## Overview

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**. It comes with a low-level interfaces for:

* **Fitting a Gaussian Process model to data** (`boax.prediction`): Kernels, Mean Functions, Gaussian Processes
* **Constructing and optimizing acquisition functions** (`boax.optimization`) Acquisition Functions, Maximizers, Samplers

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
from functools import partial

from jax import config

# Double precision is highly recommended.
config.update("jax_enable_x64", True)

from jax import jit
from jax import lax
from jax import nn
from jax import numpy as jnp
from jax import random
from jax import scipy
from jax import value_and_grad
from jax import vmap

import optax
import matplotlib.pyplot as plt

from boax.prediction import kernels, means, models
from boax.optimization import acquisitions, maximizers, samplers

bounds = jnp.array([[-3, 3]])

def objective(x):
  return jnp.sin(4 * x[..., 0]) + jnp.cos(2 * x[..., 0])

sample_key, noise_key, maximizer_key = random.split(random.key(0), 3)
x_train = random.uniform(sample_key, minval=bounds[0, 0], maxval=bounds[0, 1], shape=(10, 1))
y_train = objective(x_train) + 0.3 * random.normal(noise_key, shape=(10,))
```

2. Fit a Gaussian Process model to the training dataset.

```python
def prior(amplitude, length_scale, noise):
  return models.gaussian_process(
    means.zero(),
    kernels.scale(nn.softplus(amplitude), kernels.rbf(nn.softplus(length_scale))),
    nn.softplus(noise),
  )

optimizer = optax.adam(0.01)

def target_log_prob(params):
  mean, cov = prior(**params)(x_train)
  return -scipy.stats.multivariate_normal.logpdf(y_train, mean, cov)

params = {
  'amplitude': jnp.zeros(()),
  'length_scale': jnp.zeros(()),
  'noise': jnp.array(-5.),
}

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

3. Construct and optimize an acquisition function
```python
surrogate = models.gaussian_process_regression(
  x_train,
  y_train,
  means.zero(),
  kernels.scale(nn.softplus(next_params['amplitude']), kernels.rbf(nn.softplus(next_params['length_scale']))),
  nn.softplus(next_params['noise']),
)

acqf = acquisitions.upper_confidence_bound(2.0, surrogate)
maximizer = maximizers.bfgs(bounds, q=1, num_restarts=25, num_raw_samples=500)

init_candidates = maximizer.init(maximizer_key, acqf)
candidates, values = maximizer.maximize(init_candidates, acqf)
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
