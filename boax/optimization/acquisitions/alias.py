# Copyright 2023 The Boax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Alias for acquisition functions."""

import math
from functools import partial
from operator import attrgetter
from typing import Callable

from jax import jit, lax, vmap
from jax import numpy as jnp

from boax.core import distributions
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.optimization.acquisitions import functions
from boax.optimization.acquisitions.base import Acquisition
from boax.prediction.models.base import Model
from boax.utils.functools import call, compose
from boax.utils.typing import Array, Numeric


def probability_of_improvement(
  model: Model[MultivariateNormal], best: Numeric
) -> Acquisition:
  """
  The Probability of Improvement acquisition function.

  Probability of improvement over the best function value observed so far.

  `PI(x) = P(y >= best_f), y ~ f(x)`

  Example:
    >>> acqf = probability_of_improvement(surrogate, 0.2)
    >>> poi = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
      compose(
        jnp.squeeze,
        partial(functions.analytic.poi, best=best),
        vmap(
          compose(
            distributions.multivariate_normal.multivariate_to_normal,
            model,
          )
        )
      )
    )


def log_probability_of_improvement(
  model: Model[MultivariateNormal], best: Numeric
) -> Acquisition:
  """
  The Log Probability of Improvement acquisition function.

  Logarithm of the probability of improvement over the best function value observed so far.

  `logPI(x) = log(P(y >= best_f)), y ~ f(x)`

  Example:
    >>> acqf = log_probability_of_improvement(surrogate, 0.2)
    >>> log_poi = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      partial(functions.analytic.lpoi, best=best),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def expected_improvement(
  model: Model[MultivariateNormal], best: Numeric
) -> Acquisition:
  """
  The Expected Improvement acquisition function.

  Expected improvement over the best function value observed so far.

  `EI(x) = E(max(f(x) - best_f, 0))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  Example:
    >>> acqf = expected_improvement(surrogate, 0.2)
    >>> ei = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      partial(functions.analytic.ei, best=best),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def log_expected_improvement(
  model: Model[MultivariateNormal], best: Numeric
) -> Acquisition:
  """
  The Log Expected Improvement acquisition function.

  Logarithm of the expected improvement over the best function value observed so far.

  `LogEI(x) = log(E(max(f(x) - best_f, 0)))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  References:
    Ament, Sebastian, et al. "Unexpected improvements to expected improvement for bayesian optimization."
    arXiv preprint arXiv:2310.20708 (2023).

  Example:
    >>> acqf = log_expected_improvement(surrogate, 0.2)
    >>> log_ei = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      partial(functions.analytic.lei, best=best),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def upper_confidence_bound(
  model: Model[MultivariateNormal], beta: Numeric
) -> Acquisition:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Example:
    >>> acqf = upper_confidence_bound(surrogate, 2.0)
    >>> ucb = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.
    beta: The mean and covariance trade-off parameter.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      partial(functions.analytic.ucb, beta=beta),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def posterior_mean(
  model: Model[MultivariateNormal],
) -> Acquisition:
  """
  The Posterior mean acquisition function.

  Example:
    >>> acqf = posterior_mean(surrogate)
    >>> mean = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      attrgetter('loc'),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def posterior_scale(
  model: Model[MultivariateNormal],
) -> Acquisition:
  """
  The Posterior scale acquisition function.

  Example:
    >>> acqf = posterior_scale(surrogate)
    >>> scale = acqf(xs)

  Args:
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      jnp.squeeze,
      attrgetter('scale'),
      vmap(
        compose(
          distributions.multivariate_normal.multivariate_to_normal,
          model,
        )
      )
    )
  )


def q_expected_improvement(
  model: Model[Callable[[Array], Array]],
  base_samples: Array,
  best: Numeric,
) -> Acquisition:
  """
  MC-based batch Expected Improvement acquisition function.

  This computes qEI by (1) sampling the joint posterior over q points
  (2) evaluating the improvement over the current best for each sample
  (3) maximizing over q (4) averaging over the samples.

  `qEI(x) = E(max(max y - best_f, 0)), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_expected_improvement(surrogate, base_samples, 0.2)
    >>> qei = acqf(xs)

  Args:
    model: A surrogate model.
    base_samples: A set of samples from standard normal distribution.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    vmap(
      compose(
        jnp.mean,
        partial(jnp.amax, axis=-1),
        partial(functions.monte_carlo.qei, best=best),
        call(base_samples),
        vmap,
        model,
      )
    )
  )


def q_probability_of_improvement(
  model: Model[Callable[[Array], Array]],
  base_samples: Array,
  tau: Numeric,
  best: Numeric,
) -> Acquisition:
  """
  MC-based batch Probability of Improvement acquisition function.

  Estimates the probability of improvement over the current best observed value
  by sampling from the joint posterior distribution of the q-batch. MC-based
  estimates of a probability involves taking expectation of an indicator function;
  to support auto-differentiation, the indicator is replaced with a sigmoid
  function with temperature parameter tau.

  `qPI(x) = P(max y >= best_f), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_probability_of_improvement(surrogate, base_samples, 1.0, 0.2)
    >>> qpoi = acqf(xs)

  Args:
    model: A surrogate model.
    base_samples: A set of samples from standard normal distribution.
    tau: The temperature parameter.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    vmap(
      compose(
        jnp.mean,
        partial(jnp.amax, axis=-1),
        partial(functions.monte_carlo.qpoi, best=best, tau=tau),
        call(base_samples),
        vmap,
        model,
      )
    )
  )


def q_upper_confidence_bound(
  model: Model[Callable[[Array], Array]],
  base_samples: Array,
  beta: Numeric,
) -> Acquisition:
  """
  MC-based batch Upper Confidence Bound acquisition function.

  `qUCB = E(max(mean + |y_tilde - mean|))`,

  where `y_tilde ~ N(mean(x), beta * pi/2 * cov(x))` and `f(x) ~ N(mean(x), cov(x)).`

  Example:
    >>> acqf = q_upper_confidence_bound(surrogate, base_samples, 2.0)
    >>> qucb = acqf(xs)

  Args:
    model: A surrogate model.
    base_samples: A set of samples from standard normal distribution.
    beta: The mean and covariance trade-off parameter.

  Returns:
    The corresponding `Acquisition`.
  """

  beta_prime = lax.sqrt(beta * math.pi / 2)

  return jit(
    vmap(
      compose(
        jnp.mean,
        partial(jnp.amax, axis=-1),
        partial(functions.monte_carlo.qucb, beta=beta_prime),
        call(base_samples),
        vmap,
        model,
      )
    )
  )
