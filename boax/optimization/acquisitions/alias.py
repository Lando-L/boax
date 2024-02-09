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

from jax import jit, lax, scipy
from jax import numpy as jnp

from boax.core import distributions
from boax.core.distributions.normal import Normal
from boax.optimization.acquisitions import functions
from boax.optimization.acquisitions.base import Acquisition
from boax.utils.functools import compose
from boax.utils.typing import Array, Numeric


def probability_of_improvement(
  best: Numeric,
) -> Acquisition[Normal]:
  """
  The Probability of Improvement acquisition function.

  Probability of improvement over the best function value observed so far.

  `PI(x) = P(y >= best), y ~ f(x)`

  Example:
    >>> acqf = probability_of_improvement(0.2)
    >>> poi = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      scipy.stats.norm.cdf,
      partial(functions.analytic.scaled_improvement, best=best),
    )
  )


def log_probability_of_improvement(
  best: Numeric,
) -> Acquisition[Normal]:
  """
  The Log Probability of Improvement acquisition function.

  Logarithm of the probability of improvement over the best function value observed so far.

  `logPI(x) = log(P(y >= best)), y ~ f(x)`

  Example:
    >>> acqf = log_probability_of_improvement(0.2)
    >>> log_poi = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      scipy.stats.norm.logcdf,
      partial(functions.analytic.scaled_improvement, best=best),
    )
  )


def expected_improvement(
  best: Numeric,
) -> Acquisition[Normal]:
  """
  The Expected Improvement acquisition function.

  Expected improvement over the best function value observed so far.

  `EI(x) = E(max(f(x) - best, 0))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  Example:
    >>> acqf = expected_improvement(0.2)
    >>> ei = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(functions.analytic.ei, best=best),
    )
  )


def log_expected_improvement(
  best: Numeric,
) -> Acquisition[Normal]:
  """
  The Log Expected Improvement acquisition function.

  Logarithm of the expected improvement over the best function value observed so far.

  `LogEI(x) = log(E(max(f(x) - best, 0)))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  References:
    Ament, Sebastian, et al. "Unexpected improvements to expected improvement for bayesian optimization."
    arXiv preprint arXiv:2310.20708 (2023).

  Example:
    >>> acqf = log_expected_improvement(0.2)
    >>> log_ei = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(functions.analytic.lei, best=best),
    )
  )


def upper_confidence_bound(
  beta: Numeric,
) -> Acquisition[Normal]:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Example:
    >>> acqf = upper_confidence_bound(2.0)
    >>> ucb = acqf(model(xs))

  Args:
    beta: The mean and covariance trade-off parameter.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      partial(distributions.normal.sample, base_samples=jnp.sqrt(beta)),
    )
  )


def posterior_mean() -> Acquisition:
  """
  The Posterior mean acquisition function.

  Example:
    >>> acqf = posterior_mean()
    >>> mean = acqf(model(xs))

  Args:
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      attrgetter('loc'),
    )
  )


def posterior_scale() -> Acquisition:
  """
  The Posterior scale acquisition function.

  Example:
    >>> acqf = posterior_scale()
    >>> scale = acqf(model(xs))

  Args:
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.squeeze, axis=-1),
      attrgetter('scale'),
    )
  )


def q_probability_of_improvement(
  best: Numeric,
  tau: Numeric = 1.0,
) -> Acquisition[Array]:
  """
  MC-based batch Probability of Improvement acquisition function.

  Estimates the probability of improvement over the current best observed value
  by sampling from the joint posterior distribution of the q-batch. MC-based
  estimates of a probability involves taking expectation of an indicator function;
  to support auto-differentiation, the indicator is replaced with a sigmoid
  function with temperature parameter tau.

  `qPI(x) = P(max y >= best), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_probability_of_improvement(1.0, 0.2)
    >>> qpoi = acqf(model(xs))

  Args:
    tau: The temperature parameter.
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.mean, axis=-1),
      partial(jnp.amax, axis=-1),
      partial(functions.monte_carlo.qpoi, best=best, tau=tau),
    )
  )


def q_expected_improvement(
  best: Numeric,
) -> Acquisition[Array]:
  """
  MC-based batch Expected Improvement acquisition function.

  This computes qEI by (1) sampling the joint posterior over q points
  (2) evaluating the improvement over the current best for each sample
  (3) maximizing over q (4) averaging over the samples.

  `qEI(x) = E(max(max y - best, 0)), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_expected_improvement(0.2)
    >>> qei = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.mean, axis=-1),
      partial(jnp.amax, axis=-1),
      partial(functions.monte_carlo.qei, best=best),
    )
  )


def q_upper_confidence_bound(
  beta: Numeric,
) -> Acquisition[Array]:
  """
  MC-based batch Upper Confidence Bound acquisition function.

  `qUCB = E(max(mean + |y_tilde - mean|))`,

  where `y_tilde ~ N(mean(x), beta * pi/2 * cov(x))` and `f(x) ~ N(mean(x), cov(x)).`

  Example:
    >>> acqf = q_upper_confidence_bound(2.0)
    >>> qucb = acqf(model(xs))

  Args:
    beta: The mean and covariance trade-off parameter.

  Returns:
    The corresponding `Acquisition`.
  """

  beta_prime = lax.sqrt(beta * math.pi / 2)

  return jit(
    compose(
      partial(jnp.mean, axis=-1),
      partial(jnp.amax, axis=-1),
      partial(functions.monte_carlo.qucb, beta=beta_prime),
    )
  )


def q_knowledge_gradient(
  best: Numeric,
) -> Acquisition[Normal]:
  """
  MC-based batch Knowledge Gradient acquisition function.

  Example:
    >>> acqf = q_knowledge_gradient(0.2)
    >>> qucb = acqf(model(xs))

  Args:
    best: The best function value observed so far.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(jnp.mean, axis=-1),
      partial(jnp.squeeze, axis=-1),
      partial(lax.sub, y=best),
      attrgetter('loc'),
    )
  )
