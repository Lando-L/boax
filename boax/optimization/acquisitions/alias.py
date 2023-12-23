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
from operator import itemgetter
from typing import Tuple

from jax import jit, lax, scipy

from boax.optimization.acquisitions import functions
from boax.optimization.acquisitions.base import Acquisition
from boax.prediction.models.base import Model
from boax.typing import Array, Numeric
from boax.util import compose, tupled


def probability_of_improvement(
  best: Numeric, model: Model[Tuple[Array, Array]]
) -> Acquisition:
  """
  The Probability of Improvement acquisition function.

  Probability of improvement over the best function value observed so far.

  `PI(x) = P(y >= best_f), y ~ f(x)`

  Example:
    >>> acqf = probability_of_improvement(0.2, surrogate)
    >>> poi = acqf(xs)

  Args:
    best: The best function value observed so far.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      scipy.stats.norm.cdf,
      tupled(partial(functions.analytic.scaled_improvement, best=best)),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def log_probability_of_improvement(
  best: Numeric, model: Model[Tuple[Array, Array]]
) -> Acquisition:
  """
  The Log Probability of Improvement acquisition function.

  Logarithm of the probability of improvement over the best function value observed so far.

  `logPI(x) = log(P(y >= best_f)), y ~ f(x)`

  Example:
    >>> acqf = log_probability_of_improvement(0.2, surrogate)
    >>> log_poi = acqf(xs)

  Args:
    best: The best function value observed so far.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      scipy.stats.norm.logcdf,
      tupled(partial(functions.analytic.scaled_improvement, best=best)),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def expected_improvement(
  best: Numeric, model: Model[Tuple[Array, Array]]
) -> Acquisition:
  """
  The Expected Improvement acquisition function.

  Expected improvement over the best function value observed so far.

  `EI(x) = E(max(f(x) - best_f, 0))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  Example:
    >>> acqf = expected_improvement(0.2, surrogate)
    >>> ei = acqf(xs)

  Args:
    best: The best function value observed so far.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      tupled(partial(functions.analytic.ei, best=best)),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def log_expected_improvement(
  best: Numeric, model: Model[Tuple[Array, Array]]
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
    >>> acqf = log_expected_improvement(0.2, surrogate)
    >>> log_ei = acqf(xs)

  Args:
    best: The best function value observed so far.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      tupled(partial(functions.analytic.lei, best=best)),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def upper_confidence_bound(
  beta: Numeric, model: Model[Tuple[Array, Array]]
) -> Acquisition:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Example:
    >>> acqf = upper_confidence_bound(2.0, surrogate)
    >>> ucb = acqf(xs)

  Args:
    beta: The mean and covariance trade-off parameter.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      tupled(partial(functions.analytic.ucb, beta=beta)),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def posterior_mean(
  model: Model[Tuple[Array, Array]],
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
      itemgetter(0),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def posterior_scale(
  model: Model[Tuple[Array, Array]],
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
      itemgetter(1),
      tupled(functions.analytic.analytic),
      model,
    )
  )


def q_expected_improvement(
  best: Numeric,
  base_samples: Array,
  model: Model[Tuple[Array, Array]],
) -> Acquisition:
  """
  MC-based batch Expected Improvement acquisition function.

  This computes qEI by (1) sampling the joint posterior over q points
  (2) evaluating the improvement over the current best for each sample
  (3) maximizing over q (4) averaging over the samples.

  `qEI(x) = E(max(max y - best_f, 0)), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_expected_improvement(0.2, base_samples, surrogate)
    >>> qei = acqf(xs)

  Args:
    best: The best function value observed so far.
    base_samples: A set of samples from standard normal distribution.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(functions.monte_carlo.qei, best=best),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      model,
    )
  )


def q_probability_of_improvement(
  best: Numeric,
  tau: Numeric,
  base_samples: Array,
  model: Model[Tuple[Array, Array]],
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
    >>> acqf = q_probability_of_improvement(0.2, 1.0, base_samples, surrogate)
    >>> qpoi = acqf(xs)

  Args:
    best: The best function value observed so far.
    base_samples: A set of samples from standard normal distribution.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(
      partial(functions.monte_carlo.qpoi, best=best, tau=tau),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      model,
    )
  )


def q_upper_confidence_bound(
  beta: Numeric,
  base_samples: Array,
  model: Model[Tuple[Array, Array]],
) -> Acquisition:
  """
  MC-based batch Upper Confidence Bound acquisition function.

  `qUCB = E(max(mean + |y_tilde - mean|))`,

  where `y_tilde ~ N(mean(x), beta * pi/2 * cov(x))` and `f(x) ~ N(mean(x), cov(x)).`

  Example:
    >>> acqf = q_upper_confidence_bound(2.0, base_samples, surrogate)
    >>> qucb = acqf(xs)

  Args:
    beta: The mean and covariance trade-off parameter.
    base_samples: A set of samples from standard normal distribution.
    model: A gaussian process regression surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """
  beta_prime = lax.sqrt(beta * math.pi / 2)

  return jit(
    compose(
      partial(functions.monte_carlo.qucb, beta=beta_prime),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      model,
    )
  )
