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
from operator import itemgetter
from typing import Callable, Tuple

from jax import numpy as jnp
from jax import scipy

from boax.optimization.acquisitions.base import Acquisition
from boax.typing import Array, Numeric
from boax.util import compose, tupled


def analytic(
  posterior: Callable[[Array], Tuple[Array, Array]],
) -> Callable[[Array], Tuple[Array, Array]]:
  def acquisition(candidates: Array) -> Tuple[Array, Array]:
    loc, cov = posterior(candidates)
    scale = jnp.sqrt(jnp.diag(cov))
    return loc, scale

  return acquisition


def scaled_improvement(best: Numeric) -> Callable[[Array, Array], Array]:
  def acquisition(loc: Array, scale: Array) -> Array:
    return (loc - best) / scale

  return acquisition


def probability_of_improvement(
  best: Numeric, posterior: Callable[[Array], Tuple[Array, Array]]
) -> Acquisition:
  """
  The Probability of Improvement acquisition function.

  Probability of improvement over the best function value observed so far.

  `PI(x) = P(y >= best_f), y ~ f(x)`

  Args:
    best: The best function value observed so far.
    process: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return compose(
    scipy.stats.norm.cdf,
    tupled(scaled_improvement(best)),
    analytic(posterior),
  )


def log_probability_of_improvement(
  best: Numeric, posterior: Callable[[Array], Tuple[Array, Array]]
) -> Acquisition:
  """
  The Log Probability of Improvement acquisition function.

  Logarithm of the probability of improvement over the best function value observed so far.

  `logPI(x) = log(P(y >= best_f)), y ~ f(x)`

  Args:
    best: The best function value observed so far.
    process: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return compose(
    scipy.stats.norm.logcdf,
    tupled(scaled_improvement(best)),
    analytic(posterior),
  )


def expected_improvement(
  best: Numeric, posterior: Callable[[Array], Tuple[Array, Array]]
) -> Acquisition:
  """
  The Expected Improvement acquisition function.

  Expected improvement over the best function value observed so far.

  `EI(x) = E(max(f(x) - best_f, 0)),`

  where the expectation is taken over the value of stochastic function `f` at `x`.

  Args:
    best: The best function value observed so far.
    process: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  def ei(loc: Array, scale: Array) -> Array:
    u = scaled_improvement(best)(loc, scale)
    return (scipy.stats.norm.pdf(u) + u * scipy.stats.norm.cdf(u)) * scale

  return compose(tupled(ei), analytic(posterior))


def log_expected_improvement(
  best: Numeric, posterior: Callable[[Array], Tuple[Array, Array]]
) -> Acquisition:
  """
  The Log Expected Improvement acquisition function.

  Logarithm of the expected improvement over the best function value observed so far.

  `LogEI(x) = log(E(max(f(x) - best_f, 0))),`

  where the expectation is taken over the value of stochastic function `f` at `x`.

  References:
    Ament, Sebastian, et al. "Unexpected improvements to expected improvement for bayesian optimization."
    arXiv preprint arXiv:2310.20708 (2023).

  Args:
    best: The best function value observed so far.
    process: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  log2 = math.log(2)
  inv_sqrt2 = 1 / math.sqrt(2)
  c1 = math.log(2 * math.pi) / 2
  c2 = math.log(math.pi / 2) / 2

  def log1mexp(x):
    upper = jnp.where(x > -log2, x, -log2)
    lower = jnp.where(x <= -log2, x, -log2)

    return jnp.where(
      x > -log2,
      jnp.log(-jnp.expm1(upper)),
      jnp.log1p(-jnp.exp(lower))
    )

  def logerfcx(x):
    upper = jnp.where(x > 0., x, 0.)
    lower = jnp.where(x <= 0., x, 0.)

    return jnp.where(
      x > 0,
      jnp.log(jnp.exp(upper**2) * scipy.special.erfc(upper)),
      jnp.log(scipy.special.erfc(lower)) + lower**2,
    )

  def log_ei_upper(x):
    return jnp.log(scipy.stats.norm.pdf(x) + x * scipy.stats.norm.cdf(x))

  def log_ei_lower(x):
    return (
      -(x**2) / 2 - c1 + log1mexp(logerfcx(-x * inv_sqrt2) * jnp.abs(x) + c2)
    )

  def logh(x):
    upper = jnp.where(x > -1., x, -1.)
    lower = jnp.where(x <= -1., x, -1.)

    return jnp.where(
      x > -1,
      log_ei_upper(upper),
      log_ei_lower(lower)
    )

  def lei(loc: Array, scale: Array) -> Array:
    u = scaled_improvement(best)(loc, scale)
    return logh(u) + jnp.log(scale)

  return compose(tupled(lei), analytic(posterior))


def upper_confidence_bound(
  beta: Numeric, posterior: Callable[[Array], Tuple[Array, Array]]
) -> Acquisition:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Args:
    beta: The mean and covariance trade-off parameter.
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  def ucb(loc: Array, scale: Array) -> Array:
    return loc + jnp.sqrt(beta) * scale

  return compose(tupled(ucb), analytic(posterior))


def posterior_mean(
  posterior: Callable[[Array], Tuple[Array, Array]],
) -> Acquisition:
  """
  The Posterior mean acquisition function.

  Args:
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return compose(itemgetter(0), analytic(posterior))


def posterior_scale(
  posterior: Callable[[Array], Tuple[Array, Array]],
) -> Acquisition:
  """
  The Posterior scale acquisition function.

  Args:
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return compose(itemgetter(1), analytic(posterior))
