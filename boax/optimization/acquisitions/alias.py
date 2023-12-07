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

from jax import numpy as jnp
from jax import scipy

from boax.optimization.acquisitions.base import Acquisition
from boax.prediction.processes.base import Process
from boax.typing import Array, Numeric


def log_probability_of_improvement(
  best: Numeric, process: Process
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

  def acquisition(candidates: Array) -> Array:
    loc, cov = process(candidates)
    scale = jnp.sqrt(jnp.diag(cov))
    return scipy.stats.norm.logcdf(loc, loc=best, scale=scale)

  return acquisition


def log_expected_improvement(best: Numeric, process: Process) -> Acquisition:
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
    return jnp.where(-log2 < x, jnp.log(jnp.expm1(-x)), jnp.log1p(jnp.exp(-x)))

  def logerfcx(x):
    return jnp.log(jnp.exp(x**2) * scipy.special.erfc(x))

  def log_ei_upper(x):
    return jnp.log(scipy.stats.norm.pdf(x) + x * scipy.stats.norm.cdf(x))

  def log_ei_lower(x):
    return (
      -(x**2) / 2 - c1 + log1mexp(logerfcx(-x * inv_sqrt2) * jnp.abs(x) + c2)
    )

  def logh(x):
    return jnp.where(x > -1, log_ei_upper(x), log_ei_lower(x))

  def acquisition(candidates: Array) -> Array:
    loc, cov = process(candidates)
    scale = jnp.sqrt(jnp.diag(cov))
    return logh((loc - best) / scale) + jnp.log(scale)

  return acquisition


def upper_confidence_bound(beta: Numeric, process: Process) -> Acquisition:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Args:
    beta: The mean and covariance trade-off parameter.
    process: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(candidates: Array) -> Array:
    loc, cov = process(candidates)
    scale = jnp.sqrt(jnp.diag(cov))
    return loc + jnp.sqrt(beta) * scale

  return acquisition
