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
  best: Numeric, posterior: Model[Tuple[Array, Array]]
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

  return jit(
    compose(
      scipy.stats.norm.cdf,
      tupled(partial(functions.analytic.scaled_improvement, best=best)),
      tupled(functions.analytic.analytic),
      posterior,
    )
  )


def log_probability_of_improvement(
  best: Numeric, posterior: Model[Tuple[Array, Array]]
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

  return jit(
    compose(
      scipy.stats.norm.logcdf,
      tupled(partial(functions.analytic.scaled_improvement, best=best)),
      tupled(functions.analytic.analytic),
      posterior,
    )
  )


def expected_improvement(
  best: Numeric, posterior: Model[Tuple[Array, Array]]
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

  return jit(
    compose(
      tupled(partial(functions.analytic.ei, best=best)),
      tupled(functions.analytic.analytic),
      posterior,
    )
  )


def log_expected_improvement(
  best: Numeric, posterior: Model[Tuple[Array, Array]]
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

  return jit(
    compose(
      tupled(partial(functions.analytic.lei, best=best)),
      tupled(functions.analytic.analytic),
      posterior,
    )
  )


def upper_confidence_bound(
  beta: Numeric, posterior: Model[Tuple[Array, Array]]
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

  return jit(
    compose(
      tupled(partial(functions.analytic.ucb, beta=beta)),
      tupled(functions.analytic.analytic),
      posterior,
    )
  )


def posterior_mean(
  posterior: Model[Tuple[Array, Array]],
) -> Acquisition:
  """
  The Posterior mean acquisition function.

  Args:
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(itemgetter(0), tupled(functions.analytic.analytic), posterior)
  )


def posterior_scale(
  posterior: Model[Tuple[Array, Array]],
) -> Acquisition:
  """
  The Posterior scale acquisition function.

  Args:
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """

  return jit(
    compose(itemgetter(1), tupled(functions.analytic.analytic), posterior)
  )


def q_expected_improvement(
  best: Numeric,
  base_samples: Array,
  posterior: Model[Tuple[Array, Array]],
) -> Acquisition:
  """
  The q Expected Improvement acquisition function.

  Args:
    best: The best function value observed so far.
    base_samples: A gaussian posterior.
    posterior: A gaussian posterior.

  Returns:
    The corresponding `Acquisition`.
  """
  
  return jit(
    compose(
      partial(functions.monte_carlo.qei, best=best),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      posterior
    )
  )


def q_probability_of_improvement(
  best: Numeric,
  tau: Numeric,
  base_samples: Array,
  posterior: Model[Tuple[Array, Array]],
) -> Acquisition:
  
  return jit(
    compose(
      partial(functions.monte_carlo.qpoi, best=best, tau=tau),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      posterior
    )
  )


def q_upper_confidence_bound(
  beta: Numeric,
  base_samples: Array,
  posterior: Model[Tuple[Array, Array]],
) -> Acquisition:
  
  beta_prime = lax.sqrt(beta * math.pi / 2)

  return jit(
    compose(
      partial(functions.monte_carlo.qucb, beta=beta_prime),
      tupled(partial(functions.monte_carlo.sampler, base_samples=base_samples)),
      posterior
    )
  )
