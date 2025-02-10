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
from typing import TypeVar

from jax import numpy as jnp
from jax import random, scipy

from boax.acquisitions import functions
from boax.acquisitions.base import Acquisition
from boax.core import distributions, models, optimizers, samplers
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.models import Model
from boax.core.samplers import Sampler
from boax.utils.math import fatmax, logmeanexp
from boax.utils.typing import Array, Numeric, PRNGKey, Shape

T = TypeVar('T')


def probability_of_improvement(
  bounds: Array,
  improvement_factor: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition[MultivariateNormal]:
  """
  The Probability of Improvement acquisition function.

  Probability of improvement over the best function value observed so far.

  `PI(x) = P(y >= best), y ~ f(x)`

  Example:
    >>> acqf = probability_of_improvement(0.2, model)
    >>> poi = acqf(xs)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(sampler_key, (num_raw_samples, 1))(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
    )

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        partial(functions.analytic.mean_and_sigma),
        partial(functions.analytic.scaled_improvement, best=best),
        scipy.stats.norm.cdf,
        partial(jnp.squeeze, axis=-1),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def log_probability_of_improvement(
  bounds: Array,
  improvement_factor: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  The Log Probability of Improvement acquisition function.

  Logarithm of the probability of improvement over the best function value observed so far.

  `logPI(x) = log(P(y >= best)), y ~ f(x)`

  Example:
    >>> acqf = log_probability_of_improvement(0.2, model)
    >>> log_poi = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(sampler_key, (num_raw_samples, 1))(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
    )

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        partial(functions.analytic.mean_and_sigma),
        partial(functions.analytic.scaled_improvement, best=best),
        scipy.stats.norm.logcdf,
        partial(jnp.squeeze, axis=-1),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def expected_improvement(
  bounds: Array,
  improvement_factor: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  The Expected Improvement acquisition function.

  Expected improvement over the best function value observed so far.

  `EI(x) = E(max(f(x) - best, 0))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  Example:
    >>> acqf = expected_improvement(0.2, model)
    >>> ei = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(sampler_key, (num_raw_samples, 1))(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
    )

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        partial(functions.analytic.mean_and_sigma),
        partial(functions.analytic.ei, best=best),
        partial(jnp.squeeze, axis=-1),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def log_expected_improvement(
  bounds: Array,
  improvement_factor: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  The Log Expected Improvement acquisition function.

  Logarithm of the expected improvement over the best function value observed so far.

  `LogEI(x) = log(E(max(f(x) - best, 0)))`,

  where the expectation is taken over the value of stochastic function `f` at `x`.

  References:
    Ament, Sebastian, et al. "Unexpected improvements to expected improvement for bayesian optimization."
    arXiv preprint arXiv:2310.20708 (2023).

  Example:
    >>> acqf = log_expected_improvement(0.2, model)
    >>> log_ei = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(sampler_key, (num_raw_samples, 1))(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
    )

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        partial(functions.analytic.mean_and_sigma),
        partial(functions.analytic.lei, best=best),
        partial(jnp.squeeze, axis=-1),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def upper_confidence_bound(
  bounds: Array,
  beta: Numeric = 2.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  The Upper Confidence Bound (UCB) acquisition function.

  Upper confidence bound comprises of the posterior mean plus an additional term:
  the posterior standard deviation weighted by a trade-off parameter, `beta`.

  `UCB(x) = loc(x) + sqrt(beta) * scale(x)`

  Example:
    >>> acqf = upper_confidence_bound(2.0, model)
    >>> ucb = acqf(index_points)

  Args:
    beta: The mean and covariance trade-off parameter.
    model: The surrogate model.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(sampler_key, (num_raw_samples, 1))(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
    )

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        partial(functions.analytic.mean_and_sigma),
        partial(functions.analytic.ucb, beta=beta),
        partial(jnp.squeeze, axis=-1),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def q_probability_of_improvement(
  bounds: Array,
  batch_size: int,
  sampler: Sampler[T],
  sample_axis: Shape = (0,),
  improvement_factor: Numeric = 1.0,
  tau: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition[T]:
  """
  MC-based batch Probability of Improvement acquisition function.

  Estimates the probability of improvement over the current best observed value
  by sampling from the joint posterior distribution of the q-batch. MC-based
  estimates of a probability involves taking expectation of an indicator function;
  to support auto-differentiation, the indicator is replaced with a sigmoid
  function with temperature parameter tau.

  `qPI(x) = P(max y >= best), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_probability_of_improvement(1.0, model, sampler)
    >>> qpoi = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.
    sampler: The posterior sampler.
    tau: The temperature parameter.
    sample_axis: The sample axis.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(
      sampler_key, (num_raw_samples, batch_size)
    )(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        sampler,
        partial(functions.monte_carlo.qpoi, best=best, tau=tau),
        partial(jnp.amax, axis=-1),
        partial(jnp.mean, axis=sample_axis),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def q_expected_improvement(
  bounds: Array,
  batch_size: int,
  sampler: Sampler[T],
  sample_axis: Shape = (0,),
  improvement_factor: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  MC-based batch Expected Improvement acquisition function.

  This computes qEI by
  (1) sampling the joint posterior over q points
  (2) evaluating the improvement over the current best for each sample
  (3) maximizing over q
  (4) averaging over the samples.

  `qEI(x) = E(max(max y - best, 0)), y ~ f(x), x = (x_1,...,x_q)`

  Example:
    >>> acqf = q_expected_improvement(1.0, model, sampler)
    >>> qei = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.
    sampler: The posterior sampler.
    sample_axis: The sample axis.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(
      sampler_key, (num_raw_samples, batch_size)
    )(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        sampler,
        partial(functions.monte_carlo.qei, best=best),
        partial(jnp.amax, axis=-1),
        partial(jnp.mean, axis=sample_axis),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def q_log_expected_improvement(
  bounds: Array,
  batch_size: int,
  sampler: Sampler[T],
  sample_axis: Shape = (0,),
  improvement_factor: Numeric = 1.0,
  tau: Numeric = 1.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  MC-based batch Log Expected Improvement acquisition function.

  This computes qLogEI by
  (1) sampling the joint posterior over q points,
  (2) evaluating the smoothed log improvement over the current best for each sample,
  (3) smoothly maximizing over q, and
  (4) averaging over the samples in log space.

  `qLogEI(X) ~ log(qEI(X)) = log(E(max(max Y - best_f, 0)))`,

  where `Y ~ f(X)`, and `X = (x_1,...,x_q)`.

  Example:
    >>> acqf = q_log_expected_improvement(1.0, model, sampler)
    >>> qlei = acqf(index_points)

  Args:
    best: The best function value observed so far.
    model: The surrogate model.
    sampler: The posterior sampler.
    sample_axis: The sample axis.

  Returns:
    The corresponding `Acquisition`.
  """

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(
      sampler_key, (num_raw_samples, batch_size)
    )(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))

    best = jnp.max(observations) * improvement_factor

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        sampler,
        partial(functions.monte_carlo.qlei, best=best, tau=1e-6),
        partial(fatmax, axis=-1, tau=1e-2),
        partial(logmeanexp, axis=sample_axis),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition


def q_upper_confidence_bound(
  bounds: Array,
  batch_size: int,
  sampler: Sampler[T],
  sample_axis: Shape = (0,),
  beta: Numeric = 2.0,
  num_raw_samples: int = 512,
  num_restarts: int = 10,
) -> Acquisition:
  """
  MC-based batch Upper Confidence Bound acquisition function.

  `qUCB = E(max(mean + |y_tilde - mean|))`,

  where `y_tilde ~ N(mean(x), beta * pi/2 * cov(x))` and `f(x) ~ N(mean(x), cov(x)).`

  Example:
    >>> acqf = q_upper_confidence_bound(2.0, model, sampler)
    >>> qucb = acqf(index_points)

  Args:
    beta: The mean and covariance trade-off parameter.
    model: The surrogate model.
    sampler: The posterior sampler.
    sample_axis: The sample axis.

  Returns:
    The corresponding `Acquisition`.
  """

  beta_prime = jnp.sqrt(beta * math.pi / 2)

  def acquisition(
    key: PRNGKey,
    model: Model[MultivariateNormal],
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(
      sampler_key, (num_raw_samples, batch_size)
    )(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=models.transformations.utils.chained(
        sampler,
        partial(functions.monte_carlo.qucb, beta=beta_prime),
        partial(jnp.amax, axis=-1),
        partial(jnp.mean, axis=sample_axis),
      ),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    return optimizer(fun, optimizer_key)

  return acquisition
