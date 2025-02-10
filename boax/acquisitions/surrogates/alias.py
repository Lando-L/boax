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

"""Alias for evaluator functions."""

from operator import attrgetter

import optax
from jax import jit, lax, nn, random, value_and_grad
from jax import numpy as jnp

from boax.acquisitions.surrogates.base import Surrogate
from boax.core import distributions, models, objectives, optimizers, samplers
from boax.core.distributions import multivariate_normal, normal
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.models import Model
from boax.core.objectives import Objective
from boax.utils.math import sqrt2, sqrt3
from boax.utils.typing import Array, PRNGKey


def single_task_gaussian_process(
  bounds: Array,
  fit_learning_rate: float = 0.01,
  fit_num_steps: int = 500,
  optimizer_num_raw_samples: int = 512,
  optimizer_num_restarts: int = 10,
) -> Surrogate[dict, MultivariateNormal]:
  """
  The single task gaussian process surrogate model.

  Has strict priors on its paramters.

  Example:
    >>> surrogate = single_task_gaussian_process(bounds, learning_rate, num_steps)
    >>> params = surrogate.init()
    >>> fitted_params = surrogate.update(params, observation_index_points, observations)
    >>> prior = surrogate.prior(params)
    >>> posterior = surrogate.update(params, observation_index_points, observations)
    >>> best = surrogate.best(posterior)

  Args:
    bounds: The domain bounds.
    learning_rate: The learning for fitting the parameters during parameter update.
    num_steps: The number of steps for fitting the parameters during parameter update.

  Returns;
    The corresponding `Surrogate`.
  """

  optimizer = optax.adam(fit_learning_rate)

  priors = {
    'length_scale': normal.normal(
      sqrt2 + jnp.log(bounds.shape[0]) * 0.5,
      sqrt3,
    ),
    'noise': normal.normal(
      jnp.array(-4.0),
      jnp.array(1.0),
    ),
  }

  def _model_fn(
    params: dict,
    observation_index_points: Array | None,
    observations: Array | None,
  ) -> Model[MultivariateNormal]:
    return models.gaussian_process.exact(
      models.means.constant(params['mean']),
      models.kernels.rbf(jnp.exp(params['length_scale'])),
      models.likelihoods.gaussian(jnp.exp(params['noise'])),
      observation_index_points,
      observations,
    )

  def _objective_fn(params: dict) -> Objective[MultivariateNormal]:
    return objectives.transformations.penalized(
      objectives.negative_log_likelihood(multivariate_normal.logpdf),
      -jnp.sum(
        normal.logpdf(priors['length_scale'], jnp.exp(params['length_scale'])),
      ),
      -jnp.sum(
        normal.logpdf(priors['noise'], jnp.exp(params['noise'])),
      ),
    )

  def _loss_fn(
    params: dict, observation_index_points: Array, observations: Array
  ) -> Array:
    y_hat = _model_fn(params, None, None)(observation_index_points)
    objective = _objective_fn(params)
    return objective(y_hat, observations)

  def init() -> dict:
    return {
      'mean': jnp.zeros(()),
      'length_scale': jnp.repeat(priors['length_scale'].loc, bounds.shape[0]),
      'noise': priors['noise'].loc,
    }

  def update(
    params: dict, observation_index_points: Array, observations: Array
  ) -> dict:
    def fit(state, i):
      loss, grads = value_and_grad(_loss_fn)(
        state[0], observation_index_points, observations
      )
      updates, opt_state = optimizer.update(grads, state[1])
      params = optax.apply_updates(state[0], updates)

      return (params, opt_state), loss

    (fitted_params, _), _ = lax.scan(
      jit(fit),
      (params, optimizer.init(params)),
      jnp.arange(fit_num_steps),
    )

    return fitted_params

  def prior(params: dict) -> Model[MultivariateNormal]:
    return models.transformations.transformed(
      _model_fn(params, None, None),
      input_transformation_fn=models.transformations.input.normalized(bounds),
    )

  def posterior(
    params: dict, observation_index_points: Array, observations: Array
  ) -> Model[MultivariateNormal]:
    index_points_normalized = (observation_index_points - bounds[:, 0]) / (
      bounds[:, 1] - bounds[:, 0]
    )
    observations_mean, observations_var = (
      jnp.mean(observations),
      jnp.var(observations),
    )
    observations_standardized = nn.standardize(
      observations, mean=observations_mean, variance=observations_var
    )

    return models.transformations.transformed(
      _model_fn(params, index_points_normalized, observations_standardized),
      input_transformation_fn=models.transformations.input.normalized(bounds),
      outcome_transformation_fn=models.transformations.outcome.scaled(
        observations_mean, observations_var, multivariate_normal.scale
      ),
    )

  def best(
    key: PRNGKey, model: Model[MultivariateNormal]
  ) -> tuple[Array, float]:
    sampler_key, optimizer_key = random.split(key)

    samples = samplers.halton_uniform(
      sampler_key, (optimizer_num_raw_samples,)
    )(
      distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]),
    )

    fun = models.transformations.transformed(
      model,
      outcome_transformation_fn=attrgetter('mean'),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(samples, optimizer_num_restarts),
      optimizers.solvers.scipy(bounds),
    )

    next_candidate, value = optimizer(fun, optimizer_key)

    return next_candidate, float(value)

  return Surrogate(init, update, prior, posterior, best)
