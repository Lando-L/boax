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

"""Optimization functions for experiments."""

import math
from collections.abc import Callable
from typing import Any

import optax
from jax import jit, lax, nn, random, value_and_grad, vmap
from jax import numpy as jnp

from boax.core import distributions, samplers
from boax.core.distributions import multivariate_normal, uniform
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.optimization import acquisitions, optimizers
from boax.core.prediction import models, objectives
from boax.core.prediction.models import Model
from boax.core.prediction.objectives import Objective
from boax.experiments.base import Step, Trial
from boax.utils.typing import Array, PRNGKey

SQRT2 = math.sqrt(2)
SQRT3 = math.sqrt(3)


def probability_of_improvement(
  key: PRNGKey,
  bounds: Array,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  improvement_factor: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
) -> Trial[list[tuple[dict[str, Any], float]]]:
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, 1))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep)
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, 1))

      acqf = models.transformations.transformed(
        vmap(
          models.transformations.transformed(
            model,
            outcome_transformation_fn=distributions.transformations.mvn_to_norm,
          )
        ),
        outcome_transformation_fn=acquisitions.log_probability_of_improvement(
          jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch_nonnegative(
          acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def expected_improvement(
  key: PRNGKey,
  bounds: Array,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  improvement_factor: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, 1))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep)
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, 1))

      acqf = models.transformations.transformed(
        vmap(
          models.transformations.transformed(
            model,
            outcome_transformation_fn=distributions.transformations.mvn_to_norm,
          )
        ),
        outcome_transformation_fn=acquisitions.log_expected_improvement(
          jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch_nonnegative(
          acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def upper_confidence_bound(
  key: PRNGKey,
  bounds: Array,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  confidence: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, 1))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep)
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, 1))

      acqf = models.transformations.transformed(
        vmap(
          models.transformations.transformed(
            model,
            outcome_transformation_fn=distributions.transformations.mvn_to_norm,
          )
        ),
        outcome_transformation_fn=acquisitions.upper_confidence_bound(
          confidence,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch(
          acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def batch_probability_of_improvement(
  key: PRNGKey,
  bounds: Array,
  batch_size: int,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  optimization_num_samples: int,
  improvement_factor: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, 1))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key, initializer_key, solver_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep), 4
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, 1))

      initializer_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              initializer_key,
              (optimization_num_samples, optimization_num_results, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_probability_of_improvement(
          best=jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      solver_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              solver_key,
              (optimization_num_samples, optimization_num_restarts, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_probability_of_improvement(
          best=jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch_nonnegative(
          initializer_acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(solver_acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def batch_expected_improvement(
  key: PRNGKey,
  bounds: Array,
  batch_size: int,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  optimization_num_samples: int,
  improvement_factor: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, 1))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key, initializer_key, solver_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep), 4
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, 1))

      initializer_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              initializer_key,
              (optimization_num_samples, optimization_num_results, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_expected_improvement(
          best=jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      solver_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              solver_key,
              (optimization_num_samples, optimization_num_restarts, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_expected_improvement(
          best=jnp.max(jnp.array(scores)) * improvement_factor,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch_nonnegative(
          initializer_acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(solver_acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def batch_upper_confidence_bound(
  key: PRNGKey,
  bounds: Array,
  batch_size: int,
  num_warm_up_steps: int,
  fit_learning_rate: float,
  fit_num_steps: int,
  optimization_num_results: int,
  optimization_num_restarts: int,
  optimization_num_samples: int,
  confidence: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  warm_up_key, acqf_optimization_key, mean_optimization_key = random.split(
    key, 3
  )

  samples = samplers.halton_uniform(
    uniform.uniform(bounds[:, 0], bounds[:, 1])
  )(warm_up_key, (num_warm_up_steps, batch_size))

  surrogate_fn = _single_task_gp(
    bounds,
    fit_learning_rate,
    fit_num_steps,
  )

  def next(
    step: Step[list[tuple[dict[str, Any], float]]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[list[tuple[dict[str, Any], float]]], list[dict[str, Any]]]:
    next_step = Step(
      0 if not step else step.timestep + 1,
      results if not step else step.state + results,
    )

    if next_step.timestep < num_warm_up_steps:
      parameterizations = list(
        map(to_parameterization, samples[next_step.timestep])
      )

      return next_step, parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_step.state)))

      model = surrogate_fn(
        jnp.array(list(map(from_parameterization, parameterizations))),
        jnp.array(scores),
      )

      candidate_key, optimizer_key, initializer_key, solver_key = random.split(
        random.fold_in(acqf_optimization_key, next_step.timestep), 4
      )

      candidates = samplers.halton_uniform(
        uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(candidate_key, (optimization_num_results, batch_size))

      initializer_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              initializer_key,
              (optimization_num_samples, optimization_num_results, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_upper_confidence_bound(
          confidence,
        ),
      )

      solver_acqf = models.transformations.transformed(
        models.transformations.sampled(
          vmap(model),
          vmap(multivariate_normal.sample),
          jnp.squeeze(
            samplers.halton_normal()(
              solver_key,
              (optimization_num_samples, optimization_num_restarts, batch_size),
            )
          ),
        ),
        outcome_transformation_fn=acquisitions.q_upper_confidence_bound(
          confidence,
        ),
      )

      optimizer = optimizers.batch(
        optimizers.initializers.q_batch(
          initializer_acqf, candidates, optimization_num_restarts
        ),
        optimizers.solvers.scipy(solver_acqf, bounds),
      )

      values, _ = optimizer(optimizer_key)

      parameterizations = list(map(to_parameterization, values))

      return next_step, parameterizations

  def best(
    step: Step[list[tuple[dict[str, Any], float]]],
  ) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state)))

    model = surrogate_fn(
      jnp.array(list(map(from_parameterization, parameterizations))),
      jnp.array(scores),
    )

    candidate_key, optimizer_key = random.split(
      random.fold_in(mean_optimization_key, step.timestep)
    )

    candidates = samplers.halton_uniform(
      uniform.uniform(bounds[:, 0], bounds[:, 1])
    )(candidate_key, (optimization_num_results, 1))

    acqf = models.transformations.transformed(
      vmap(
        models.transformations.transformed(
          model,
          outcome_transformation_fn=distributions.transformations.mvn_to_norm,
        )
      ),
      outcome_transformation_fn=acquisitions.posterior_mean(),
    )

    optimizer = optimizers.batch(
      optimizers.initializers.q_batch(
        acqf, candidates, optimization_num_restarts
      ),
      optimizers.solvers.scipy(acqf, bounds),
    )

    values, scores = optimizer(optimizer_key)

    return to_parameterization(values[0]), float(scores)

  return Trial(next, best)


def _single_task_gp(
  bounds: Array,
  fit_learning_rate: float,
  fit_num_steps: int,
) -> Callable[[Array, Array], Model[MultivariateNormal]]:
  optimizer = optax.adam(fit_learning_rate)

  priors = {
    'length_scale': distributions.normal.normal(
      SQRT2 + jnp.log(bounds.shape[0]) * 0.5,
      SQRT3,
    ),
    'noise': distributions.normal.normal(
      jnp.array(-4.0),
      jnp.array(1.0),
    ),
  }

  params = {
    'mean': jnp.zeros(()),
    'length_scale': jnp.repeat(priors['length_scale'].loc, bounds.shape[0]),
    'noise': priors['noise'].loc,
  }

  def model_fn(
    params: dict, x_train: Array | None, y_train: Array | None
  ) -> Model[MultivariateNormal]:
    return models.gaussian_process.exact(
      models.means.constant(params['mean']),
      models.kernels.rbf(jnp.exp(params['length_scale'])),
      models.likelihoods.gaussian(jnp.exp(params['noise'])),
      x_train,
      y_train,
    )

  def objective_fn(params: dict) -> Objective[MultivariateNormal]:
    return objectives.transformations.penalized(
      objectives.negative_log_likelihood(
        distributions.multivariate_normal.logpdf
      ),
      -jnp.sum(
        distributions.normal.logpdf(
          priors['length_scale'], jnp.exp(params['length_scale'])
        ),
      ),
      -jnp.sum(
        distributions.normal.logpdf(priors['noise'], jnp.exp(params['noise'])),
      ),
    )

  def loss_fn(params: dict, x_train: Array, y_train: Array) -> Array:
    y_hat = model_fn(params, None, None)(x_train)
    objective = objective_fn(params)
    return objective(y_hat, y_train)

  def fit_fn(x_train: Array, y_train: Array) -> dict:
    def step(state, i):
      loss, grads = value_and_grad(loss_fn)(state[0], x_train, y_train)
      updates, opt_state = optimizer.update(grads, state[1])
      params = optax.apply_updates(state[0], updates)

      return (params, opt_state), loss

    (next_params, _), _ = lax.scan(
      jit(step),
      (params, optimizer.init(params)),
      jnp.arange(fit_num_steps),
    )

    return next_params

  def surrogate(x_train: Array, y_train: Array) -> Model[MultivariateNormal]:
    x_standardized = (x_train - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y_mean, y_var = (jnp.mean(y_train), jnp.var(y_train))
    y_normalized = nn.standardize(y_train, mean=y_mean, variance=y_var)

    params = fit_fn(x_standardized, y_normalized)
    model = model_fn(params, x_standardized, y_normalized)

    return models.transformations.transformed(
      model,
      input_transformation_fn=models.transformations.input.normalized(bounds),
      outcome_transformation_fn=models.transformations.outcome.scaled(
        y_mean, y_var, multivariate_normal.scale
      ),
    )

  return surrogate
