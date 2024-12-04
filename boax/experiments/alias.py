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

"""Alias for experiment functions."""

import math
from functools import partial
from typing import Any, Literal, overload

from jax import numpy as jnp
from jax import random

from boax.core.distributions.beta import Beta
from boax.core.optimization.policies.believes import ActionValues
from boax.experiments import functions, search_spaces
from boax.experiments.base import Trial


@overload
def optimization(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['probability_of_improvement'],
  seed: int = 0,
  num_warm_up_steps: int = 12,
  batch_size: int = 1,
  fit_learning_rate: float = 0.01,
  fit_num_steps: int = 500,
  optimization_num_results: int = 512,
  optimization_num_restarts: int = 10,
  optimization_num_samples: int = 128,
  improvement_factor: float = 1.0,
) -> Trial[list[tuple[dict[str, Any], float]]]:
  ...


@overload
def optimization(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['expected_improvement'],
  seed: int = 0,
  num_warm_up_steps: int = 12,
  batch_size: int = 1,
  fit_learning_rate: float = 0.01,
  fit_num_steps: int = 500,
  optimization_num_results: int = 512,
  optimization_num_restarts: int = 10,
  optimization_num_samples: int = 128,
  improvement_factor: float = 1.0,
) -> Trial[list[tuple[dict[str, Any], float]]]:
  ...


@overload
def optimization(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['upper_confidence_bound'],
  seed: int = 0,
  num_warm_up_steps: int = 12,
  batch_size: int = 1,
  fit_learning_rate: float = 0.01,
  fit_num_steps: int = 500,
  optimization_num_results: int = 512,
  optimization_num_restarts: int = 10,
  optimization_num_samples: int = 128,
  confidence: float = 1.0,
) -> Trial[list[tuple[dict[str, Any], float]]]:
  ...


def optimization(
  parameters: list[dict[str, Any]],
  *,
  method: Literal[
    'probability_of_improvement',
    'expected_improvement',
    'upper_confidence_bound',
  ] = 'upper_confidence_bound',
  seed: int = 0,
  num_warm_up_steps: int = 12,
  batch_size: int = 1,
  fit_learning_rate: float = 0.01,
  fit_num_steps: int = 500,
  optimization_num_results: int = 512,
  optimization_num_restarts: int = 10,
  optimization_num_samples: int = 128,
  improvement_factor: float = 1.0,
  confidence: float = 2.0,
) -> Trial[list[tuple[dict[str, Any], float]]]:
  """
  Setup for a bayesian optimization experiment.

  Example:
    >>> experiment = optimization([{'name': 'x', 'type': 'range', 'bounds': [0.0, 1.0]}])

  Args:
    parameters: List of parameters to be optimized via bayesian optimization.
      Each parameter is described by a dictionary with a 'name', a 'type' of either
      fixed, range, or log_range, and a 'value' for fixed parameters,
      or 'bounds' for range and log_range parameters.
    method: Name of the acquisiton function to use.
    seed: The initial random seed.
    num_warm_up_steps: Number of randomized exploration steps avoiding a cold start
      for the optimization process.
    batch_size: Number of parameterizations to be proposed at each step.
    fit_learning_rate: Learning rate parameter for fitting a surrogate model.
    fit_num_steps: Number of steps for fitting a surrogate model.
    optimization_num_results: Number of results to consider during acquisition function optimization.
    optimization_num_restarts: Number of restarts to consider during acquisiton functionn optimization.
    optimization_num_samples: Number of samples to consider during acquisiton functionn optimization.
    improvement_factor: Hyperparamter for probability of improvement and expected improvement optimization.
    confidence: Hyperparamter for upper confidence bound optimization.

  Returns:
    A trail object with next and best functions.
  """

  search_space = search_spaces.from_dicts(parameters)
  bounds = jnp.array(list(search_spaces.get_bounds(search_space)))

  from_parameterization_fn = partial(
    search_spaces.from_parameterizations, search_space.range_parameters
  )
  to_parameterization_fn = partial(
    search_spaces.to_parameterizations, search_space.range_parameters
  )

  match method:
    case 'probability_of_improvement':
      if batch_size == 1:
        return functions.optimization.probability_of_improvement(
          key=random.key(seed),
          bounds=bounds,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          improvement_factor=improvement_factor,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )

      else:
        return functions.optimization.batch_probability_of_improvement(
          key=random.key(seed),
          bounds=bounds,
          batch_size=batch_size,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          optimization_num_samples=optimization_num_samples,
          improvement_factor=improvement_factor,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )

    case 'expected_improvement':
      if batch_size == 1:
        return functions.optimization.expected_improvement(
          key=random.key(seed),
          bounds=bounds,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          improvement_factor=improvement_factor,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )

      else:
        return functions.optimization.batch_expected_improvement(
          key=random.key(seed),
          bounds=bounds,
          batch_size=batch_size,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          optimization_num_samples=optimization_num_samples,
          improvement_factor=improvement_factor,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )

    case 'upper_confidence_bound':
      if batch_size == 1:
        return functions.optimization.upper_confidence_bound(
          key=random.key(seed),
          bounds=bounds,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          confidence=confidence,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )

      else:
        return functions.optimization.batch_upper_confidence_bound(
          key=random.key(seed),
          bounds=bounds,
          batch_size=batch_size,
          num_warm_up_steps=num_warm_up_steps,
          fit_learning_rate=fit_learning_rate,
          fit_num_steps=fit_num_steps,
          optimization_num_results=optimization_num_results,
          optimization_num_restarts=optimization_num_restarts,
          optimization_num_samples=optimization_num_samples,
          confidence=confidence,
          from_parameterization=from_parameterization_fn,
          to_parameterization=to_parameterization_fn,
        )


@overload
def bandit(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['boltzmann'],
  seed: int = 0,
  tau: float = 1.0,
) -> Trial[ActionValues]:
  ...


@overload
def bandit(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['epsilon_greedy'],
  seed: int = 0,
  epsilon: float = 0.1,
) -> Trial[ActionValues]:
  ...


@overload
def bandit(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['thompson_sampling'],
  seed: int = 0,
) -> Trial[Beta]:
  ...


@overload
def bandit(
  parameters: list[dict[str, Any]],
  *,
  method: Literal['upper_confidence_bound'],
  seed: int = 0,
  confidence: float = math.sqrt(2),
) -> Trial[ActionValues]:
  ...


def bandit(
  parameters: list[dict[str, Any]],
  *,
  method: Literal[
    'thompson_sampling', 'upper_confidence_bound', 'epsilon_greedy', 'boltzmann'
  ] = 'upper_confidence_bound',
  seed: int = 0,
  tau: float = 1.0,
  epsilon: float = 0.1,
  confidence: float = math.sqrt(2),
) -> Trial[ActionValues] | Trial[Beta]:
  """
  Setup for a multi-arm bandit optimization experiment.

  Example:
    >>> experiment = bandit([{'name': 'arm', 'type': 'choice', 'values': ['left', 'middle', 'right']}])

  Args:
    parameters: List of parameters describing the variants to be optimized.
      Each parameter is described by a dictionary with a 'name', a 'type' of choice,
      and 'values' of the variants for each parameter.
    method: Name of the bandit algorithm to use.
    seed: The initial random seed.
    tau: Hyperparameter for boltzman bandits.
    epsilon: Hyperparameter for epsilon greedy bandits.
    confidence: Hyperparameter for upper confidence bound bandits.

  Returns:
    A trail object with next and best functions.
  """

  search_space = search_spaces.from_dicts(parameters)
  variants = list(search_spaces.get_variants(search_space))

  from_parameterization_fn = partial(
    search_spaces.from_parameterizations, search_space.choice_parameters
  )
  to_parameterization_fn = partial(
    search_spaces.to_parameterizations, search_space.choice_parameters
  )

  match method:
    case 'thompson_sampling':
      return functions.bandit.thompson_sampling(
        key=random.key(seed),
        variants=variants,
        from_parameterization=from_parameterization_fn,
        to_parameterization=to_parameterization_fn,
      )

    case 'upper_confidence_bound':
      return functions.bandit.upper_confidence_bound(
        key=random.key(seed),
        variants=variants,
        confidence=confidence,
        from_parameterization=from_parameterization_fn,
        to_parameterization=to_parameterization_fn,
      )

    case 'epsilon_greedy':
      return functions.bandit.epsilon_greedy(
        key=random.key(seed),
        variants=variants,
        epsilon=epsilon,
        from_parameterization=from_parameterization_fn,
        to_parameterization=to_parameterization_fn,
      )

    case 'boltzmann':
      return functions.bandit.boltzmann(
        key=random.key(seed),
        variants=variants,
        tau=tau,
        from_parameterization=from_parameterization_fn,
        to_parameterization=to_parameterization_fn,
      )
