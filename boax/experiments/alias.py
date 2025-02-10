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

from collections import deque
from functools import partial
from typing import Any, TypeVar

from jax import numpy as jnp
from jax import random

from boax import acquisitions, policies
from boax.acquisitions import Acquisition, surrogates
from boax.acquisitions.surrogates import Surrogate
from boax.core import distributions, samplers
from boax.experiments import functions, search_spaces
from boax.experiments.base import Trial
from boax.policies import Policy, believes
from boax.policies.believes import Belief

T = TypeVar('T')
S = TypeVar('S')


def optimization(
  parameters: list[dict[str, Any]],
  *,
  seed: int = 0,
  num_warm_up_steps: int = 12,
  batch_size: int = 1,
  acquisition: Acquisition[S] | None = None,
  surrogate: Surrogate[T, S] | None = None,
) -> Trial[functions.optimization.State[T]]:
  """
  Setup for a bayesian optimization experiment.

  Example:
    >>> experiment = optimization([{'name': 'x', 'type': 'range', 'bounds': [0.0, 1.0]}])

  Args:
    parameters: List of parameters to be optimized via bayesian optimization.
      Each parameter is described by a dictionary with a 'name', a 'type' of either
      fixed, range, or log_range, and a 'value' for fixed parameters,
      or 'bounds' for range and log_range parameters.
    seed: The initial random seed.

  Returns:
    A trail object with next and best functions.

  Raises:
    ValueError: If given parameters cannot be parsed or don't match requirements.
  """

  search_space = search_spaces.from_dicts(parameters)

  if not search_space:
    raise ValueError('One or more of the given parameters is invalid.')

  if not search_space.range_parameters:
    raise ValueError('Found no valid range parameters in given parameter list.')

  if search_space.choice_parameters:
    raise ValueError(
      'Optimization currently does not support choice parameters.'
    )

  bounds = jnp.array(list(search_spaces.get_bounds(search_space)))

  from_parameterization_fn = partial(
    search_spaces.from_parameterizations, search_space.range_parameters
  )

  to_parameterization_fn = partial(
    search_spaces.to_parameterizations, search_space.range_parameters
  )

  sample_key, optimize_key, acquisition_key = random.split(random.key(seed), 3)

  sampler = samplers.halton_uniform(sample_key, (num_warm_up_steps, batch_size))

  samples = sampler(distributions.uniform.uniform(bounds[:, 0], bounds[:, 1]))

  return functions.optimization.optimize(
    key=optimize_key,
    warm_up_samples=deque(samples),
    from_parameterization=from_parameterization_fn,
    to_parameterization=to_parameterization_fn,
    surrogate=surrogate or surrogates.single_task_gaussian_process(bounds),
    acquisition=acquisition
    or acquisitions.q_log_expected_improvement(
      bounds, batch_size, samplers.multivariate_normal(acquisition_key, (256,))
    ),
  )


def bandit(
  parameters: list[dict[str, Any]],
  *,
  seed: int = 0,
  policy: Policy[T] | None = None,
  belief: Belief[T, S] | None = None,
) -> Trial[T]:
  """
  Setup for a multi-arm bandit optimization experiment.

  Example:
    >>> experiment = bandit([{'name': 'arm', 'type': 'choice', 'values': ['left', 'middle', 'right']}])

  Args:
    parameters: List of parameters describing the variants to be optimized.
      Each parameter is described by a dictionary with a 'name', a 'type' of choice,
      and 'values' of the variants for each parameter.
    seed: The initial random seed.
    policy: The policy to be used for optimization.
    belief: The belief to be used for optimization.

  Returns:
    A trail object with next and best functions.

  Raises:
    ValueError: If given parameters cannot be parsed or don't match requirements.
  """

  search_space = search_spaces.from_dicts(parameters)

  if not search_space:
    raise ValueError('One or more of the given parameters is invalid.')

  if not search_space.choice_parameters:
    raise ValueError(
      'Found no valid choice parameters in given parameter list.'
    )

  if search_space.range_parameters:
    raise ValueError('Bandit optimization does not support range parameters.')

  variants = list(search_spaces.get_variants(search_space))

  from_parameterization_fn = partial(
    search_spaces.from_parameterizations, search_space.choice_parameters
  )

  to_parameterization_fn = partial(
    search_spaces.to_parameterizations, search_space.choice_parameters
  )

  return functions.bandit.optimize(
    key=random.key(seed),
    variants=variants,
    from_parameterization=from_parameterization_fn,
    to_parameterization=to_parameterization_fn,
    policy=policy or policies.upper_confidence_bound(2.0),
    belief=belief or believes.continuous(len(variants)),
  )
