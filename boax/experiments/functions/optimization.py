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

from collections import deque
from collections.abc import Callable
from typing import Any, Generic, NamedTuple, TypeVar

from jax import numpy as jnp
from jax import random

from boax.acquisitions import Acquisition
from boax.acquisitions.surrogates import Surrogate
from boax.experiments.base import Step, Trial
from boax.utils.typing import Array, PRNGKey

T = TypeVar('T')
S = TypeVar('S')


class State(NamedTuple, Generic[T]):
  params: T
  observations: list[tuple[dict[str, Any], float]]


def optimize(
  key: PRNGKey,
  warm_up_samples: deque[Array],
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
  acquisition: Acquisition[S],
  surrogate: Surrogate[T, S],
) -> Trial[State[T]]:
  next_key, best_key = random.split(key)

  def next(
    step: Step[State[T]] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[State[T]], list[dict[str, Any]]]:
    timestep, (params, observations) = step or (0, (surrogate.init(), []))
    next_observations = observations + results

    if warm_up_samples:
      next_parameterizations = list(
        map(to_parameterization, warm_up_samples.pop())
      )

      return Step(
        timestep + 1, State(params, next_observations)
      ), next_parameterizations

    else:
      parameterizations, scores = tuple(map(list, zip(*next_observations)))

      observation_index_points = jnp.array(
        list(map(from_parameterization, parameterizations))
      )
      observations = jnp.array(scores)

      next_params = surrogate.update(
        params, observation_index_points, observations
      )
      posterior = surrogate.posterior(
        next_params, observation_index_points, observations
      )

      values, _ = acquisition(
        random.fold_in(next_key, timestep),
        posterior,
        observation_index_points,
        observations,
      )

      next_parameterizations = list(map(to_parameterization, values))

      return Step(
        timestep + 1, State(next_params, next_observations)
      ), next_parameterizations

  def best(step: Step[State[T]]) -> tuple[dict[str, Any], float]:
    parameterizations, scores = tuple(map(list, zip(*step.state.observations)))

    observation_index_points = jnp.array(
      list(map(from_parameterization, parameterizations))
    )
    observations = jnp.array(scores)

    posterior = surrogate.posterior(
      step.state.params, observation_index_points, observations
    )

    values, score = surrogate.best(
      random.fold_in(best_key, step.timestep),
      posterior,
    )

    best_parameterizations = to_parameterization(values)

    return best_parameterizations, score

  return Trial(next, best)
