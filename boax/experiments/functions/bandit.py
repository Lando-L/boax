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

"""Bandit functions for experiments."""

from collections.abc import Callable
from typing import Any, TypeVar

from jax import random

from boax.experiments.base import Step, Trial
from boax.policies import Policy
from boax.policies.believes import Belief
from boax.utils.typing import PRNGKey

T = TypeVar('T')
S = TypeVar('S')


def optimize(
  key: PRNGKey,
  variants: list[Any],
  policy: Policy[T],
  belief: Belief[T, S],
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
) -> Trial[T]:
  def next(
    step: Step[T] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[T], list[dict[str, Any]]]:
    timestep, state = step or (0, belief.init())

    for parameterization, score in results:
      variant = tuple(from_parameterization(parameterization))

      if variant in variants:
        state = belief.update(
          state,
          variants.index(variant),
          score,
        )

    idx = policy(
      state,
      timestep,
      random.fold_in(key, timestep),
    )
    parameterizations = [to_parameterization(variants[int(idx)])]

    return Step(timestep + 1, state), parameterizations

  def best(step: Step[T]) -> tuple[dict[str, Any], float]:
    idx, reward = belief.best(step.state)
    parameterizations = to_parameterization(variants[int(idx)])

    return parameterizations, reward

  return Trial(next, best)
