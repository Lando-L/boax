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
from typing import Any

from jax import numpy as jnp
from jax import random

from boax.core.distributions.beta import Beta
from boax.core.optimization import policies
from boax.core.optimization.policies import believes
from boax.core.optimization.policies.believes import ActionValues
from boax.experiments.base import Step, Trial
from boax.utils.typing import PRNGKey


def thompson_sampling(
  key: PRNGKey,
  variants: list[Any],
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  policy = policies.thompson_sampling()
  belief = believes.binary(len(variants))

  def next(
    step: Step[Beta] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[Beta], list[dict[str, Any]]]:
    state = belief.init() if not step else step.state

    for parameterization, score in results:
      variant = tuple(from_parameterization(parameterization))

      if variant in variants:
        state = belief.update(
          state,
          variants.index(variant),
          bool(score),
        )

    next_step = Step(
      0 if not step else step.timestep + 1,
      state,
    )

    idx = policy(
      next_step.state,
      next_step.timestep,
      random.fold_in(key, next_step.timestep),
    )
    parameterizations = [to_parameterization(variants[int(idx)])]

    return next_step, parameterizations

  def best(step: Step[Beta]) -> tuple[dict[str, Any], float]:
    mean = step.state.a / step.state.b
    idx = jnp.argmax(mean)
    parameterizations = to_parameterization(variants[int(idx)])
    score = mean[idx]

    return parameterizations, float(score)

  return Trial(next, best)


def upper_confidence_bound(
  key: PRNGKey,
  variants: list[Any],
  confidence: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  policy = policies.upper_confidence_bound(confidence)
  belief = believes.continuous(len(variants))

  def next(
    step: Step[ActionValues] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[ActionValues], list[dict[str, Any]]]:
    state = belief.init() if not step else step.state

    for parameterization, score in results:
      variant = tuple(from_parameterization(parameterization))

      if variant in variants:
        state = belief.update(
          state,
          variants.index(variant),
          score,
        )

    next_step = Step(
      0 if not step else step.timestep + 1,
      state,
    )

    idx = policy(
      next_step.state,
      next_step.timestep,
      random.fold_in(key, next_step.timestep),
    )
    parameterizations = [to_parameterization(variants[int(idx)])]

    return next_step, parameterizations

  def best(step: Step[ActionValues]) -> tuple[dict[str, Any], float]:
    idx = jnp.argmax(step.state.n)
    parameterizations = to_parameterization(variants[int(idx)])
    score = step.state.q[idx]

    return parameterizations, float(score)

  return Trial(next, best)


def epsilon_greedy(
  key: PRNGKey,
  variants: list[Any],
  epsilon: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  policy = policies.epsilon_greedy(epsilon)
  belief = believes.continuous(len(variants))

  def next(
    step: Step[ActionValues] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[ActionValues], list[dict[str, Any]]]:
    state = belief.init() if not step else step.state

    for parameterization, score in results:
      variant = tuple(from_parameterization(parameterization))

      if variant in variants:
        state = belief.update(
          state,
          variants.index(variant),
          score,
        )

    next_step = Step(
      0 if not step else step.timestep + 1,
      state,
    )

    idx = policy(
      next_step.state,
      next_step.timestep,
      random.fold_in(key, next_step.timestep),
    )
    parameterizations = [to_parameterization(variants[int(idx)])]

    return next_step, parameterizations

  def best(step: Step[ActionValues]) -> tuple[dict[str, Any], float]:
    idx = jnp.argmax(step.state.n)
    parameterizations = to_parameterization(variants[int(idx)])
    score = step.state.q[idx]

    return parameterizations, float(score)

  return Trial(next, best)


def boltzmann(
  key: PRNGKey,
  variants: list[Any],
  tau: float,
  from_parameterization: Callable[[dict[str, Any]], list[Any]],
  to_parameterization: Callable[[list[Any]], dict[str, Any]],
):
  policy = policies.boltzmann(tau)
  belief = believes.continuous(len(variants))

  def next(
    step: Step[ActionValues] | None = None,
    results: list[tuple[dict[str, Any], float]] = [],
  ) -> tuple[Step[ActionValues], list[dict[str, Any]]]:
    state = belief.init() if not step else step.state

    for parameterization, score in results:
      variant = tuple(from_parameterization(parameterization))

      if variant in variants:
        state = belief.update(
          state,
          variants.index(variant),
          score,
        )

    next_step = Step(
      0 if not step else step.timestep + 1,
      state,
    )

    idx = policy(
      next_step.state,
      next_step.timestep,
      random.fold_in(key, next_step.timestep),
    )
    parameterizations = [to_parameterization(variants[int(idx)])]

    return next_step, parameterizations

  def best(step: Step[ActionValues]) -> tuple[dict[str, Any], float]:
    idx = jnp.argmax(step.state.n)
    parameterizations = to_parameterization(variants[int(idx)])
    score = step.state.q[idx]

    return parameterizations, float(score)

  return Trial(next, best)
