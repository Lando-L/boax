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

from jax import jit, lax, nn, random
from jax import numpy as jnp

from boax.core.distributions.beta import Beta
from boax.policies.base import Policy
from boax.policies.believes import ActionValues
from boax.utils.typing import Numeric, PRNGKey


def epsilon_greedy(epsilon: Numeric) -> Policy[ActionValues]:
  """
  The epsilon greedy policy function.

  Greedily selects the variant with the highest action value with a probability
  of `1 - epsilon` or uniform randomly selects a variant with probability of `epsilon`.

  Example:
    >>> policy = epsilon_greedy(epsilon)
    >>> variant = policy(params, timestep, key)

  Args:
    epsilon: The parameter guiding exploration vs exploitation.

  Returns:
    The corresponding `Policy`.
  """

  def explore(params: ActionValues, key: PRNGKey) -> int:
    return random.choice(key, jnp.arange(len(params.q)))

  def exploit(params: ActionValues, key: PRNGKey) -> int:
    return jnp.argmax(params.q)

  def policy(params: ActionValues, timestep: int, key: PRNGKey) -> int:
    uniform_rng, choice_rng = random.split(key)

    return lax.cond(
      random.uniform(uniform_rng) < epsilon,
      explore,
      exploit,
      params,
      choice_rng,
    )

  return jit(policy)


def boltzmann(tau: Numeric) -> Policy[ActionValues]:
  """
  The boltzmann policy function.

  Randomly selects a variant proportional to the current action-values.

  Example:
    >>> policy = boltzmann(tau)
    >>> variant = policy(params, timestep, key)

  Args:
    tau: The temperature parameter guiding exploration vs exploitation.

  Returns:
    The corresponding `Policy`.
  """

  def policy(params: ActionValues, timestep: int, key: PRNGKey) -> int:
    return random.choice(
      key, jnp.arange(len(params.q)), p=nn.softmax(params.q / tau)
    )

  return jit(policy)


def upper_confidence_bound(confidence: Numeric) -> Policy[ActionValues]:
  """
  The upper confidence bound policy function.

  Selects the variant with highest action-value plus the upper confidence bound.

  Example:
    >>> policy = upper_confidence_bound(confidence)
    >>> variant = policy(params, timestep, key)

  Args:
    confidence: The confidence parameter guiding exploration vs exploitation.

  Returns:
    The corresponding `Policy`.
  """

  def policy(params: ActionValues, timestep: int, key: PRNGKey) -> int:
    return jnp.argmax(
      params.q + confidence * jnp.sqrt(jnp.log(timestep) / params.n)
    )

  return jit(policy)


def thompson_sampling() -> Policy[Beta]:
  """
  The thompson sampling policy function.

  Randomly samples action values for all variants
  and selects the variant with the highest sampled values.

  Example:
    >>> policy = thompson_sampling()
    >>> variant = policy(params, timestep, key)

  Returns:
    The corresponding `Policy`.
  """

  def policy(params: Beta, timestep: int, key: PRNGKey) -> int:
    return jnp.argmax(random.beta(key, params.a, params.b))

  return jit(policy)
