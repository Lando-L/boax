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

from typing import NamedTuple

from jax import jit, lax
from jax import numpy as jnp

from boax.core.distributions.beta import Beta
from boax.optimization.policies.believes.base import Belief
from boax.utils.typing import Array


class ActionValues(NamedTuple):
  n: Array
  q: Array


def continuous(num_variants: int) -> Belief[ActionValues, float]:
  """
  The continous belief.

  Stores the number of tries and average reward for each variant.

  Example:
    >>> belief = continous(10)
    >>> params = belief.init()
    >>> updated_params = belief.update(params, variant, reward)

  Args:
    num_variants: The number of variants.

  Returns;
    The corresponding `Belief`.
  """

  def init_fn() -> ActionValues:
    return ActionValues(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.float32),
    )

  def update_fn(params: ActionValues, variant: int, reward: float) -> ActionValues:
    return ActionValues(
      params.n.at[variant].add(1),
      params.q.at[variant].add(
        (reward - params.q[variant]) / params.n[variant]
      ),
    )

  return Belief(jit(init_fn), jit(update_fn))


def binary(num_variants: int) -> Belief[Beta, bool]:
  """
  The binary Beta belief.

  Stores a beta distribution for each variant.

  Example:
    >>> belief = beta(10)
    >>> params = belief.init()
    >>> updated_params = belief.update(params, variant, reward)

  Args:
    num_variants: The number of variants.

  Returns;
    The corresponding `Belief`.
  """

  def init_fn() -> Beta:
    return Beta(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.int32),
    )

  def update_fn(params: Beta, variant: int, reward: bool) -> Beta:
    def increment_alpha():
      return Beta(params.a.at[variant].add(1), params.b)

    def increment_beta():
      return Beta(params.a, params.b.at[variant].add(1))

    return lax.cond(
      reward,
      increment_alpha,
      increment_beta,
    )

  return Belief(jit(init_fn), jit(update_fn))
