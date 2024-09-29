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
from boax.optimization.policies.evaluators.base import Evaluator
from boax.utils.typing import Array, Numeric


class ActionValues(NamedTuple):
  n: Array
  q: Array


def action_value(num_variants: Numeric) -> Evaluator[ActionValues]:
  """
  The Action value evaluator.

  Stores the number of tries and average reward for each variant.

  Example:
    >>> evaluator = action_value(10)
    >>> params = evaluator.init()
    >>> updated_params = evaluator.update(params, variant, reward)

  Args:
    num_variants: The number of variants.

  Returns;
    The corresponding `Evaluator`.
  """

  def init_fn() -> ActionValues:
    return ActionValues(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.float32),
    )

  def update_fn(
    params: ActionValues, variant: Numeric, reward: Numeric
  ) -> ActionValues:
    return ActionValues(
      params.n.at[variant].add(1),
      params.q.at[variant].add(
        (reward - params.q[variant]) / params.n[variant]
      ),
    )

  return Evaluator(jit(init_fn), jit(update_fn))


def beta(num_variants: Numeric) -> Evaluator[Beta]:
  """
  The Beta evaluator.

  Stores a beta distribution for each variant.

  Example:
    >>> evaluator = beta(10)
    >>> params = evaluator.init()
    >>> updated_params = evaluator.update(params, variant, reward)

  Args:
    num_variants: The number of variants.

  Returns;
    The corresponding `Evaluator`.
  """

  def init_fn() -> Beta:
    return Beta(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.int32),
    )

  def update_fn(params: Beta, variant: Numeric, reward: Numeric) -> Beta:
    def increment_alpha():
      return Beta(params.a.at[variant].add(1), params.b)

    def increment_beta():
      return Beta(params.a, params.b.at[variant].add(1))

    return lax.cond(
      reward,
      increment_alpha,
      increment_beta,
    )

  return Evaluator(jit(init_fn), jit(update_fn))
