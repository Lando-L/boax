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

"""Alias for initialization functions."""

from jax import lax, nn, random
from jax import numpy as jnp

from boax.core.optimizers.initializers.base import Initializer
from boax.utils.typing import Array, Numeric


def q_batch(
  samples: Array,
  num_restarts: int,
  eta: Numeric = 1.0,
) -> Initializer:
  """
  Q batch initializer.

  Example:
    >>> initializer = q_batch(fun, samples, num_restarts)
    >>> candidates = initializer(key)

  Args:
    fun: The scoring function.
    samples: The candidate samples.
    num_restarts: The number of restarts.
    eta: The temperature parameter.

  Returns:
    The q-batch `Initializer`.
  """

  def initializer(fun, key):
    y = fun(samples)

    return random.choice(
      key,
      samples,
      (num_restarts,),
      p=jnp.exp(eta * nn.standardize(y, axis=0)),
    )

  return initializer


def q_batch_nonnegative(
  samples: Array,
  num_restarts: int,
  eta: Numeric = 1.0,
  alpha: Numeric = 1e-4,
) -> Initializer:
  """
  Q batch initializer.

  Example:
    >>> initializer = q_batch_nonnegative(fun, samples, num_restarts)
    >>> candidates = initializer(key)

  Args:
    fun: The scoring function.
    samples: The candidate samples.
    num_restarts: The number of restarts.
    eta: The temperature parameter.
    alpha: The alpha parameter.

  Returns:
    The q-batch non-negative `Initializer`.
  """

  def initializer(fun, key):
    y = fun(samples)
    max_val = jnp.max(samples)

    def cond(x):
      return jnp.sum(x[1]) < num_restarts

    def body(x):
      alpha = x[0] * 0.1
      alpha_pos = y >= alpha * max_val

      return (alpha, alpha_pos)

    _, alpha_pos = lax.while_loop(cond, body, (alpha, y >= alpha * max_val))

    return random.choice(
      key,
      samples[alpha_pos],
      (num_restarts,),
      p=jnp.exp(eta * (y[alpha_pos] / max_val - 1)),
    )

  return initializer
