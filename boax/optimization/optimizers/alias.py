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

"""Alias for optimizers."""

from jax import numpy as jnp
from jax import random

from boax.core import distributions, samplers
from boax.optimization.optimizers.base import Optimizer
from boax.optimization.optimizers.initializers.base import Initializer
from boax.optimization.optimizers.solvers.base import Solver


def batch(initializer: Initializer, solver: Solver) -> Optimizer:
  """
  Batch optimizer.

  Example:
    >>> optimizer = batch(initializer, solver)
    >>> next_candidates = optimizer(key, fun, bounds, q, num_samples, num_restarts)

  Args:
    initializer: The initializer function.
    solver: The solver function.

  Returns:
    The batch `Optimizer`.
  """

  def optimizer(key, fun, bounds, q, num_samples, num_restarts):
    key1, key2 = random.split(key)

    x = jnp.reshape(
      samplers.halton_uniform(
        distributions.uniform.uniform(bounds[:, 0], bounds[:, 1])
      )(key1, num_samples * q),
      (num_samples, q, -1),
    )
    y = fun(x)

    candidates = initializer(key2, x, y, num_restarts)
    next_candidates, values = solver(fun, bounds, candidates)

    return next_candidates[jnp.argmax(values)]

  return optimizer


def sequential(initializer: Initializer, solver: Solver) -> Optimizer:
  """
  Sequential optimizer.

  Example:
    >>> optimizer = sequential(initializer, solver)
    >>> next_candidates = optimizer(key, fun, bounds, q, num_samples, num_restarts)

  Args:
    initializer: The initializer function.
    solver: The solver function.

  Returns:
    The sequential `Optimizer`.
  """

  inner = batch(initializer, solver)

  def optimizer(key, fun, bounds, q, num_samples, num_restarts):
    return jnp.concatenate(
      [
        inner(random.fold_in(key, i), fun, bounds, 1, num_samples, num_restarts)
        for i in range(q)
      ]
    )

  return optimizer
