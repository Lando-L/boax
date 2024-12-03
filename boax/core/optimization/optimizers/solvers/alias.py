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

"""Alias for solver functions."""

from functools import partial
from typing import Callable

from jax import numpy as jnp
from jax.scipy import optimize

from boax.core.optimization.optimizers.solvers.base import Solver
from boax.utils.functools import compose
from boax.utils.typing import Array


def scipy(
  fun: Callable[[Array], Array],
  bounds: Array,
  method: str = 'bfgs',
) -> Solver:
  """
  Scipy solver.

  Example:
    >>> solver = scipy(fun, bounds)
    >>> next_candidates, values = solver(candidates)

  Args:
    bounds: The bounds of the search space.
    method: The solver method.

  Returns:
    The scipy `Solver`.
  """

  def solver(candidates):
    results = optimize.minimize(
      fun=compose(
        jnp.negative,
        jnp.sum,
        fun,
        partial(jnp.reshape, newshape=candidates.shape),
      ),
      x0=candidates.flatten(),
      method=method,
    )

    clipped = jnp.clip(
      jnp.reshape(results.x, candidates.shape),
      a_min=bounds[:, 0],
      a_max=bounds[:, 1],
    )

    return clipped, fun(clipped)

  return solver
