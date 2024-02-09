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

"""The scipy maximization functions."""

from functools import partial
from typing import Callable, Tuple

from jax import numpy as jnp
from jax.scipy import optimize

from boax.utils.functools import compose
from boax.utils.typing import Array


def maximize(
  candidates: Array,
  acquisition_fn: Callable[[Array], Array],
  bounds: Array,
  method: str,
) -> Tuple[Array, Array]:
  results = optimize.minimize(
    fun=compose(
      jnp.negative,
      jnp.sum,
      acquisition_fn,
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

  return clipped, acquisition_fn(clipped)
