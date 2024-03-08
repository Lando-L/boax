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

"""Alias for acquisition function maximizers."""

from functools import partial
from typing import Callable

from boax.optimization.optimizers import functions
from boax.optimization.optimizers.base import Optimizer
from boax.utils.typing import Array


def bfgs(
  fn: Callable[[Array], Array],
  bounds: Array,
  x0: Array,
  num_samples: int,
) -> Optimizer:
  """
  The BFGS acquisition function optimizer.

  Example:
    >>> optimizer = bfgs(fn, bounds, x0, num_samples)
    >>> candidates = optimizer.init(key)
    >>> next_candidates, values = optimizer(candidates)

  Args:
    fn: The function to be optimized.
    bounds: The bounds of the search space.
    x0: The index points to consider.
    num_samples: The number of sampled candidates.

  Returns:
    The corresponding `Optimizer`.
  """

  return Optimizer(
    partial(
      functions.initialization.q_batch,
      fn=fn,
      x0=x0,
      num_samples=num_samples,
    ),
    partial(
      functions.scipy.maximize,
      fn=fn,
      bounds=bounds,
      method='bfgs',
    ),
  )
