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

from boax.optimization.maximizers import functions
from boax.optimization.maximizers.base import Maximizer
from boax.typing import Array, Numeric


def bfgs(
  bounds: Array,
  q: int,
  num_restarts: int,
  num_raw_samples: int,
  eta: Numeric = 1.0,
) -> Maximizer:
  """
  The BFGS acquisition function maximizer.

  Args:
    bounds: The bounds of the search space.
    q: The q batch size
    num_restarts: The number of maximization restarts.
    num_initial_samples: The number of initial samples used by the maximizer.
    eta: The temperature parameter.

  Returns:
    The corresponding `Maximizer`.
  """

  return Maximizer(
    init=partial(
      functions.initialization.q_batch_initialization,
      bounds=bounds,
      q=q,
      num_restarts=num_restarts,
      num_raw_samples=num_raw_samples,
      eta=eta,
    ),
    maximize=partial(
      functions.scipy.bfgs,
      bounds=bounds,
    ),
  )
