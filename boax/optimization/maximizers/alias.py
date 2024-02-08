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
from boax.utils.typing import Array, Numeric


def bfgs(
  bounds: Array,
  q: int,
  num_raw_samples: int,
  num_restarts: int,
  eta: Numeric = 1.0,
) -> Maximizer:
  """
  The BFGS acquisition function maximizer.

  Example:
    >>> acqf = upper_confidence_bound(surrogate, 2.0)
    >>> maximizer = bfgs(bounds, q, num_raw_samples, num_restarts)
    >>> candidates = maximizer.init(acqf, key)
    >>> next_candidates, values = maximizer.update(acqf, candidates)

  Args:
    bounds: The bounds of the search space.
    q: The bounds of the search space.
    num_raw_samples: The bounds of the search space.
    num_restarts: The bounds of the search space.
    eta: The bounds of the search space.

  Returns:
    The corresponding `Maximizer`.
  """

  return Maximizer(
    partial(
      functions.initialization.q_batch,
      bounds=bounds,
      q=q,
      num_raw_samples=num_raw_samples,
      num_restarts=num_restarts,
      eta=eta,
    ),
    partial(
      functions.scipy.bfgs,
      bounds=bounds,
    )
  )
