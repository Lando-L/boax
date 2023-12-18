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

"""Base interface for acquisition function maximizers."""

from typing import NamedTuple, Protocol, Tuple

from boax.optimization.acquisitions.base import Acquisition
from boax.typing import Array, PRNGKey


class MaximizerInitFn(Protocol):
  """Base interface for maximization functions."""

  def __call__(self, key: PRNGKey, acquisition: Acquisition) -> Array:
    """
    Maximizes an acquisition.

    Args:
      key: The pseudo-random number generator key.
      acquisition: The acquisition function to be maximized.

    Returns:
      A tuple of candidates and their acquisition values.
    """


class MaximizerMaximizationFn(Protocol):
  """Base interface for maximization functions."""

  def __call__(
    self, candidates: Array, acquisition: Acquisition
  ) -> Tuple[Array, Array]:
    """
    Maximizes an acquisition.

    Args:
      candidates: Initial candidates.
      acquisition: The acquisition function to be maximized.

    Returns:
      A tuple of candidates and their acquisition values.
    """


class Maximizer(NamedTuple):
  """
  Base interface for acquisition function maximizers.

  Attributes:
    init: Initializes candidates for a given acquisition function.
    maximize: Maximizes a given acquisition function.
  """

  init: MaximizerInitFn
  maximize: MaximizerMaximizationFn
