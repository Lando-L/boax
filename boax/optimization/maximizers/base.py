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

from typing import Protocol, Tuple

from boax.optimization.acquisitions.base import Acquisition
from boax.optimization.spaces.base import SearchSpace
from boax.typing import Array


class Maximizer(Protocol):
  """Base interface for acquisition function maximizers."""

  def __call__(
    self, acquisition: Acquisition, space: SearchSpace
  ) -> Tuple[Array, Array]:
    """
    Maximizes an acquisition within the given bounds.

    Args:
      acquisition: The acquisition function to be maximized.
      space: The search space to be maximimzed over.

    Returns:
      A tuple of candidates and their acquisition values.
    """
