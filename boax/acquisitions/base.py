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

"""Base interface for acquisition functions."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array, PRNGKey

T = TypeVar('T')


class Acquisition(Protocol, Generic[T]):
  """
  A callable type for acquisition functions.

  An acquisition function takes a set of index_points
  as input and returns a numeric acquisition value.
  """

  def __call__(
    self,
    key: PRNGKey,
    model: T,
    observation_index_points: Array,
    observations: Array,
  ) -> tuple[Array, float]:
    """
    Evaluates the acquisition function on a set of index points.

    Args:
      index_points: The set of points the acquisition function is evaluated.

    Returns:
      The acquisition values.
    """
