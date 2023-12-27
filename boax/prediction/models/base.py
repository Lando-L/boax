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

"""Base interface for surrogate models."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array

T = TypeVar('T')


class Model(Protocol, Generic[T]):
  """
  A callable type for surrogate models.

  A model takes a set of `n x d`-dim index points as input
  and returns a posterior prediction of type `T`.
  """

  def __call__(self, index_points: Array) -> T:
    """
    Computes the posterior prediction at the index points.

    Args:
      index_points: The `n x d` index points.

    Returns:
      The model evaluated at the given index points.
    """
