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

"""Base interface for model transformation functions."""

from typing import Generic, Protocol, TypeVar

from boax.utils.typing import Array

A = TypeVar('A')
B = TypeVar('B')


class InputTransformation(Protocol):
  """
  A callable type for model input transformation functions.

  A transformation function takes a set of `n x d`-dim index points as input
  and returns a transformed set of index points.
  """

  def __call__(self, index_points: Array, **kwargs) -> Array:
    """
    Computes the transformation function at the index points.

    Args:
      index_points: The `n x d` index points.

    Returns:
      The transformed index points.
    """


class OutcomeTransformation(Protocol, Generic[A, B]):
  """
  A callable type for model outcome transformation functions.

  A transformation function takes a posterior prediction of type `A` as input
  and returns a transformed prediction of type `B`.
  """

  def __call__(self, posterior: A, **kwargs) -> B:
    """
    Computes the transformation function given the posterior.

    Args:
      posterior: The posterior prediction.

    Returns:
      The transformed posterior.
    """
