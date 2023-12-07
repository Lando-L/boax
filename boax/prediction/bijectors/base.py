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

"""Base interface for bijectors."""

from typing import NamedTuple, Protocol

from boax.typing import Array


class BijectorForwardFn(Protocol):
  """Base interface for forward functions."""

  def __call__(self, value: Array) -> Array:
    """
    Computes the forward function.

    `y = f(x)`

    Args:
      value: The input to the forward function.

    Returns:
      Result of the forward function.
    """


class BijectorInverseFn(Protocol):
  """Base interface for inverse functions."""

  def __call__(self, value: Array) -> Array:
    """
    Computes the inverse function

    `x = f^{-1}(y)`

    Args:
      value: The input to the inverse function.

    Returns:
      Result of the inverse function.
    """


class Bijector(NamedTuple):
  """Base interface for bijectors."""

  forward: BijectorForwardFn
  inverse: BijectorInverseFn
