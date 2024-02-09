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

"""Base interface for acquisition function optimization."""

from typing import NamedTuple, Protocol, Tuple, TypeVar

from boax.utils.typing import Array, PRNGKey

T = TypeVar('T')


class OptimizerInitFn(Protocol):
  """
  A callable type for the `init` step of a `Optimizer`.
  """

  def __call__(self, key: PRNGKey) -> Array:
    """
    The `init` function.

    Args:
      key: A PRNG key.

    Returns:
      The initial set of initial candidates.
    """


class OptimizerUpdateFn(Protocol):
  """
  A callable type for the `update` step of a `Optimizer`.
  """

  def __call__(self, candidates: Array) -> Tuple[Array, Array]:
    """
    The `update` function.

    Args:
      candidates: The initial guess.

    Returns:
      A tuple of the maxima and their acquisition values.
    """


class Optimizer(NamedTuple):
  """
  A pair of pure functions implementing acquisition function optimization.

  Attributes:
    init: A pure function which, when called with an acquisition function
      and a pseudo-random key, returns an initial set of candidates.
    update: A pure function which takes an acquisition function and a set
      of initial set of candidates as inputs. The update function then
      computes the maximized set of candidates as well as their values.
  """

  init: OptimizerInitFn
  update: OptimizerUpdateFn
