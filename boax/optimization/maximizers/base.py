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
  """
  A callable type for the `init` step of a `Maximizer`.

  The `init` step takes PRNG key and an `Acquisition` function
  to construct the initial `n x q x d`-dim candidates for the maximizer.
  """

  def __call__(self, key: PRNGKey, acquisition: Acquisition) -> Array:
    """
    The `init` function.

    Args:
      key: The pseudo-random number generator key.
      acquisition: The acquisition function.

    Returns:
      An initial set of `n x q x d`-dim candidates.
    """


class MaximizerMaximizationFn(Protocol):
  """
  A callable type for the `maximize` step of a `Maximizer`.

  The `maximizer` step takes a set of `n x q x d`-dim initial candidates
  and an `Acquisition` function to find the function's maxima.
  """

  def __call__(
    self, candidates: Array, acquisition: Acquisition
  ) -> Tuple[Array, Array]:
    """
    The `maximize` function.

    Args:
      candidates: The set of initial candidates.
      acquisition: The acquisition function.

    Returns:
      A tuple of maxima candidates and their acquisition values.
    """


class Maximizer(NamedTuple):
  """
  A pair of pure functions implementing acqusition function maximization.

  Boax acquisition function maximizers are all implemented as _maximizer_.
  A maximizer is defined as a pair of pure functions, which are combined
  together in  a `NamedTuple` so that they can be referred to by name.

  Attributes:
    init: A pure function which, when called with a PRNG key and an
      `Acquisition` function, returns a set of initial candidates
      for the acqusition function maximization process.
    maximize: A pure function which takes as input an initial set of
      maxima candidates and an `Acquisition` function to find. The
      maximize function then returns the found maxima and their
      corresponding acquisition values.
  """

  init: MaximizerInitFn
  maximize: MaximizerMaximizationFn
