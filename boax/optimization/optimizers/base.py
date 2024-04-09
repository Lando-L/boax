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

"""Base interface for optimizers."""

from typing import Callable, Protocol

from boax.utils.typing import Array, PRNGKey


class Optimizer(Protocol):
  """
  A callable type for the optimization functions.
  """

  def __call__(
    self,
    key: PRNGKey,
    fun: Callable[[Array], Array],
    bounds: Array,
    q: int,
    num_samples: int,
    num_restarts: int,
  ) -> Array:
    """
    The optimization function.

    Args:
      key: A PRNG key.
      fun: The function to be optimized.
      bounds: The bounds of the search space.
      q: The batch size.
      num_samples: The number of samples.
      num_restarts: The number of restarts.

    Returns:
      The maxima resulting of the optimization.
    """
