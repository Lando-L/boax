# Copyright 2023 The Bojax Authors.
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

from operator import itemgetter

from jax import numpy as jnp
from jax.scipy.optimize import minimize

from bojax._src.optimization.acquisitions.base import Acquisition
from bojax._src.optimization.maximizers.base import Maximizer
from bojax._src.optimization.spaces.base import SearchSpace
from bojax._src.typing import Numeric
from bojax._src.util import compose


def bfgs(num_initial_samples: int) -> Maximizer:
  """
  The BFGS acquisition function maximizer.

  Logarithm of the probability of improvement over the best function value observed so far.

  Args:
    num_initial_samples: The number of initial samples used by the maximizer.

  Returns:
    The corresponding `Maximizer`.
  """

  def maximizer(acquisition: Acquisition, space: SearchSpace) -> Numeric:
    results = minimize(
      fun=compose(
        jnp.negative, jnp.sum, acquisition, itemgetter((..., jnp.newaxis))
      ),
      x0=space.sample(num_initial_samples)[..., 0],
      method='bfgs',
    )

    candidates = jnp.clip(
      results.x[..., jnp.newaxis],
      a_min=space.bounds[..., 0],
      a_max=space.bounds[..., 1],
    )

    return candidates, acquisition(candidates)

  return maximizer
