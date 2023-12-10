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

"""Alias for acquisition function maximizers."""

from operator import itemgetter

from jax import numpy as jnp
from jax.scipy.optimize import minimize

from boax.optimization.acquisitions.base import Acquisition
from boax.optimization.maximizers.base import Maximizer
from boax.optimization.maximizers.util import halton_sequence
from boax.typing import Array, Numeric
from boax.util import compose


def bfgs(num_initial_samples: int, bounds: Array) -> Maximizer:
  """
  The BFGS acquisition function maximizer.

  Args:
    num_initial_samples: The number of initial samples used by the maximizer.
    bounds: The bounds of the search space.

  Returns:
    The corresponding `Maximizer`.
  """

  ndims = bounds.shape[0]
  lower_bounds = bounds[..., 0]
  upper_bounds = bounds[..., 1]

  initial_samples = lower_bounds + (
    upper_bounds - lower_bounds
  ) * halton_sequence(num_initial_samples, ndims)

  def maximizer(acquisition: Acquisition) -> Numeric:
    results = minimize(
      fun=compose(
        jnp.negative, jnp.sum, acquisition, itemgetter((..., jnp.newaxis))
      ),
      x0=initial_samples[..., 0],
      method='bfgs',
    )

    candidates = jnp.clip(
      results.x[..., jnp.newaxis],
      a_min=bounds[..., 0],
      a_max=bounds[..., 1],
    )

    return candidates, acquisition(candidates)

  return maximizer
