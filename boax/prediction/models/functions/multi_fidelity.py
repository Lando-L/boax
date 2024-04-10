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

"""Multi fidelity functions."""

from typing import Callable, Tuple

from jax import numpy as jnp
from jax import scipy

from boax.core import distributions
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.prediction.models.kernels.base import Kernel
from boax.prediction.models.means.base import Mean
from boax.utils.typing import Array, Numeric


def prior(
  index_points: Array,
  fidelities: Array,
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  jitter: Numeric,
) -> MultivariateNormal:
  Kxx = kernel_fn(fidelities, fidelities)(index_points, index_points)
  mean = mean_fn(index_points)
  cov = Kxx + jitter * jnp.identity(Kxx.shape[-1])

  return distributions.multivariate_normal.multivariate_normal(mean, cov)


def posterior(
  index_points: Array,
  fidelities: Array,
  observation_index_points: Array,
  observation_fidelities: Array,
  observations: Array,
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  jitter: Numeric,
) -> MultivariateNormal:
  mz = mean_fn(index_points)
  mx = mean_fn(observation_index_points)

  Kxx = kernel_fn(observation_fidelities, observation_fidelities)(
    observation_index_points, observation_index_points
  )
  
  Kxz = kernel_fn(observation_fidelities, fidelities)(
    observation_index_points, index_points
  )

  Kzz = kernel_fn(fidelities, fidelities)(
    index_points, index_points
  )

  K = Kxx + jitter * jnp.identity(Kxx.shape[-1])
  chol = scipy.linalg.cholesky(K, lower=True)
  kinvy = scipy.linalg.solve_triangular(
    chol.T, scipy.linalg.solve_triangular(chol, observations - mx, lower=True)
  )
  v = scipy.linalg.solve_triangular(chol, Kxz, lower=True)

  mean = mz + jnp.dot(Kxz.T, kinvy)
  cov = Kzz - jnp.dot(v.T, v) + jitter * jnp.identity(Kzz.shape[-1])

  return distributions.multivariate_normal.multivariate_normal(mean, cov)
