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

"""Gaussian Process functions."""

from jax import numpy as jnp
from jax import scipy

from boax.core import distributions
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.models.kernels.base import Kernel
from boax.core.models.means.base import Mean
from boax.utils.typing import Array, Numeric


def prior(
  index_points: Array,
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric,
) -> MultivariateNormal:
  Kxx = kernel_fn(index_points, index_points)
  mean = mean_fn(index_points)
  cov = Kxx + jitter * jnp.identity(Kxx.shape[-1])

  return distributions.multivariate_normal.multivariate_normal(mean, cov)


def posterior(
  index_points: Array,
  observation_index_points: Array,
  observations: Array,
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric,
) -> MultivariateNormal:
  mz = mean_fn(index_points)
  mx = mean_fn(observation_index_points)

  Kxx = kernel_fn(observation_index_points, observation_index_points)
  Kxz = kernel_fn(observation_index_points, index_points)
  Kzz = kernel_fn(index_points, index_points)

  K = Kxx + jitter * jnp.identity(Kxx.shape[-1])
  chol = scipy.linalg.cholesky(K, lower=True)
  kinvy = scipy.linalg.solve_triangular(
    jnp.matrix_transpose(chol),
    scipy.linalg.solve_triangular(chol, observations - mx, lower=True),
  )
  v = scipy.linalg.solve_triangular(chol, Kxz, lower=True)

  mean = mz + jnp.squeeze(
    jnp.matmul(jnp.matrix_transpose(Kxz), kinvy[..., jnp.newaxis]), axis=-1
  )
  cov = (
    Kzz
    - jnp.matmul(jnp.matrix_transpose(v), v)
    + jitter * jnp.identity(Kzz.shape[-1])
  )

  return distributions.multivariate_normal.multivariate_normal(mean, cov)
