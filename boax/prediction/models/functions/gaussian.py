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
from boax.prediction.kernels.base import Kernel
from boax.prediction.means.base import Mean
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
    chol.T, scipy.linalg.solve_triangular(chol, observations - mx, lower=True)
  )
  v = scipy.linalg.solve_triangular(chol, Kxz, lower=True)

  mean = mz + jnp.dot(Kxz.T, kinvy)
  cov = Kzz - jnp.dot(v.T, v) + jitter * jnp.identity(Kzz.shape[-1])

  return distributions.multivariate_normal.multivariate_normal(mean, cov)


def likelihood(mvn: MultivariateNormal, noise: Array) -> MultivariateNormal:
  return distributions.multivariate_normal.multivariate_normal(
    mvn.mean, mvn.cov + noise * jnp.identity(mvn.cov.shape[-1])
  )


# def variational(
#   index_points: Array,
#   inducing_points: Array,
#   variational_mean: Array,
#   variational_root_cov: Array,
#   mean: Mean,
#   kernel: Kernel,
#   jitter: Numeric,
# ) -> Tuple[Array, Array]:
#   mx = mean(inducing_points)
#   mz = mean(index_points)

#   Kxx = kernel(inducing_points, inducing_points)
#   Kxz = kernel(inducing_points, index_points)
#   Kzz = kernel(index_points, index_points)

#   Lz = jnp.linalg.cholesky(Kxx + jitter * jnp.identity(inducing_points.shape[0]), lower=True)
#   Lz_inv_Kxz = scipy.linalg.solve_triangular(Lz, Kxz, lower=True)
#   Kxx_inv_Kxz = scipy.linalg.solve_triangular(Lz.T, Lz_inv_Kxz)
#   Kxx_inv_Kxz_sqrt = jnp.matmul(Kxx_inv_Kxz.T, variational_root_cov)

#   loc = mz + jnp.matmul(Kxx_inv_Kxz.T, variational_mean - mx)
#   cov = (
#     Kzz
#     - jnp.matmul(Lz_inv_Kxz.T, Lz_inv_Kxz)
#     + jnp.matmul(Kxx_inv_Kxz_sqrt, Kxx_inv_Kxz_sqrt.T)
#   ) + jitter * jnp.identity(inducing_points.shape[0])

#   return loc, cov
