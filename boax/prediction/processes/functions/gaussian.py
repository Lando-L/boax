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

from typing import Tuple

from jax import numpy as jnp
from jax import scipy

from boax.prediction.kernels.base import Kernel
from boax.prediction.means.base import Mean
from boax.typing import Array, Numeric


def prior(
  index_points: Array, mean: Mean, kernel: Kernel, jitter: Numeric
) -> Tuple[Array, Array]:
  Kxx = kernel(index_points, index_points)
  loc = mean(index_points)
  cov = Kxx + jitter * jnp.identity(Kxx.shape[-1])
  return loc, cov


def posterior(
  index_points: Array,
  observation_index_points: Array,
  observations: Array,
  mean: Mean,
  kernel: Kernel,
  noise: Numeric,
  jitter: Numeric,
) -> Tuple[Array, Array]:
  mz = mean(index_points)
  mx = mean(observation_index_points)

  Kxx = kernel(observation_index_points, observation_index_points)
  Kxz = kernel(observation_index_points, index_points)
  Kzz = kernel(index_points, index_points)

  K = Kxx + (noise + jitter) * jnp.identity(observation_index_points.shape[0])
  chol = scipy.linalg.cholesky(K, lower=True)
  kinvy = scipy.linalg.solve_triangular(
    chol.T, scipy.linalg.solve_triangular(chol, observations - mx, lower=True)
  )
  v = scipy.linalg.solve_triangular(chol, Kxz, lower=True)

  loc = mz + jnp.dot(Kxz.T, kinvy)
  cov = Kzz - jnp.dot(v.T, v) + jitter * jnp.identity(index_points.shape[0])

  return loc, cov
