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

"""Gaussian process functions."""

from typing import Tuple

from jax import numpy as jnp
from jax import scipy

from bojax._src.prediction.kernels.base import Kernel
from bojax._src.prediction.means.base import Mean
from bojax._src.prediction.processes.base import Process
from bojax._src.typing import Array, Numeric


def prior(
  mean: Mean,
  kernel: Kernel,
  noise: Numeric,
  jitter: Numeric = 1e-3,
) -> Process[Tuple[Array, Array]]:
  def process(value: Array) -> Tuple[Array, Array]:
    Kxx = kernel(value, value)
    loc = mean(value).flatten()
    cov = Kxx + (noise + jitter) * jnp.identity(Kxx.shape[-1])

    return loc, cov

  return process


def posterior(
  x_train: Array,
  y_train: Array,
  mean: Mean,
  kernel: Kernel,
  noise: Numeric,
  jitter: Numeric = 1e-3,
) -> Process[Tuple[Array, Array]]:
  def process(value: Array) -> Tuple[Array, Array]:
    mx = mean(x_train).flatten()
    mz = mean(value).flatten()

    Kxx = kernel(x_train, x_train)
    Kxz = kernel(x_train, value)
    Kzz = kernel(value, value)

    K = Kxx + (noise + jitter) * jnp.identity(Kxx.shape[-1])
    chol = scipy.linalg.cholesky(K, lower=True)
    kinvy = scipy.linalg.solve_triangular(
      chol.T, scipy.linalg.solve_triangular(chol, y_train - mx, lower=True)
    )
    v = scipy.linalg.solve_triangular(chol, Kxz, lower=True)

    loc = mz + jnp.dot(Kxz.T, kinvy)
    cov = Kzz - jnp.dot(v.T, v) + jitter * jnp.identity(Kzz.shape[-1])

    return loc, cov

  return process
