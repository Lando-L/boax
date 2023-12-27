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

"""The stats sub-package."""

from typing import Tuple

from jax import numpy as jnp
from jax import scipy

from boax.utils.typing import Array


def mvn_to_norm(mean: Array, cov: Array) -> Tuple[Array, Array]:
  return mean, jnp.sqrt(jnp.diag(cov))


def sample_mvn(mean: Array, cov: Array, base_samples: Array) -> Array:
  return mean + jnp.dot(scipy.linalg.cholesky(cov, lower=True), base_samples)


def scale_improvement(loc: Array, scale: Array, best: Array) -> Array:
  return (loc - best) / scale
