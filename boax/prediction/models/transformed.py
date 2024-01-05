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

"""Transformation functions for models."""

from typing import Callable, Tuple

from jax import scipy
from jax import numpy as jnp

from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.prediction.models.base import Model
from boax.utils.functools import compose
from boax.utils.typing import Array


def sampled(
  model: Model[Tuple[Array, Array]],
) -> Model[Callable[[Array], Array]]:
  def sample(mvn: MultivariateNormal):
    def fn(base_samples):
      return mvn.mean + jnp.dot(
        scipy.linalg.cholesky(mvn.cov, lower=True), base_samples
      )

    return fn

  return compose(
    sample,
    model,
  )
