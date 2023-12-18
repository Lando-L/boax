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

"""Alias for surrogate models."""

from functools import partial
from typing import Tuple

from jax import jit

from boax.prediction.kernels.base import Kernel
from boax.prediction.means.base import Mean
from boax.prediction.models import functions
from boax.prediction.models.base import Model
from boax.typing import Array, Numeric


def gaussian_process(
  mean: Mean, kernel: Kernel, noise: Numeric, jitter: Numeric = 1e-6
) -> Model[Tuple[Array, Array]]:
  """
  Gaussian process model.

  Args:
    mean: The process' mean function.
    kernel: The process' covariance function.
    noise: The noise variance in the Normal likelihood distribution of the model.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process `Model`.
  """

  return jit(
    partial(
      functions.gaussian.prior,
      mean=mean,
      kernel=kernel,
      noise=noise,
      jitter=jitter,
    )
  )


def gaussian_process_regression(
  observation_index_points: Array,
  observations: Array,
  mean: Mean,
  kernel: Kernel,
  noise: Numeric,
  jitter: Numeric = 1e-6,
) -> Model[Tuple[Array, Array]]:
  """
  Gaussian process regression model.

  Args:
    observation_index_points: The index observation points.
    observations: The values at the index observation points.
    mean: The process' mean function.
    kernel: The process' covariance function.
    noise: The noise variance in the Normal likelihood distribution of the model.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process regression `Model`.
  """

  return jit(
    partial(
      functions.gaussian.posterior,
      observation_index_points=observation_index_points,
      observations=observations,
      mean=mean,
      kernel=kernel,
      noise=noise,
      jitter=jitter,
    )
  )
