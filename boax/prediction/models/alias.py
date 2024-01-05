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

from jax import jit

from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.prediction.kernels.base import Kernel
from boax.prediction.means.base import Mean
from boax.prediction.models import functions
from boax.prediction.models.base import Model
from boax.utils.functools import compose
from boax.utils.typing import Array, Numeric


def gaussian_process(
  mean_fn: Mean,
  kernel_fn: Kernel,
  noise: Numeric,
  jitter: Numeric = 1e-6,
) -> Model[MultivariateNormal]:
  """
  The gaussian process model.

  Example:
    >>> model = gaussian_process(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    noise: The noise variance in the Normal likelihood distribution of the model.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process `Model`.
  """

  return jit(
    compose(
      partial(
        functions.gaussian.likelihood,
        noise=noise,
      ),
      partial(
        functions.gaussian.prior,
        mean_fn=mean_fn,
        kernel_fn=kernel_fn,
        jitter=jitter,
      ),
    )
  )


def gaussian_process_regression(
  observation_index_points: Array,
  observations: Array,
  mean_fn: Mean,
  kernel_fn: Kernel,
  noise: Numeric,
  jitter: Numeric = 1e-6,
) -> Model[MultivariateNormal]:
  """
  The gaussian process regression model.

  Example:
    >>> model = gaussian_process_regression(x_train, y_train, mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(xs)

  Args:
    observation_index_points: The index observation points.
    observations: The values at the index observation points.
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    noise: The noise variance in the Normal likelihood distribution of the model.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process regression `Model`.
  """

  return jit(
    compose(
      partial(
        functions.gaussian.likelihood,
        noise=noise,
      ),
      partial(
        functions.gaussian.posterior,
        observation_index_points=observation_index_points,
        observations=observations,
        mean_fn=mean_fn,
        kernel_fn=kernel_fn,
        jitter=jitter,
      )
    )
  )
