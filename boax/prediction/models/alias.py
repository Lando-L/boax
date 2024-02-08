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
from typing import Callable

from jax import jit, vmap

from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.prediction.kernels.base import Kernel
from boax.prediction.means.base import Mean
from boax.prediction.models import functions
from boax.prediction.models.base import Model
from boax.utils.typing import Array, Numeric


def gaussian_process(
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric = 1e-6,
) -> Model[MultivariateNormal]:
  """
  The gaussian process model.

  Example:
    >>> model = gaussian_process(mean_fn, kernel_fn)
    >>> mean, cov = model(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process `Model`.
  """

  return jit(
    partial(
      functions.gaussian.prior,
      mean_fn=mean_fn,
      kernel_fn=kernel_fn,
      jitter=jitter,
    )
  )


def gaussian_process_regression(
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric = 1e-6,
) -> Callable[[Array, Array], Model[MultivariateNormal]]:
  """
  The gaussian process regression model.

  Example:
    >>> model = gaussian_process_regression(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(x_train, y_train)(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process regression `Model`.
  """

  def regression(observation_index_points, observations):
    return jit(
      partial(
        functions.gaussian.posterior,
        observation_index_points=observation_index_points,
        observations=observations,
        mean_fn=mean_fn,
        kernel_fn=kernel_fn,
        jitter=jitter,
      )
    )
  
  return regression


def gaussian_process_fantasy(
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric = 1e-6,
) -> Callable[[Array, Array], Model[MultivariateNormal]]:
  """
  The gaussian process fantasy model.

  Example:
    >>> model = gaussian_process_fantasy(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(x_train, y_train)(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process fantasy `Model`.
  """

  def fantasy(observation_index_points, observations):
    def model(index_points):
      return vmap(
        vmap(
          partial(
            functions.gaussian.posterior,
            mean_fn=mean_fn,
            kernel_fn=kernel_fn,
            jitter=jitter,
          ),
          in_axes=(0, None, 0),
        ),
        in_axes=(0, 0, 0),
      )(index_points, observation_index_points, observations)
    
    return jit(model)
  
  return fantasy


def multi_fidelity_regression(
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  jitter: Numeric = 1e-6,
) -> Model[MultivariateNormal]:
  """
  The multi fidelity gaussian process regression model.

  Example:
    >>> model = multi_fidelity_regression(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(x_train, y_train)(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The multi fidelity gaussian process regression `Model`.
  """

  def regression(observation_index_points, observations):
    observation_values, observation_fidelities = functions.multi_fidelity.split(observation_index_points)

    def model(index_points):
      values, fidelities = functions.multi_fidelity.split(index_points)

      return functions.multi_fidelity.posterior(
        index_points=values,
        index_points_fidelities=fidelities,
        observation_index_points=observation_values,
        observation_index_points_fidelities=observation_fidelities,
        observations=observations,
        mean_fn=mean_fn,
        kernel_fn=kernel_fn,
        jitter=jitter,
      )
    
    return jit(model)
  
  return regression


def multi_fidelity_fantasy(
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  jitter: Numeric = 1e-6,
) -> Model[MultivariateNormal]:
  """
  The multi fidelity gaussian process fantasy model.

  Example:
    >>> model = multi_fidelity_fantasy(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(x_train, y_train)(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The multi fidelity gaussian process fantasy `Model`.
  """

  def fantasy(observation_index_points, observations):
    observation_values, observation_fidelities = functions.multi_fidelity.split(observation_index_points)

    def model(index_points):
      values, fidelities = functions.multi_fidelity.split(index_points)

      return vmap(
        vmap(
          partial(
            functions.multi_fidelity.posterior,
            mean_fn=mean_fn,
            kernel_fn=kernel_fn,
            jitter=jitter,
          ),
          in_axes=(None, None, None, None, 0),
        )
      )(
        values,
        fidelities,
        observation_values,
        observation_fidelities,
        observations
      )
    
    return jit(model)
  
  return fantasy
