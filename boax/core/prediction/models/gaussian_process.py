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

"""Gaussian for surrogate models."""

from functools import partial
from typing import Callable, TypeVar

from jax import jit, vmap
from jax import numpy as jnp

from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.prediction.models import functions
from boax.core.prediction.models.base import Model
from boax.core.prediction.models.kernels.base import Kernel
from boax.core.prediction.models.likelihoods.base import Likelihood
from boax.core.prediction.models.means.base import Mean
from boax.utils.functools import compose
from boax.utils.typing import Array, Numeric

T = TypeVar('T')


def exact(
  mean_fn: Mean,
  kernel_fn: Kernel,
  likelihood_fn: Likelihood[MultivariateNormal, T],
  observation_index_points: Array | None = None,
  observations: Array | None = None,
  jitter: Numeric = 1e-6,
) -> Model[T]:
  """
  The exact gaussian process model.

  Example:
    >>> model = gaussian_process(mean_fn, kernel_fn)
    >>> mean, cov = model(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    observation_index_points: The index points of the given observations.
    observations: The observed values.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The gaussian process `Model`.
  """

  if observation_index_points is None or observations is None:
    return jit(
      compose(
        likelihood_fn,
        partial(
          functions.gaussian.prior,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        ),
      )
    )

  else:
    return jit(
      compose(
        likelihood_fn,
        partial(
          functions.gaussian.posterior,
          observation_index_points=observation_index_points,
          observations=observations,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        ),
      )
    )


def fantasy(
  mean_fn: Mean,
  kernel_fn: Kernel,
  jitter: Numeric = 1e-6,
) -> Callable[[Array, Array, Array], MultivariateNormal]:
  return vmap(
    vmap(
      jit(
        partial(
          functions.gaussian.posterior,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        )
      ),
      in_axes=(0, None, 0),
    )
  )


def multi_fidelity(
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  likelihood_fn: Likelihood[MultivariateNormal, T],
  observation_index_points: Array | None = None,
  observation_fidelities: Array | None = None,
  observations: Array | None = None,
  jitter: Numeric = 1e-6,
) -> Model[T]:
  """
  The multi fidelity gaussian process model.

  Example:
    >>> model = multi_fidelity(mean_fn, kernel_fn, 1e-4)
    >>> mean, cov = model(x_train, y_train)(xs)

  Args:
    mean_fn: The process' mean function.
    kernel_fn: The process' covariance function.
    observation_index_points: The index points of the given observations.
    observation_fidelities: The fidelities of the given observatons.
    observations: The observed values.
    jitter: The scalar added to the diagonal of the covariance matrix to ensure positive definiteness.

  Returns:
    The multi fidelity gaussian process `Model`.
  """

  if (
    observation_index_points is None
    or observation_fidelities is None
    or observations is None
  ):
    return jit(
      compose(
        likelihood_fn,
        partial(
          functions.multi_fidelity.prior,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        ),
      )
    )

  else:
    return jit(
      compose(
        likelihood_fn,
        partial(
          functions.multi_fidelity.posterior,
          observation_index_points=observation_index_points,
          observation_fidelities=observation_fidelities,
          observations=observations,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        ),
      )
    )


def multi_fidelity_fantasy(
  mean_fn: Mean,
  kernel_fn: Callable[[Array, Array], Kernel],
  jitter: Numeric = 1e-6,
) -> Callable[[Array, Array, Array], MultivariateNormal]:
  fantasy_fn = vmap(
    vmap(
      jit(
        partial(
          functions.multi_fidelity.posterior,
          mean_fn=mean_fn,
          kernel_fn=kernel_fn,
          jitter=jitter,
        )
      ),
      in_axes=(0, 0, None, None, 0),
    )
  )

  def fn(
    fantasy_points,
    observation_index_points,
    observation_fidelities,
    observations,
  ):
    return fantasy_fn(
      fantasy_points,
      jnp.ones_like(fantasy_points),
      observation_index_points,
      observation_fidelities,
      observations,
    )

  return fn
