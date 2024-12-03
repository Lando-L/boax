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

"""Alias for likelihoods."""

from functools import partial

from jax import jit

from boax.core.distributions.beta import Beta
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.prediction.models.likelihoods import functions
from boax.core.prediction.models.likelihoods.base import Likelihood
from boax.utils.typing import Array, Numeric


def gaussian(
  noise: Numeric,
) -> Likelihood[MultivariateNormal, MultivariateNormal]:
  """
  The gaussian likelihood function.

  Example:
    >>> likelihood = gaussian(1e-4)
    >>> predictive = likelihood(mvn)

  Args:
    noise: The noise parameter.

  Returns:
    The gaussian `Likelihood` function.
  """

  return jit(
    partial(
      functions.marginal.gaussian,
      noise=noise,
    ),
  )


def beta(scale: Numeric) -> Likelihood[Array, Beta]:
  """
  The beta likelihood function.

  Example:
    >>> likelihood = beta(1e-4)
    >>> predictive = likelihood(xs)

  Args:
    scale: The scale parameter.

  Returns:
    The beta `Likelihood` function.
  """

  return jit(
    partial(
      functions.conditional.beta,
      scale=scale,
    ),
  )
