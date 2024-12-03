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

"""Transformation functions for distributions."""

from jax import numpy as jnp

from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.core.distributions.normal import Normal


def norm_to_mvn(normal: Normal) -> MultivariateNormal:
  """
  Transforms a normal into a multivariate normal distribution object.

  Args:
    norm: The normal distribution.

  Returns:
    The `MultivariateNormal` distribution object.
  """

  return MultivariateNormal(
    normal.loc,
    (normal.scale**2) * jnp.identity(normal.scale.shape[0]),
  )


def mvn_to_norm(mvn: MultivariateNormal) -> Normal:
  """
  Transforms a multivariate normal into a batched normal distribution object.

  Args:
    mvn: The multivariate normal distribution.

  Returns:
    The batched `Normal` distribution object.
  """

  return Normal(
    mvn.mean,
    jnp.sqrt(jnp.diag(mvn.cov)),
  )
