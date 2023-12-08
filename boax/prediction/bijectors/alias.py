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

"""Alias for bijectors."""

from functools import partial

from jax import lax, nn
from jax import numpy as jnp

from boax.prediction.bijectors.base import Bijector
from boax.typing import Numeric
from boax.util import identity as identity_fn

identity = Bijector(identity_fn, identity_fn)
log = Bijector(jnp.log, jnp.exp)
exp = Bijector(jnp.exp, jnp.log)
softplus = Bijector(nn.softplus, lambda x: x + jnp.log(-jnp.expm1(-x)))


def shift(x: Numeric) -> Bijector:
  """
  Shift bijector.

  Computes `y = f(x; shift) = x + shift`.

  Args:
    x: The shift parameter.

  Returns:
    A shift `Bijector`.
  """

  return Bijector(partial(lax.add, y=x), partial(lax.sub, y=x))


def scale(x: Numeric) -> Bijector:
  """
  Scale bijector.

  Computes `y = f(x; scale) = x * scale`.

  Args:
    x: The scale parameter.

  Returns:
    A scale `Bijector`.
  """

  return Bijector(partial(lax.mul, y=x), partial(lax.div, y=x))
