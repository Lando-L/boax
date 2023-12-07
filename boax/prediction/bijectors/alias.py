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

from jax import lax, nn
from jax import numpy as jnp

from boax.prediction.bijectors.base import Bijector
from boax.typing import Array, Numeric
from boax.util import identity as identity_fn


def identity() -> Bijector:
  """
  Identity bijector.

  Computes `y = f(x) = x`.

  Returns:
    An identity `Bijector`.
  """

  return Bijector(
    identity_fn,
    identity_fn,
  )


def shift(x: Numeric) -> Bijector:
  """
  Shift bijector.

  Computes `y = f(x; shift) = x + shift`.

  Args:
    x: The shift parameter.

  Returns:
    A shift `Bijector`.
  """

  def forward(value: Array) -> Array:
    return value + x

  def inverse(value: Array) -> Array:
    return value - x

  return Bijector(
    forward,
    inverse,
  )


def scale(x: Numeric) -> Bijector:
  """
  Scale bijector.

  Computes `y = f(x; scale) = x * scale`.

  Args:
    x: The scale parameter.

  Returns:
    A scale `Bijector`.
  """

  inv_x = 1 / x

  def forward(value: Array) -> Array:
    out_shape = lax.broadcast_shapes(x.shape, value.shape)
    return jnp.broadcast_to(x, out_shape) * value

  def inverse(value: Array) -> Array:
    out_shape = lax.broadcast_shapes(x.shape, value.shape)
    return jnp.broadcast_to(inv_x, out_shape) * value

  return Bijector(
    forward,
    inverse,
  )


def log():
  """
  Logarithmic bijector.

  Computes `y = f(x) = log(x)`.

  Returns:
    A logarithmic `Bijector`.
  """

  def forward(value: Array) -> Array:
    return jnp.log(value)

  def inverse(value: Array) -> Array:
    return jnp.exp(value)

  return Bijector(
    forward,
    inverse,
  )


def exp():
  """
  Exponential bijector.

  Computes `y = f(x) = exp(x)`.

  Returns:
    An exponential `Bijector`.
  """

  def forward(value: Array) -> Array:
    return jnp.exp(value)

  def inverse(value: Array) -> Array:
    return jnp.log(value)

  return Bijector(
    forward,
    inverse,
  )


def softplus():
  """
  Softplus bijector.

  Computes `y = f(x) = Log[1 + exp(x)]`.

  Returns:
    A softplus `Bijector`.
  """

  def forward(value: Array) -> Array:
    return nn.softplus(value)

  def inverse(value: Array) -> Array:
    return value + jnp.log(-jnp.expm1(-value))

  return Bijector(
    forward,
    inverse,
  )
