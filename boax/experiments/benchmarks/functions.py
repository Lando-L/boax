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

"""The benchmark function implementations."""

from jax import numpy as jnp

from boax.utils.typing import Array, Numeric


def beale(x: Array) -> Numeric:
  a = 1.5 - x[..., 0] + x[..., 0] * x[..., 1]
  b = 2.25 - x[..., 0] + x[..., 0] * (x[..., 1] ** 2)
  c = 2.625 - x[..., 0] + x[..., 0] * (x[..., 1] ** 3)

  return (a**2) + (b**2) + (c**2)


def bohachevsky(x: Array) -> Numeric:
  a = x[..., 0] ** 2
  b = 2 * x[..., 1] ** 2
  c = 0.3 * jnp.cos(3 * jnp.pi * x[..., 0])
  d = 0.4 * jnp.cos(4 * jnp.pi * x[..., 1])

  return a + b - c - d + 0.7


def branin(x: Array) -> Numeric:
  a = (
    x[..., 1]
    - 5.1 / 4 * jnp.pi**2 * x[..., 0] ** 2
    + 5 / jnp.pi * x[..., 0]
    - 6
  )
  b = 10 * (1 - 1 / 8 * jnp.pi) * jnp.cos(x[..., 0])

  return a**2 + b + 10


def forrester_1d(x: Array) -> Numeric:
  return (6 * x[..., 0] - 2) ** 2 * jnp.sin(12 * x[..., 0] - 4)


def grammacy_lee(x: Array) -> Numeric:
  return jnp.sin(10 * jnp.pi * x[..., 0]) / 2 * x[..., 0] + (x[..., 0] - 1) ** 4


def hartmann_6d(x: Array) -> Numeric:
  alpha = jnp.array([1.0, 1.2, 3.0, 3.2])

  A = jnp.array(
    [
      [10, 3, 17, 3.5, 1.7, 8],
      [0.05, 10, 17, 0.1, 8, 14],
      [3, 3.5, 1.7, 10, 17, 8],
      [17, 8, 0.05, 10, 0.1, 14],
    ]
  )

  P = jnp.array(
    [
      [1312, 1696, 5569, 124, 8283, 5886],
      [2329, 4135, 8307, 3736, 1004, 9991],
      [2348, 1451, 3522, 2883, 3047, 6650],
      [4047, 8828, 8732, 5743, 1091, 381.0],
    ]
  )

  inner_sum = jnp.sum(
    A * (x[jnp.newaxis, jnp.newaxis, ...] - 0.0001 * P) ** 2, axis=-1
  )

  return -jnp.sum(alpha * jnp.exp(-inner_sum), axis=-1)
