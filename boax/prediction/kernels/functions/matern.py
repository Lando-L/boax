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

"""Matern kernels."""

import math

from jax import numpy as jnp

from boax.prediction.kernels.functions.distance import euclidean_distance
from boax.utils.typing import Numeric

sqrt_3 = math.sqrt(3)
sqrt_5 = math.sqrt(5)


def one_half(x: Numeric, y: Numeric, length_scale: Numeric) -> Numeric:
  return jnp.exp(-euclidean_distance(x / length_scale, y / length_scale))


def three_halves(x: Numeric, y: Numeric, length_scale: Numeric) -> Numeric:
  K = sqrt_3 * euclidean_distance(x / length_scale, y / length_scale)
  K = (1.0 + K) * jnp.exp(-K)
  return K


def five_halves(x: Numeric, y: Numeric, length_scale: Numeric) -> Numeric:
  K = sqrt_5 * euclidean_distance(x / length_scale, y / length_scale)
  K = (1.0 + K + K**2 / 3.0) * jnp.exp(-K)
  return K
