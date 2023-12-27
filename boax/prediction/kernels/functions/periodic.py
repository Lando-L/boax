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

"""Periodic kernels."""

from jax import numpy as jnp

from boax.utils.typing import Numeric


def periodic(
  x: Numeric,
  y: Numeric,
  length_scale: Numeric,
  variance: Numeric,
  period: Numeric,
) -> Numeric:
  sine_squared = (jnp.sin(jnp.pi * (x - y) / period) / length_scale) ** 2
  K = variance * jnp.exp(-0.5 * jnp.sum(sine_squared, axis=0))
  return K
