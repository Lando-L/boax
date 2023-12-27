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

"""Monte Carlo acquisition functions."""

from jax import lax, nn
from jax import numpy as jnp

from boax.utils.typing import Array, Numeric


def qei(values: Array, best: Numeric) -> Array:
  return lax.max(values - best, 0.0)


def qpoi(values: Array, best: Numeric, tau: Numeric) -> Array:
  return nn.sigmoid((values - best) / tau)


def qucb(values: Array, beta: Numeric) -> Array:
  mean = jnp.mean(values)
  return mean + beta * jnp.abs(values - mean)
