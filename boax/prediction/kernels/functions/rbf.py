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

"""Radial Basis Function (RBF) kernels."""

from jax import numpy as jnp

from boax.prediction.kernels.functions.distance import squared_distance
from boax.utils.typing import Numeric


def rbf(x: Numeric, y: Numeric, length_scale: Numeric) -> Numeric:
  return jnp.exp(-0.5 * squared_distance(x / length_scale, y / length_scale))
