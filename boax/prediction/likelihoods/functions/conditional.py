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

from jax import nn

from boax.core import distributions
from boax.core.distributions.beta import Beta
from boax.utils.typing import Array


def beta(samples: Array, scale: Array) -> Beta:
  mixture = nn.sigmoid(samples)
  alpha = mixture * scale + 1
  beta = scale - alpha + 2

  return distributions.beta.beta(alpha, beta)
