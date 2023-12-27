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

"""Range constraints."""

from jax import scipy

from boax.utils.typing import Array, Numeric


def le(loc: Array, scale: Array, x: Numeric) -> Array:
  return scipy.stats.norm.sf(x, loc, scale)


def lle(loc: Array, scale: Array, x: Numeric) -> Array:
  return scipy.stats.norm.logsf(x, loc, scale)


def ge(loc: Array, scale: Array, x: Numeric) -> Array:
  return scipy.stats.norm.cdf(x, loc, scale)


def lge(loc: Array, scale: Array, x: Numeric) -> Array:
  return scipy.stats.norm.logcdf(x, loc, scale)
