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

"""Input transformation functions."""

from boax.core.prediction.models.transformations.base import InputTransformation
from boax.utils.typing import Array


def normalized(bounds: Array) -> InputTransformation:
  def normalize(index_points: Array) -> Array:
    return (index_points - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

  return normalize
