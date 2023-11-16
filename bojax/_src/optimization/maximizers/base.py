# Copyright 2023 The Bojax Authors.
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

"""Base interface for acquisition function maximizers."""

from typing import Protocol, Tuple

from bojax._src.optimization.acquisitions.base import Acquisition
from bojax._src.typing import Array


class Maximizer(Protocol):
  def __call__(
    self, acquisition: Acquisition, bounds: Array
  ) -> Tuple[Array, Array]:
    """Maximizes an acquisition within the given bounds"""
