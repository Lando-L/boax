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

"""Transformation functions for bijectors."""

from operator import attrgetter

from boax.prediction.bijectors.base import Bijector
from boax.util import compose


def chain(*bijectors: Bijector) -> Bijector:
  """
  Chain Bijectors.

  Bijector which applies a composition of bijectors.

  Args:
    bijectors: A sequence of bijectors.

  Returns:
    A chain `bijector`.
  """

  return Bijector(
    compose(*map(attrgetter('forward'), bijectors)),
    compose(*map(attrgetter('inverse'), reversed(bijectors))),
  )
