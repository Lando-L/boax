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

"""The policies sub-package."""

from . import believes as believes
from .alias import boltzmann as boltzmann
from .alias import epsilon_greedy as epsilon_greedy
from .alias import thompson_sampling as thompson_sampling
from .alias import upper_confidence_bound as upper_confidence_bound
from .base import Policy as Policy
