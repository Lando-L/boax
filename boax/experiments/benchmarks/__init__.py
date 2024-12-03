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

"""The benchmark sub-package."""

from . import functions as functions
from .alias import beale as beale
from .alias import bohachevsky as bohachevsky
from .alias import branin as branin
from .alias import forrester_1d as forrester_1d
from .alias import grammacy_lee as grammacy_lee
from .alias import hartmann_6d as hartmann_6d
from .base import Benchmark as Benchmark
