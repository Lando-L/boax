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

"""The kernels sub-package."""

from bojax._src.prediction.kernels.alias import (
    matern_five_halves as matern_five_halves,
)
from bojax._src.prediction.kernels.alias import matern_half as matern_half
from bojax._src.prediction.kernels.alias import (
    matern_three_halves as matern_three_halves,
)
from bojax._src.prediction.kernels.alias import periodic as periodic
from bojax._src.prediction.kernels.alias import rbf as rbf
from bojax._src.prediction.kernels.base import Kernel as Kernel
from bojax._src.prediction.kernels.transformed import combine as combine
from bojax._src.prediction.kernels.transformed import scale as scale
