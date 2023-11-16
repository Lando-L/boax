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

"""The acquisition sub-package."""

from bojax._src.optimization.acquisitions.alias import (
    log_expected_improvement as log_expected_improvement,
)
from bojax._src.optimization.acquisitions.alias import (
    log_probability_of_improvement as log_probability_of_improvement,
)
from bojax._src.optimization.acquisitions.alias import (
    scaled_posterior_mean as scaled_posterior_mean,
)
from bojax._src.optimization.acquisitions.alias import (
    upper_confidence_bound as upper_confidence_bound,
)
from bojax._src.optimization.acquisitions.base import Acquisition as Acquisition
