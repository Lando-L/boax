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

"""Alias for objectives."""

from jax import jit

from boax.core import distributions
from boax.core.distributions.multivariate_normal import MultivariateNormal
from boax.prediction.objectives.base import Objective
from boax.utils.typing import Array


def exact_marginal_log_likelihood() -> Objective[MultivariateNormal]:
  def objective(predictions: MultivariateNormal, targets: Array):
    return distributions.multivariate_normal.logpdf(targets, predictions)

  return jit(objective)
