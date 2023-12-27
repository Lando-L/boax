from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import maximizers
from boax.utils.functools import const


class MaximizersTest(parameterized.TestCase):
  def test_bfgs(self):
    bounds = jnp.array([[-1.0, 1.0]])
    q = 5
    num_restarts = 11
    num_raw_samples = 101

    init_acqf = const(jnp.linspace(-1, 1, num_raw_samples) ** 2)
    max_acqf = const(jnp.linspace(-1, 1, num_restarts) ** 2)
    maximizer = maximizers.bfgs(bounds, q, num_restarts, num_raw_samples)

    init_candidates = maximizer.init(random.key(0), init_acqf)
    candidates, values = maximizer.maximize(init_candidates, max_acqf)

    self.assertEqual(init_candidates.shape, (num_restarts, q, 1))
    self.assertEqual(candidates.shape, (num_restarts, q, 1))
    self.assertEqual(values.shape, (num_restarts,))


if __name__ == '__main__':
  absltest.main()
