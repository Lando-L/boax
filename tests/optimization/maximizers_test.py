from operator import itemgetter

from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import maximizers


class MaximizersTest(parameterized.TestCase):
  def test_bfgs(self):
    key = random.key(0)
    bounds = jnp.array([[-1.0, 1.0]])
    num_restarts = 5
    num_raw_samples = 20
    q, d = 3, 1

    acqf = itemgetter((..., 0, 0))
    maximizer = maximizers.bfgs(acqf, bounds, q, num_restarts, num_raw_samples)
    candidates, values = maximizer(key)

    self.assertEqual(candidates.shape, (num_restarts, q, d))
    self.assertEqual(values.shape, (num_restarts,))


if __name__ == '__main__':
  absltest.main()
