from operator import itemgetter

from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import maximizers


class MaximizersTest(parameterized.TestCase):
  def test_bfgs(self):
    key = random.key(0)
    bounds = jnp.array([[-1.0, 1.0]])
    n, q, d = 10, 3, 1

    acqf = itemgetter((..., 0, 0))
    maximizer = maximizers.bfgs(bounds, q, 25, 10)
    candidates = maximizer.init(acqf, key)
    next_candidates, values = maximizer.update(acqf, candidates)

    self.assertEqual(next_candidates.shape, (n, q, d))
    self.assertEqual(values.shape, (n,))


if __name__ == '__main__':
  absltest.main()
