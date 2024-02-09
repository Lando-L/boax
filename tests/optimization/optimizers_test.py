from operator import itemgetter

from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import optimizers


class OptimizersTest(parameterized.TestCase):
  def test_bfgs(self):
    key1, key2 = random.split(random.key(0))
    bounds = jnp.array([[-1.0, 1.0]])
    r, n, q, d = 25, 5, 3, 1

    x0 = random.uniform(
      key1, minval=bounds[:, 0], maxval=bounds[:, 1], shape=(r, q, d)
    )
    acqf = itemgetter((..., 0, 0))

    bfgs = optimizers.bfgs(acqf, bounds, x0, n)
    candidates = bfgs.init(key2)
    next_candidates, values = bfgs.update(candidates)

    self.assertEqual(next_candidates.shape, (n, q, d))
    self.assertEqual(values.shape, (n,))


if __name__ == '__main__':
  absltest.main()
