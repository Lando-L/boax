from absl.testing import absltest, parameterized
from jax import random

from boax.optimization import samplers


class SamplersTest(parameterized.TestCase):
  def test_halton_normal(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    scale = random.uniform(key2, (10,))

    result = samplers.halton_normal(loc, scale)(key3, 5)

    self.assertEqual(result.shape, (5, 10))

  def test_halton_uniform(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    minval = random.uniform(key1, (10,))
    maxval = minval + random.uniform(key2, (10,))

    result = samplers.halton_uniform(minval, maxval)(key3, 5)

    self.assertEqual(result.shape, (5, 10))


if __name__ == '__main__':
  absltest.main()
