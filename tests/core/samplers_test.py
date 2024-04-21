from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.core import distributions, samplers


class SamplersTest(parameterized.TestCase):
  def test_normal(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    scale = random.uniform(key2, (10,))
    normal = distributions.normal.normal(loc, scale)

    result = samplers.normal(normal)(key3, (2, 5, 3))

    assert_shape(result, (2, 5, 3, 10))

  def test_uniform(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    a = random.uniform(key1, (10,))
    b = a + random.uniform(key2, (10,))
    uniform = distributions.uniform.uniform(a, b)

    result = samplers.uniform(uniform)(key3, (2, 5, 3))

    assert_shape(result, (2, 5, 3, 10))
  
  def test_halton_normal(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    loc = random.uniform(key1, (10,))
    scale = random.uniform(key2, (10,))
    normal = distributions.normal.normal(loc, scale)

    result = samplers.halton_normal(normal)(key3, (2, 5, 3))

    assert_shape(result, (2, 5, 3, 10))

  def test_halton_uniform(self):
    key1, key2, key3 = random.split(random.key(0), 3)

    a = random.uniform(key1, (10,))
    b = a + random.uniform(key2, (10,))
    uniform = distributions.uniform.uniform(a, b)

    result = samplers.halton_uniform(uniform)(key3, (2, 5, 3))

    assert_shape(result, (2, 5, 3, 10))


if __name__ == '__main__':
  absltest.main()
