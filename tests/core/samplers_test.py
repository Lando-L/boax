from collections.abc import Sequence

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.core import distributions, samplers
from boax.utils.typing import PRNGKey


class SamplersTest(parameterized.TestCase):
  @parameterized.parameters(
    {"key": random.key(0), "distribution_shape": (10,), "sample_shape": (1, 1, 1)},
    {"key": random.key(1), "distribution_shape": (15,), "sample_shape": (2, 5, 3)},
    {"key": random.key(2), "distribution_shape": (20,), "sample_shape": (4, 25, 9)},
  )
  def test_normal(self, key: PRNGKey, distribution_shape: Sequence[int], sample_shape: Sequence[int]):
    key1, key2, key3 = random.split(key, 3)

    loc = random.uniform(key1, distribution_shape)
    scale = random.uniform(key2, distribution_shape)
    normal = distributions.normal.normal(loc, scale)

    result = samplers.normal(normal)(key3, sample_shape)

    assert_shape(result, sample_shape + distribution_shape)

  @parameterized.parameters(
    {"key": random.key(0), "distribution_shape": (10,), "sample_shape": (1, 1, 1)},
    {"key": random.key(1), "distribution_shape": (15,), "sample_shape": (2, 5, 3)},
    {"key": random.key(2), "distribution_shape": (20,), "sample_shape": (4, 25, 9)},
  )
  def test_uniform(self, key: PRNGKey, distribution_shape: Sequence[int], sample_shape: Sequence[int]):
    key1, key2, key3 = random.split(key, 3)

    a = random.uniform(key1, distribution_shape)
    b = a + random.uniform(key2, distribution_shape)
    uniform = distributions.uniform.uniform(a, b)

    result = samplers.uniform(uniform)(key3, sample_shape)

    assert_shape(result, sample_shape + distribution_shape)

  @parameterized.parameters(
    {"key": random.key(0), "distribution_shape": (10,), "sample_shape": (1, 1, 1)},
    {"key": random.key(1), "distribution_shape": (15,), "sample_shape": (2, 5, 3)},
    {"key": random.key(2), "distribution_shape": (20,), "sample_shape": (4, 25, 9)},
  )
  def test_halton_normal(self, key: PRNGKey, distribution_shape: Sequence[int], sample_shape: Sequence[int]):
    key1, key2, key3 = random.split(key, 3)

    loc = random.uniform(key1, distribution_shape)
    scale = random.uniform(key2, distribution_shape)
    normal = distributions.normal.normal(loc, scale)

    result = samplers.halton_normal(normal)(key3, sample_shape)

    assert_shape(result, sample_shape + distribution_shape)

  @parameterized.parameters(
    {"key": random.key(0), "distribution_shape": (10,), "sample_shape": (1, 1, 1)},
    {"key": random.key(1), "distribution_shape": (15,), "sample_shape": (2, 5, 3)},
    {"key": random.key(2), "distribution_shape": (20,), "sample_shape": (4, 25, 9)},
  )
  def test_halton_uniform(self, key: PRNGKey, distribution_shape: Sequence[int], sample_shape: Sequence[int]):
    key1, key2, key3 = random.split(key, 3)

    a = random.uniform(key1, distribution_shape)
    b = a + random.uniform(key2, distribution_shape)
    uniform = distributions.uniform.uniform(a, b)

    result = samplers.halton_uniform(uniform)(key3, sample_shape)

    assert_shape(result, sample_shape + distribution_shape)


if __name__ == '__main__':
  absltest.main()
