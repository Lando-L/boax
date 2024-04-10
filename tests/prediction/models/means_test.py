from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.prediction.models import means


class MeansTest(parameterized.TestCase):
  def test_zero(self):
    value = jnp.empty((10,))

    result = means.zero()(value)
    expected = jnp.zeros(())

    assert_trees_all_close(result, expected, atol=1e-4)

  def test_constant(self):
    x = jnp.array(2.0)
    value = jnp.empty((10,))

    result = means.constant(x)(value)
    expected = x

    assert_trees_all_close(result, expected, atol=1e-4)

  def test_linear(self):
    scale = jnp.array(2.0)
    bias = jnp.array(5.0)

    value = random.uniform(random.key(0), shape=(10,))

    result = means.linear(scale, bias)(value)
    expected = scale * value + bias

    assert_trees_all_close(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
