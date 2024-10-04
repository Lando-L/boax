from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.prediction.models import means
from boax.utils.typing import Numeric, PRNGKey


class MeansTest(parameterized.TestCase):
  @parameterized.parameters(
    {"num_index_points": 10}
  )
  def test_zero(self, num_index_points: int):
    value = jnp.empty((num_index_points,))

    result = means.zero()(value)
    expected = jnp.zeros(())

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {"x": jnp.array(2.0), "num_index_points": 10}
  )
  def test_constant(self, x: Numeric, num_index_points: 10):
    value = jnp.empty((num_index_points,))

    result = means.constant(x)(value)
    expected = x

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {"key": random.key(0), "scale": jnp.array(2.0), "bias": jnp.array(5.0), "num_index_points": 10}
  )
  def test_linear(self, key: PRNGKey, scale: Numeric, bias: Numeric, num_index_points: int):
    value = random.uniform(key, shape=(num_index_points,))

    result = means.linear(scale, bias)(value)
    expected = scale * value + bias

    assert_trees_all_close(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
