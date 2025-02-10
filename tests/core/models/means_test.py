from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.models import means
from boax.utils.typing import Array, Numeric, PRNGKey, Shape


class MeansTest(parameterized.TestCase):
  @parameterized.parameters(
    {'b': (), 'n': 3, 'd': 1},
    {'b': (), 'n': 10, 'd': 3},
    {'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_zero(self, b: Shape, n: int, d: int):
    value = jnp.empty(b + (n, d))

    result = means.zero()(value)
    expected = jnp.zeros(b + (n,))

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {'key': random.key(0), 'b': (), 'n': 3, 'd': 1},
    {'key': random.key(1), 'b': (), 'n': 10, 'd': 3},
    {'key': random.key(2), 'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_constant(self, key: PRNGKey, b: Shape, n: int, d: int):
    x = random.uniform(key)
    value = jnp.empty(b + (n, d))

    result = means.constant(x)(value)
    expected = jnp.full(b + (n,), x)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'key': random.key(0),
      'scale': jnp.array([0.5]),
      'bias': 3,
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'scale': jnp.array([0.5, 0.1, 3.0]),
      'bias': 1,
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(2),
      'scale': jnp.array([1.0, 2.0]),
      'bias': 10,
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_linear(
    self, key: PRNGKey, scale: Array, bias: Numeric, b: Shape, n: int, d: int
  ):
    values = random.uniform(key, b + (n, d))

    result = means.linear(scale, bias)(values)
    expected = jnp.matmul(values, scale) + bias

    assert_trees_all_close(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
