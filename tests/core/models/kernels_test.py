import math

from absl.testing import absltest, parameterized
from chex import assert_shape, assert_trees_all_close
from jax import random

from boax.core.models import kernels
from boax.utils.typing import PRNGKey, Shape


class KernelsTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'b': (), 'n': 3, 'd': 1},
    {'key': random.key(1), 'length_scale': 0.2, 'b': (), 'n': 10, 'd': 3},
    {'key': random.key(2), 'length_scale': 0.2, 'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_rbf(
    self,
    key: PRNGKey,
    length_scale: float,
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.rbf(length_scale)(x, y)

    assert_shape(result, b + (n, n))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'b': (), 'n': 3, 'd': 1},
    {'key': random.key(1), 'length_scale': 0.2, 'b': (), 'n': 10, 'd': 3},
    {'key': random.key(2), 'length_scale': 0.2, 'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_matern_one_half(
    self,
    key: PRNGKey,
    length_scale: float,
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.matern_one_half(length_scale)(x, y)

    assert_shape(result, b + (n, n))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'b': (), 'n': 3, 'd': 1},
    {'key': random.key(1), 'length_scale': 0.2, 'b': (), 'n': 10, 'd': 3},
    {'key': random.key(2), 'length_scale': 0.2, 'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_matern_three_halves(
    self,
    key: PRNGKey,
    length_scale: float,
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.matern_three_halves(length_scale)(x, y)

    assert_shape(result, b + (n, n))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'b': (), 'n': 3, 'd': 1},
    {'key': random.key(1), 'length_scale': 0.2, 'b': (), 'n': 10, 'd': 3},
    {'key': random.key(2), 'length_scale': 0.2, 'b': (5, 3), 'n': 3, 'd': 2},
  )
  def test_matern_five_halves(
    self,
    key: PRNGKey,
    length_scale: float,
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.matern_five_halves(length_scale)(x, y)

    assert_shape(result, b + (n, n))

  @parameterized.parameters(
    {
      'key': random.key(0),
      'amplitude': 0.5,
      'length_scale': 0.2,
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'amplitude': 1.0,
      'length_scale': 0.2,
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(2),
      'amplitude': 2.0,
      'length_scale': 0.2,
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_scaled(
    self,
    key: PRNGKey,
    amplitude: float,
    length_scale: float,
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    inner = kernels.rbf(length_scale)

    result = kernels.transformations.scaled(inner, amplitude)(x, y)
    expected = amplitude * inner(x, y)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(2),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_additive(
    self,
    key: PRNGKey,
    length_scales: list[float],
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.transformations.additive(*map(kernels.rbf, length_scales))(
      x, y
    )

    expected = sum(
      kernels.rbf(length_scale)(x, y) for length_scale in length_scales
    )

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (),
      'n': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (),
      'n': 10,
      'd': 3,
    },
    {
      'key': random.key(2),
      'length_scales': [0.2, 0.3, 2.0],
      'b': (5, 3),
      'n': 3,
      'd': 2,
    },
  )
  def test_product(
    self,
    key: PRNGKey,
    length_scales: list[float],
    b: Shape,
    n: int,
    d: int,
  ):
    x, y = random.uniform(key, shape=(2,) + b + (n, d))

    result = kernels.transformations.product(*map(kernels.rbf, length_scales))(
      x, y
    )

    expected = math.prod(
      kernels.rbf(length_scale)(x, y) for length_scale in length_scales
    )

    assert_trees_all_close(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
