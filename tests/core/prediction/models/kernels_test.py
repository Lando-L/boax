import math

from absl.testing import absltest, parameterized
from chex import assert_shape, assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core.prediction.models import kernels
from boax.core.prediction.models.kernels import functions
from boax.utils.typing import Array, PRNGKey


class KernelsTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'num_index_points': 10}
  )
  def test_rbf_function(
    self, key: PRNGKey, length_scale: float, num_index_points: int
  ):
    x, y = random.uniform(key, shape=(2, num_index_points))

    result = functions.rbf.rbf(x, y, length_scale)
    expected = jnp.exp(-jnp.linalg.norm((x - y) ** 2) / (2 * length_scale**2))

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {'length_scale': jnp.array([0.2, 0.5]), 'num_index_points': 10}
  )
  def test_rbf_kernel(self, length_scale: Array, num_index_points: int):
    x, y = jnp.ones((2, num_index_points, 2))

    result = kernels.rbf(length_scale)(x, y)

    assert_shape(result, (num_index_points, num_index_points))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'num_index_points': 10}
  )
  def test_matern_one_half_function(
    self, key: PRNGKey, length_scale: float, num_index_points: int
  ):
    x, y = random.uniform(key, (2, num_index_points), minval=-1.0, maxval=1.0)

    result = functions.matern.one_half(x, y, length_scale)
    expected = jnp.exp(-jnp.linalg.norm(x - y) / length_scale)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {'length_scale': jnp.array([0.2, 0.5]), 'num_index_points': 10}
  )
  def test_matern_one_half_kernel(
    self, length_scale: Array, num_index_points: int
  ):
    x, y = jnp.ones((2, num_index_points, 2))

    result = kernels.matern_one_half(length_scale)(x, y)

    assert_shape(result, (num_index_points, num_index_points))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'num_index_points': 10}
  )
  def test_matern_three_halves_function(
    self, key: PRNGKey, length_scale: float, num_index_points: int
  ):
    x, y = random.uniform(key, (2, num_index_points), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(3) * jnp.linalg.norm(x - y) / length_scale

    result = functions.matern.three_halves(x, y, length_scale)
    expected = (1 + z) * jnp.exp(-z)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {'length_scale': jnp.array([0.2, 0.5]), 'num_index_points': 10}
  )
  def test_matern_three_halves_kernel(
    self, length_scale: Array, num_index_points: int
  ):
    x, y = jnp.ones((2, num_index_points, 2))

    result = kernels.matern_three_halves(length_scale)(x, y)

    assert_shape(result, (num_index_points, num_index_points))

  @parameterized.parameters(
    {'key': random.key(0), 'length_scale': 0.2, 'num_index_points': 10}
  )
  def test_matern_five_halves_function(
    self, key: PRNGKey, length_scale: float, num_index_points: int
  ):
    x, y = random.uniform(key, (2, num_index_points), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(5) * jnp.linalg.norm(x - y) / length_scale

    result = functions.matern.five_halves(x, y, length_scale)
    expected = (1 + z + z**2 / 3) * jnp.exp(-z)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {'length_scale': jnp.array([0.2, 0.5]), 'num_index_points': 10}
  )
  def test_matern_five_halves_kernel(
    self, length_scale: Array, num_index_points: int
  ):
    x, y = jnp.ones((2, num_index_points, 2))

    result = kernels.matern_five_halves(length_scale)(x, y)

    assert_shape(result, (num_index_points, num_index_points))

  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': 0.2,
      'period': 1.0,
      'variance': 0.1,
      'num_index_points': 10,
    }
  )
  def test_periodic_function(
    self,
    key: PRNGKey,
    length_scale: float,
    period: float,
    variance: float,
    num_index_points: int,
  ):
    x, y = random.uniform(key, (2, num_index_points), minval=-1.0, maxval=1.0)
    z = (jnp.sin(jnp.pi * (x - y) / period) / length_scale) ** 2

    result = functions.periodic.periodic(x, y, length_scale, variance, period)
    expected = variance * jnp.exp(-0.5 * jnp.sum(z, axis=0))

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'length_scale': jnp.array([0.2, 0.5]),
      'period': jnp.array([1.0, 2.0]),
      'variance': 0.1,
      'num_index_points': 10,
    }
  )
  def test_periodic_kernel(
    self,
    length_scale: Array,
    period: Array,
    variance: float,
    num_index_points: int,
  ):
    x, y = jnp.ones((2, num_index_points, 2))

    result = kernels.periodic(length_scale, variance, period)(x, y)

    assert_shape(result, (num_index_points, num_index_points))

  @parameterized.parameters(
    {
      'key': random.key(0),
      'amplitude': 5.0,
      'length_scale': 0.2,
      'num_index_points': 10,
    },
  )
  def test_scaled(
    self,
    key: PRNGKey,
    amplitude: float,
    length_scale: float,
    num_index_points: int,
  ):
    x, y = random.uniform(key, shape=(2, num_index_points, 1))

    inner = kernels.rbf(length_scale)

    result = kernels.transformations.scaled(inner, amplitude)(x, y)
    expected = amplitude * inner(x, y)

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scale': 0.2,
      'power': 1.0,
      'num_index_points': 10,
    },
  )
  def test_linear_truncated(
    self, key: PRNGKey, length_scale: float, power: float, num_index_points: int
  ):
    x, x_fid, y, y_fid = random.uniform(key, shape=(4, num_index_points, 1))

    inner = kernels.matern_five_halves(length_scale)

    result = kernels.transformations.linear_truncated(inner, inner, power)(
      x_fid, y_fid
    )(x, y)
    factor = (1 - x_fid) * (1 - y_fid.T) * (1 + x_fid * y_fid.T)
    expected = inner(x, y) + inner(x, y) * factor

    assert_trees_all_close(result, expected, atol=1e-4)

  @parameterized.parameters(
    {
      'key': random.key(0),
      'length_scales': [0.2, 0.3, 0.4],
      'num_index_points': 10,
    },
  )
  def test_additive(
    self, key: PRNGKey, length_scales: list[int], num_index_points: int
  ):
    x, y = random.uniform(key, shape=(2, num_index_points, 1))

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
      'length_scales': [0.2, 0.3, 0.4],
      'num_index_points': 10,
    },
  )
  def test_product(
    self, key: PRNGKey, length_scales: list[int], num_index_points: int
  ):
    x, y = random.uniform(key, shape=(2, num_index_points, 1))

    result = kernels.transformations.product(
      *map(kernels.rbf, [0.2, 0.3, 0.4])
    )(x, y)
    expected = math.prod(
      kernels.rbf(length_scale)(x, y) for length_scale in length_scales
    )

    assert_trees_all_close(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
