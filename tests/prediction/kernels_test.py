from functools import partial

import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.prediction import kernels
from boax.prediction.kernels import functions


class KernelsTest(parameterized.TestCase):
  def test_rbf_function(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    result = functions.rbf.rbf(x, y, length_scale)
    expected = jnp.exp(-jnp.linalg.norm((x - y) ** 2) / (2 * length_scale**2))

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_rbf_kernel(self):
    length_scale = jnp.array([0.2, 0.5])

    x = jnp.ones((10, 2))
    y = jnp.ones((10, 2))

    result = kernels.rbf(length_scale)(x, y)

    self.assertEqual(result.shape, (10, 10))

  def test_matern_one_half_function(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)

    result = functions.matern.one_half(x, y, length_scale)
    expected = jnp.exp(-jnp.linalg.norm(x - y) / length_scale)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_one_half_kernel(self):
    length_scale = jnp.array([0.2, 0.5])

    x = jnp.ones((10, 2))
    y = jnp.ones((10, 2))

    result = kernels.matern_one_half(length_scale)(x, y)

    self.assertEqual(result.shape, (10, 10))

  def test_matern_three_halves_function(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(3) * jnp.linalg.norm(x - y) / length_scale

    result = functions.matern.three_halves(x, y, length_scale)
    expected = (1 + z) * jnp.exp(-z)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_three_halves_kernel(self):
    length_scale = jnp.array([0.2, 0.5])

    x = jnp.ones((10, 2))
    y = jnp.ones((10, 2))

    result = kernels.matern_three_halves(length_scale)(x, y)

    self.assertEqual(result.shape, (10, 10))

  def test_matern_five_halves_function(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(5) * jnp.linalg.norm(x - y) / length_scale

    result = functions.matern.five_halves(x, y, length_scale)
    expected = (1 + z + z**2 / 3) * jnp.exp(-z)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_three_halves_kernel(self):
    length_scale = jnp.array([0.2, 0.5])

    x = jnp.ones((10, 2))
    y = jnp.ones((10, 2))

    result = kernels.matern_five_halves(length_scale)(x, y)

    self.assertEqual(result.shape, (10, 10))

  def test_periodic_function(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2
    period = 1.0
    variance = 0.1

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)
    z = (jnp.sin(jnp.pi * (x - y) / period) / length_scale) ** 2

    result = functions.periodic.periodic(x, y, length_scale, variance, period)
    expected = variance * jnp.exp(-0.5 * jnp.sum(z, axis=0))

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_periodic_kernel(self):
    length_scale = jnp.array([0.2, 0.5])
    period = jnp.array([1.0, 2.0])
    variance = 0.1

    x = jnp.ones((10, 2))
    y = jnp.ones((10, 2))

    result = kernels.periodic(length_scale, variance, period)(x, y)

    self.assertEqual(result.shape, (10, 10))

  def test_scale(self):
    key1, key2 = random.split(random.key(0))
    amplitude = 5
    length_scale = 0.2

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    inner = kernels.rbf(length_scale)

    result = kernels.scale(amplitude, inner)(x, y)
    expected = amplitude * inner(x, y)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_combine(self):
    key1, key2 = random.split(random.key(0))
    length_scales = [0.2, 0.3, 0.4]

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    inner = list(map(kernels.rbf, length_scales))

    result = kernels.combine(partial(jnp.sum, axis=0), *inner)(x, y)
    expected = sum(kernel(x, y) for kernel in inner)

    np.testing.assert_allclose(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
