import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.prediction.kernels import (
  combine,
  matern_five_halves,
  matern_one_half,
  matern_three_halves,
  rbf,
  scale,
)


class KernelsTest(parameterized.TestCase):
  def test_rbf(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    result = rbf(length_scale)(x, y)
    expected = jnp.exp(-jnp.linalg.norm((x - y) ** 2) / (2 * length_scale**2))

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_one_half(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)

    result = matern_one_half(length_scale)(x, y)
    expected = jnp.exp(-jnp.linalg.norm(x - y) / length_scale)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_three_halves(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(3) * jnp.linalg.norm(x - y) / length_scale

    result = matern_three_halves(length_scale)(x, y)
    expected = (1 + z) * jnp.exp(-z)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_matern_five_halves(self):
    key1, key2 = random.split(random.key(0))
    length_scale = 0.2

    x = random.uniform(key1, (10,), minval=-1.0, maxval=1.0)
    y = random.uniform(key2, (10,), minval=-1.0, maxval=1.0)
    z = jnp.sqrt(5) * jnp.linalg.norm(x - y) / length_scale

    result = matern_five_halves(length_scale)(x, y)
    expected = (1 + z + z**2 / 3) * jnp.exp(-z)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_scale(self):
    key1, key2 = random.split(random.key(0))
    amplitude = 5
    length_scale = 0.2

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    inner = rbf(length_scale)

    result = scale(amplitude, inner)(x, y)
    expected = amplitude * inner(x, y)

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_combine(self):
    key1, key2 = random.split(random.key(0))
    length_scales = [0.2, 0.3, 0.4]

    x = random.uniform(key1, shape=(10,))
    y = random.uniform(key2, shape=(10,))

    inner = list(map(rbf, length_scales))

    result = combine(jnp.sum, *inner)(x, y)
    expected = sum(kernel(x, y) for kernel in inner)

    np.testing.assert_allclose(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
