from operator import itemgetter

import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.prediction.kernels import rbf
from boax.prediction.processes import gaussian


class ProcessesTest(parameterized.TestCase):
  def test_gaussian_prior(self):
    length_scale = jnp.array(0.2)
    noise = jnp.empty(())
    jitter = jnp.array(1e-3)

    xs = random.uniform(random.key(0), shape=(10, 1), minval=-1, maxval=1)

    mean = itemgetter((..., 0))
    kernel = rbf(length_scale)

    process = gaussian(mean, kernel, noise, jitter)

    result = process.prior(xs)
    expected = xs[..., 0], kernel(xs, xs) + jitter * jnp.identity(10)

    np.testing.assert_allclose(result[0], expected[0], atol=1e-4)
    np.testing.assert_allclose(result[1], expected[1], atol=1e-4)

  # def test_gaussian_posterior(self):
  #   key1, key2, key3 = random.split(random.key(0), 3)

  #   length_scale = jnp.array(0.2)
  #   noise = jnp.array(0.5)
  #   jitter = jnp.array(1e-3)

  #   xs = random.uniform(key1, shape=(10, 1), minval=-1, maxval=1)
  #   x_train = random.uniform(key2, shape=(3, 1), minval=-1, maxval=1)
  #   y_train = random.uniform(key3, shape=(3,), minval=-1, maxval=1)

  #   mean = itemgetter((..., 0))
  #   kernel = vmap(vmap(rbf(length_scale), in_axes=(None, 0)), in_axes=(0, None))

  #   result = posterior(x_train, y_train, mean, kernel, noise, jitter)(xs)
  #   expected = xs[..., 0], kernel(xs, xs) + (noise + jitter) * jnp.identity(10)

  #   np.testing.assert_allclose(result[0], expected[0], atol=1e-4)
  #   np.testing.assert_allclose(result[1], expected[1], atol=1e-4)


if __name__ == '__main__':
  absltest.main()
