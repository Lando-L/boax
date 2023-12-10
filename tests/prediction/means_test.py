import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.prediction.means import constant, linear, zero


class MeansTest(parameterized.TestCase):
  def test_zero(self):
    value = jnp.empty((10,))

    result = zero()(value)
    expected = jnp.zeros(())

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_constant(self):
    x = jnp.array(2.0)
    value = jnp.empty((10,))

    result = constant(x)(value)
    expected = x

    np.testing.assert_allclose(result, expected, atol=1e-4)

  def test_linear(self):
    scale = jnp.array(2.0)
    bias = jnp.array(5.0)

    value = random.uniform(random.key(0), shape=(10,))

    result = linear(scale, bias)(value)
    expected = scale * value + bias

    np.testing.assert_allclose(result, expected, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
