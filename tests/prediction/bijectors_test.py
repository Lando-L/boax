import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from bojax._src.prediction.bijectors.alias import (
  exp,
  identity,
  log,
  scale,
  shift,
  softplus,
)
from bojax._src.prediction.bijectors.transformed import chain


def assert_bijector(bijector, value, expected):
  forward = bijector.forward(value)
  inverse = bijector.inverse(forward)

  np.testing.assert_allclose(forward, expected, atol=1e-4)
  np.testing.assert_allclose(inverse, value, atol=1e-4)


class BijectorsTest(parameterized.TestCase):
  def test_identity(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(identity(), value, value)

  def test_exp(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(exp(), value, jnp.exp(value))

  def test_log(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(log(), value, jnp.log(value))

  def test_softplus(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(softplus(), value, jnp.log(1 + jnp.exp(value)))

  def test_scale(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(scale(jnp.array(2.0)), value, 2 * value)

  def test_shift(self):
    value = random.uniform(random.key(0), shape=(10,))
    assert_bijector(shift(jnp.array(2.0)), value, 2 + value)

  def test_chain(self):
    value = random.uniform(random.key(0), shape=(10,))
    bijector = chain(shift(jnp.array(-5.0)), scale(jnp.array(2.0)))
    assert_bijector(bijector, value, 2 * value - 5)


if __name__ == '__main__':
  absltest.main()
