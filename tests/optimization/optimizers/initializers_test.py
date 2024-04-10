from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.optimization import optimizers


class InitializersTest(parameterized.TestCase):
  def test_q_batch(self):
    key1, key2 = random.split(random.key(0))
    x = random.uniform(key1, (100, 1))
    y = x[..., 0]
    num_restarts = 10

    result = optimizers.initializers.q_batch()(
      key2,
      x,
      y,
      num_restarts,
    )

    assert_shape(result, (10, 1))

  def test_q_nonnegative(self):
    key1, key2 = random.split(random.key(0))
    x = random.uniform(key1, (100, 1))
    y = x[..., 0]
    num_restarts = 10

    result = optimizers.initializers.q_batch_nonnegative()(
      key2,
      x,
      y,
      num_restarts,
    )

    assert_shape(result, (10, 1))


if __name__ == '__main__':
  absltest.main()
