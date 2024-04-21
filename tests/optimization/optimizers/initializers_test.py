from operator import itemgetter

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import random

from boax.core import samplers
from boax.optimization import optimizers


class InitializersTest(parameterized.TestCase):
  def test_q_batch(self):
    key = random.key(0)

    fun = itemgetter((..., 0, 0))
    sampler = samplers.halton_uniform()
    s, n, q, d = 10, 5, 3, 1

    initializer = optimizers.initializers.q_batch(
      fun,
      sampler,
      q,
      s,
      n,
    )

    result = initializer(key)

    assert_shape(result, (n, q, d))

  def test_q_nonnegative(self):
    key = random.key(0)

    fun = itemgetter((..., 0, 0))
    sampler = samplers.halton_uniform()
    s, n, q, d = 10, 5, 3, 1

    initializer = optimizers.initializers.q_batch_nonnegative(
      fun,
      sampler,
      q,
      s,
      n,
    )

    result = initializer(key)

    assert_shape(result, (n, q, d))


if __name__ == '__main__':
  absltest.main()
