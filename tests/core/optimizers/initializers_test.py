from functools import partial

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core import distributions, optimizers, samplers
from boax.utils.typing import PRNGKey


class InitializersTest(parameterized.TestCase):
  @parameterized.parameters(
    {
      'key': random.key(0),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 1,
      'd': 3,
    },
    {
      'key': random.key(2),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 3,
      'd': 3,
    },
  )
  def test_q_batch(
    self,
    key: PRNGKey,
    raw_samples: int,
    num_restarts: int,
    q: int,
    d: int,
  ):
    key1, key2 = random.split(key)

    fun = partial(jnp.mean, axis=(-2, -1))

    samples = samplers.halton_uniform(key1, (raw_samples, q))(
      distributions.uniform.uniform(jnp.zeros((d,)), jnp.ones((d,)))
    )

    initializer = optimizers.initializers.q_batch(samples, num_restarts)

    result = initializer(fun, key2)

    assert_shape(result, (num_restarts, q, d))

  @parameterized.parameters(
    {
      'key': random.key(0),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 3,
      'd': 1,
    },
    {
      'key': random.key(1),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 1,
      'd': 3,
    },
    {
      'key': random.key(2),
      'raw_samples': 10,
      'num_restarts': 5,
      'q': 3,
      'd': 3,
    },
  )
  def test_q_nonnegative(
    self,
    key: PRNGKey,
    raw_samples: int,
    num_restarts: int,
    q: int,
    d: int,
  ):
    key1, key2 = random.split(key)

    fun = partial(jnp.mean, axis=(-2, -1))

    samples = samplers.halton_uniform(key1, (raw_samples, q))(
      distributions.uniform.uniform(jnp.zeros((d,)), jnp.ones((d,)))
    )

    initializer = optimizers.initializers.q_batch_nonnegative(
      samples, num_restarts
    )

    result = initializer(fun, key2)

    assert_shape(result, (num_restarts, q, d))


if __name__ == '__main__':
  absltest.main()
