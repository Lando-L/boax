from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core import distributions, samplers
from boax.utils.typing import PRNGKey, Shape


class SamplersTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_multivariate_normal(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    mean, variance = random.uniform(key1, shape=(2,) + b + (n,))
    cov = variance[..., jnp.newaxis] * jnp.broadcast_to(
      jnp.identity(n), b + (n, n)
    )
    mvn = distributions.multivariate_normal.multivariate_normal(mean, cov)

    result = samplers.multivariate_normal(key2, s)(mvn)

    assert_shape(result, s + b + (n,))

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_normal(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    normal = distributions.normal.normal(
      *random.uniform(key1, shape=(2,) + b + (n,))
    )

    result = samplers.normal(key2, s)(normal)

    assert_shape(result, s + b + (n,))

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_uniform(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    uniform = distributions.uniform.uniform(
      *random.uniform(key1, shape=(2,) + b + (n,))
    )

    result = samplers.uniform(key2, s)(uniform)

    assert_shape(result, s + b + (n,))

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_halton_multivariate_normal(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    mean, variance = random.uniform(key1, shape=(2,) + b + (n,))
    cov = variance[..., jnp.newaxis] * jnp.broadcast_to(
      jnp.identity(n), b + (n, n)
    )
    mvn = distributions.multivariate_normal.multivariate_normal(mean, cov)

    result = samplers.halton_multivariate_normal(key2, s)(mvn)

    assert_shape(result, s + b + (n,))

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_halton_normal(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    normal = distributions.normal.normal(
      *random.uniform(key1, shape=(2,) + b + (n,))
    )

    result = samplers.halton_normal(key2, s)(normal)

    assert_shape(result, s + b + (n,))

  @parameterized.parameters(
    {'key': random.key(0), 's': (1,), 'b': (), 'n': 3},
    {'key': random.key(1), 's': (4,), 'b': (), 'n': 10},
    {'key': random.key(2), 's': (4, 4), 'b': (5, 3), 'n': 3},
  )
  def test_halton_uniform(
    self,
    key: PRNGKey,
    s: Shape,
    b: Shape,
    n: int,
  ):
    key1, key2 = random.split(key)

    uniform = distributions.uniform.uniform(
      *random.uniform(key1, shape=(2,) + b + (n,))
    )

    result = samplers.halton_uniform(key2, s)(uniform)

    assert_shape(result, s + b + (n,))


if __name__ == '__main__':
  absltest.main()
