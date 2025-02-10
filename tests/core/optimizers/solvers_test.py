from functools import partial

from absl.testing import absltest, parameterized
from chex import assert_shape
from jax import numpy as jnp
from jax import random

from boax.core import distributions, optimizers, samplers
from boax.utils.typing import PRNGKey


class SolversTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'num_restarts': 5, 'q': 3, 'd': 1},
    {'key': random.key(1), 'num_restarts': 5, 'q': 1, 'd': 3},
    {'key': random.key(2), 'num_restarts': 5, 'q': 3, 'd': 3},
  )
  def test_scipy(
    self,
    key: PRNGKey,
    num_restarts: int,
    q: int,
    d: int,
  ):
    fun = partial(jnp.mean, axis=(-2, -1))

    candidates = samplers.halton_uniform(key, (num_restarts, q))(
      distributions.uniform.uniform(jnp.zeros((d,)), jnp.ones((d,)))
    )

    solver = optimizers.solvers.scipy(jnp.array([[0, 1]]))

    next_candidates, values = solver(fun, candidates)

    assert_shape(next_candidates, (num_restarts, q, d))
    assert_shape(values, (num_restarts,))


if __name__ == '__main__':
  absltest.main()
