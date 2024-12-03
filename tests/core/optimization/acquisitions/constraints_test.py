from absl.testing import absltest, parameterized
from chex import assert_trees_all_close
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.core.optimization.acquisitions import constraints
from boax.utils.typing import PRNGKey


class ConstraintsTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'n': 10, 'q': 1, 'upper': 0.8, 'lower': 0.2},
    {'key': random.key(1), 'n': 15, 'q': 1, 'upper': 0.7, 'lower': 0.3},
    {'key': random.key(2), 'n': 10, 'q': 1, 'upper': 0.8, 'lower': 0.2},
  )
  def test_range(
    self, key: PRNGKey, n: int, q: int, upper: float, lower: float
  ):
    loc, scale = random.uniform(key, (2, n, q))
    preds = distributions.normal.normal(loc, scale)

    le = constraints.less_or_equal(upper)(preds)
    lle = constraints.log_less_or_equal(upper)(preds)

    ge = constraints.greater_or_equal(lower)(preds)
    lge = constraints.log_greater_or_equal(lower)(preds)

    assert_trees_all_close(jnp.log(le), lle, atol=1e-4)
    assert_trees_all_close(jnp.log(ge), lge, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
