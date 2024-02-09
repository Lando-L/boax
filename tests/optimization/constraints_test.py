import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.optimization import constraints


class ConstraintsTest(parameterized.TestCase):
  def test_range(self):
    key = random.key(0)
    n, q = 10, 1

    loc, scale = random.uniform(key, (2, n, q))

    lower = 0.2
    upper = 0.8
    preds = distributions.normal.normal(loc, scale)

    le = constraints.less_or_equal(lower)(preds)
    lle = constraints.log_less_or_equal(lower)(preds)

    ge = constraints.greater_or_equal(upper)(preds)
    lge = constraints.log_greater_or_equal(upper)(preds)

    np.testing.assert_allclose(jnp.log(le), lle, atol=1e-4)
    np.testing.assert_allclose(jnp.log(ge), lge, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
