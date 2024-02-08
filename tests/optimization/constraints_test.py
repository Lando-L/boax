import numpy as np
from absl.testing import absltest, parameterized
from jax import random
from jax import numpy as jnp

from boax.core import distributions
from boax.optimization import constraints
from boax.utils.functools import const


class ConstraintsTest(parameterized.TestCase):
  def test_range(self):
    key = random.key(0)
    n, q, d = 10, 1, 1

    loc, scale = random.uniform(key, (2, n, q))

    model = const(distributions.normal.normal(loc, scale))
    lower = 0.2
    upper = 0.8
    candidates = jnp.empty((n, q, d))

    le = constraints.less_or_equal(model, lower)(candidates)
    lle = constraints.log_less_or_equal(model, lower)(candidates)

    ge = constraints.greater_or_equal(model, upper)(candidates)
    lge = constraints.log_greater_or_equal(model, upper)(candidates)

    np.testing.assert_allclose(jnp.log(le), lle, atol=1e-4)
    np.testing.assert_allclose(jnp.log(ge), lge, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
