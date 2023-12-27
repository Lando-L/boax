import numpy as np
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.optimization import constraints
from boax.utils.functools import const


class ConstraintsTest(parameterized.TestCase):
  def test_range(self):
    key1, key2 = random.split(random.key(0))

    loc = random.uniform(key1, (10,))
    cov = random.uniform(key2, (10,)) * jnp.identity(10)
    model = const((loc, cov))
    lower = 0.2
    upper = 0.8
    candidates = jnp.empty(())

    le = constraints.less_or_equal(model, lower)(candidates)
    lle = constraints.log_less_or_equal(model, lower)(candidates)

    ge = constraints.greater_or_equal(model, upper)(candidates)
    lge = constraints.log_greater_or_equal(model, upper)(candidates)

    np.testing.assert_allclose(jnp.log(le), lle, atol=1e-4)
    np.testing.assert_allclose(jnp.log(ge), lge, atol=1e-4)


if __name__ == '__main__':
  absltest.main()
