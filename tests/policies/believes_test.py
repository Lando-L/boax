from absl.testing import absltest, parameterized
from chex import assert_equal, assert_shape, assert_trees_all_close
from jax import numpy as jnp

from boax.policies import believes


class BelievesTest(parameterized.TestCase):
  @parameterized.parameters(
    {'num_variants': 5, 'variant': 2, 'reward': 10},
    {'num_variants': 5, 'variant': 3, 'reward': 5},
    {'num_variants': 5, 'variant': 4, 'reward': 0},
  )
  def test_action_value(self, num_variants: int, variant: int, reward: float):
    belief = believes.continuous(num_variants)
    init = belief.init()
    updated = belief.update(init, variant, reward)
    idx, score = belief.best(updated)

    assert_shape(init.n, (num_variants,))
    assert_shape(init.q, (num_variants,))
    assert_trees_all_close(init.n, jnp.ones(num_variants, dtype=jnp.int32))
    assert_trees_all_close(init.q, jnp.ones(num_variants, dtype=jnp.float32))

    assert_shape(updated.n, (num_variants,))
    assert_shape(updated.q, (num_variants,))
    assert_trees_all_close(
      updated.n, jnp.ones(num_variants, dtype=jnp.int32).at[variant].add(1)
    )
    assert_trees_all_close(
      updated.q,
      jnp.ones(num_variants, dtype=jnp.float32).at[variant].set(reward),
    )

    assert_equal(idx, variant)
    assert_equal(score, updated.q[idx])

  @parameterized.parameters(
    {'num_variants': 5, 'variant': 2, 'reward': True},
    {'num_variants': 5, 'variant': 3, 'reward': False},
    {'num_variants': 5, 'variant': 4, 'reward': True},
  )
  def test_beta(self, num_variants: int, variant: int, reward: bool):
    belief = believes.binary(num_variants)
    init = belief.init()
    updated = belief.update(init, variant, reward)
    idx, score = belief.best(updated)

    assert_shape(init.a, (num_variants,))
    assert_shape(init.b, (num_variants,))
    assert_trees_all_close(init.a, jnp.ones(num_variants, dtype=jnp.int32))
    assert_trees_all_close(init.b, jnp.ones(num_variants, dtype=jnp.int32))

    assert_shape(updated.a, (num_variants,))
    assert_shape(updated.b, (num_variants,))

    if reward:
      assert_trees_all_close(
        updated.a, jnp.ones(num_variants, dtype=jnp.int32).at[variant].add(1)
      )
      assert_trees_all_close(updated.b, jnp.ones(num_variants, dtype=jnp.int32))

      assert_equal(idx, variant)
      assert_equal(score, 2)

    else:
      assert_trees_all_close(updated.a, jnp.ones(num_variants, dtype=jnp.int32))
      assert_trees_all_close(
        updated.b, jnp.ones(num_variants, dtype=jnp.int32).at[variant].add(1)
      )

      assert_equal(idx, 0)
      assert_equal(score, 1)


if __name__ == '__main__':
  absltest.main()
