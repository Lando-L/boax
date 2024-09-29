from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax.core import distributions
from boax.optimization import policies
from boax.optimization.policies import evaluators
from boax.utils.typing import PRNGKey


class PoliciesTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'epsilon': 0.2, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'epsilon': 0.4, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(2), 'epsilon': 0.6, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(3), 'epsilon': 0.8, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(4), 'epsilon': 1.0, 'num_variants': 5, 'timestep': 1},
  )
  def test_epsilon_greedy(
    self, key: PRNGKey, epsilon: float, num_variants: int, timestep: int
  ):
    params = evaluators.ActionValues(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.float32),
    )

    variant = policies.epsilon_greedy(epsilon)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'tau': 0.3, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'tau': 0.6, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(2), 'tau': 0.9, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(3), 'tau': 1.2, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(4), 'tau': 1.5, 'num_variants': 5, 'timestep': 1},
  )
  def test_boltzman(
    self, key: PRNGKey, tau: float, num_variants: int, timestep: int
  ):
    params = evaluators.ActionValues(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.float32),
    )

    variant = policies.boltzmann(tau)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'confidence': 0.3, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'confidence': 0.6, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(2), 'confidence': 0.9, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(3), 'confidence': 1.2, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(4), 'confidence': 1.5, 'num_variants': 5, 'timestep': 1},
  )
  def test_upper_confidence_bound(
    self, key: PRNGKey, confidence: float, num_variants: int, timestep: int
  ):
    params = evaluators.ActionValues(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.float32),
    )

    variant = policies.upper_confidence_bound(confidence)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'num_variants': 5, 'timestep': 1},
    {'key': random.key(2), 'num_variants': 5, 'timestep': 1},
    {'key': random.key(3), 'num_variants': 5, 'timestep': 1},
    {'key': random.key(4), 'num_variants': 5, 'timestep': 1},
  )
  def test_thompson_sampling(
    self, key: PRNGKey, num_variants: int, timestep: int
  ):
    params = distributions.beta.Beta(
      jnp.ones(num_variants, dtype=jnp.int32),
      jnp.ones(num_variants, dtype=jnp.int32),
    )

    variant = policies.thompson_sampling()(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))


if __name__ == '__main__':
  absltest.main()
