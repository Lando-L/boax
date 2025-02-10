from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax import random

from boax import policies
from boax.policies import believes
from boax.utils.typing import PRNGKey


class PoliciesTest(parameterized.TestCase):
  @parameterized.parameters(
    {'key': random.key(0), 'epsilon': 0.2, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'epsilon': 0.4, 'num_variants': 5, 'timestep': 2},
    {'key': random.key(2), 'epsilon': 0.6, 'num_variants': 5, 'timestep': 3},
  )
  def test_epsilon_greedy(
    self, key: PRNGKey, epsilon: float, num_variants: int, timestep: int
  ):
    params = believes.continuous(num_variants).init()

    variant = policies.epsilon_greedy(epsilon)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'tau': 0.3, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'tau': 0.6, 'num_variants': 5, 'timestep': 2},
    {'key': random.key(2), 'tau': 0.9, 'num_variants': 5, 'timestep': 3},
  )
  def test_boltzman(
    self, key: PRNGKey, tau: float, num_variants: int, timestep: int
  ):
    params = believes.continuous(num_variants).init()

    variant = policies.boltzmann(tau)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'confidence': 0.3, 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'confidence': 0.6, 'num_variants': 5, 'timestep': 2},
    {'key': random.key(2), 'confidence': 0.9, 'num_variants': 5, 'timestep': 3},
  )
  def test_upper_confidence_bound(
    self, key: PRNGKey, confidence: float, num_variants: int, timestep: int
  ):
    params = believes.continuous(num_variants).init()

    variant = policies.upper_confidence_bound(confidence)(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))

  @parameterized.parameters(
    {'key': random.key(0), 'num_variants': 5, 'timestep': 1},
    {'key': random.key(1), 'num_variants': 5, 'timestep': 2},
    {'key': random.key(2), 'num_variants': 5, 'timestep': 3},
  )
  def test_thompson_sampling(
    self, key: PRNGKey, num_variants: int, timestep: int
  ):
    params = believes.binary(num_variants).init()

    variant = policies.thompson_sampling()(params, timestep, key)

    self.assertTrue(jnp.all(variant < num_variants))


if __name__ == '__main__':
  absltest.main()
