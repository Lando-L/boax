``boax.optimization`` module
============================

.. currentmodule:: boax.optimization

Implements functionalities to construct and optimize acquisition functions.


boax.optimization.acquisitions
------------------------------

.. currentmodule:: boax.optimization.acquisitions


Acquisition Types
~~~~~~~~~~~~~~~~~

.. autoclass:: Acquisition


Acquisitions
~~~~~~~~~~~~


Analytic Acquisitions
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  probability_of_improvement
  log_probability_of_improvement
  expected_improvement
  log_expected_improvement
  upper_confidence_bound
  posterior_mean
  posterior_scale


Monte Carlo Acquisitions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  q_probability_of_improvement
  q_expected_improvement
  q_upper_confidence_bound
  q_knowledge_gradient


Transformed Acquisitions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  constrained
  log_constrained


boax.optimization.acquisitions.constraints
------------------------------------------

.. currentmodule:: boax.optimization.acquisitions.constraints


Constraint Types
~~~~~~~~~~~~~~~~

.. autoclass:: Constraint


Constraints
~~~~~~~~~~~


Unequality Constraints
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  less_or_equal
  log_less_or_equal
  greater_or_equal
  log_greater_or_equal


boax.optimization.optimizers
----------------------------

.. currentmodule:: boax.optimization.optimizers


Optimizer Types
~~~~~~~~~~~~~~~

.. autoclass:: Optimizer


Optimizers
~~~~~~~~~~


Batch Optimizers
^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  batch


Sequential Optimizers
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  sequential


boax.optimization.optimizers.initializers
-----------------------------------------

.. currentmodule:: boax.optimization.optimizers.initializers


Initializer Types
~~~~~~~~~~~~~~~~~

.. autoclass:: Initializer


Initializers
~~~~~~~~~~~~


Batch Initializers
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  q_batch
  q_batch_nonnegative


boax.optimization.optimizers.solvers
------------------------------------

.. currentmodule:: boax.optimization.optimizers.solvers


Solver Types
~~~~~~~~~~~~

.. autoclass:: Solver


Solvers
~~~~~~~


Non-Linear Solvers
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  scipy


boax.optimization.policies
--------------------------

.. currentmodule:: boax.optimization.acquisitions


Policy Types
~~~~~~~~~~~~

.. autoclass:: Policy


Policies
~~~~~~~~


Action Value Policies
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  epsilon_greedy
  boltzmann
  upper_confidence_bound


Beta Policies
^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  thompson_sampling


boax.optimization.policies.evaluators
-------------------------------------

.. currentmodule:: boax.optimization.policies.evaluators


Evaluator Types
~~~~~~~~~~~~~~~

.. autoclass:: Evaluator


Evaluators
~~~~~~~~~~

.. autosummary::
  :toctree: generated

  action_values
  beta
