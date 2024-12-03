``boax.core.optimization`` module
=================================

.. currentmodule:: boax.core.optimization

Implements functionalities to construct and optimize acquisition functions.


boax.core.optimization.acquisitions
-----------------------------------

.. currentmodule:: boax.core.optimization.acquisitions


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


boax.core.optimization.acquisitions.transformations
---------------------------------------------------

.. currentmodule:: boax.core.optimization.acquisitions.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Acquisitions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  constrained
  log_constrained


boax.core.optimization.acquisitions.constraints
-----------------------------------------------

.. currentmodule:: boax.core.optimization.acquisitions.constraints


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


boax.core.optimization.optimizers
---------------------------------

.. currentmodule:: boax.core.optimization.optimizers


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


boax.core.optimization.optimizers.initializers
----------------------------------------------

.. currentmodule:: boax.core.optimization.optimizers.initializers


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


boax.core.optimization.optimizers.solvers
-----------------------------------------

.. currentmodule:: boax.core.optimization.optimizers.solvers


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


boax.core.optimization.policies
-------------------------------

.. currentmodule:: boax.core.optimization.policies


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


boax.core.optimization.policies.believes
----------------------------------------

.. currentmodule:: boax.core.optimization.policies.believes


Belief Types
~~~~~~~~~~~~

.. autoclass:: Belief


Believes
~~~~~~~~

.. autosummary::
  :toctree: generated

  binary
  continuous