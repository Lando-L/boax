``boax.optimization`` module
============================

.. currentmodule:: boax.optimization

Implements functionalities to construct and optimize acquisition functions.


boax.optimization
-----------------

Construction Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  construct
  construct_constrained
  construct_log_constrained


boax.optimization.acquisitions
------------------------------

.. currentmodule:: boax.optimization.acquisitions


Acquisition Types
~~~~~~~~~~~~~~~~~

.. autoclass:: Acquisition


Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~


Analytic Acquisition Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  probability_of_improvement
  log_probability_of_improvement
  expected_improvement
  log_expected_improvement
  upper_confidence_bound
  posterior_mean
  posterior_scale


Monte Carlo Acquisition Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  q_probability_of_improvement
  q_expected_improvement
  q_probability_of_improvement
  q_knowledge_gradient


boax.optimization.constraints
------------------------------

.. currentmodule:: boax.optimization.constraints


Constraint Types
~~~~~~~~~~~~~~~~

.. autoclass:: Constraint


Constraint Functions
~~~~~~~~~~~~~~~~~~~~


Unequality Constraint Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Optimizer Functions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  bfgs
