``boax.acquisitions`` module
============================

.. currentmodule:: boax.acquisitions

Implements functionalities to construct acquisition functions.


boax.acquisitions
-----------------

.. currentmodule:: boax.acquisitions


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


Monte Carlo Acquisitions
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  q_probability_of_improvement
  q_expected_improvement
  q_upper_confidence_bound


boax.acquisitions.surrogates
----------------------------

.. currentmodule:: boax.acquisitions.surrogates


Surrogate Types
~~~~~~~~~~~~~~~

.. autoclass:: Surrogate


Surrogates
~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  single_task_gaussian_process
