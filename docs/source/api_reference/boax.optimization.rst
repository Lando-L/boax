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


boax.optimization.maximizers
----------------------------

.. currentmodule:: boax.optimization.maximizers


Maximizer Types
~~~~~~~~~~~~~~~

.. autoclass:: Maximizer


Maximizers
~~~~~~~~~~

.. autosummary::
  :toctree: generated

  bfgs


boax.optimization.samplers
--------------------------

.. currentmodule:: boax.optimization.samplers


Sampler Types
~~~~~~~~~~~~~

.. autoclass:: Sampler


Samplers
~~~~~~~~


Quasi-Random Samplers
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  halton_normal
  halton_uniform
