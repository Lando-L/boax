``boax.optimization`` module
=============================

.. currentmodule:: boax.optimization

Implements functionalities to construct and optimize acquisition functions.


boax.optimization.acquisitions
------------------------------

.. currentmodule:: boax.optimization.acquisitions


Acquisition Types
^^^^^^^^^^^^^^^^^

.. autoclass:: Acquisition
    :members:


Acquisition Functions
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  log_expected_improvement
  log_probability_of_improvement
  upper_confidence_bound


boax.optimization.maximizers
----------------------------

.. currentmodule:: boax.optimization.maximizers


Maximizer Types
^^^^^^^^^^^^^^^

.. autoclass:: Maximizer
    :members:


Maximizers
^^^^^^^^^^

.. autosummary::
  :toctree: generated

  bfgs