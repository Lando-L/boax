``boax.prediction`` module
============================

.. currentmodule:: boax.prediction

Implements functionalities to fit a surrogate model to data.


boax.prediction
---------------

Construction Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  construct


boax.prediction.kernels
-----------------------

.. currentmodule:: boax.prediction.kernels


Kernel Types
~~~~~~~~~~~~

.. autoclass:: Kernel


Kernel Functions
~~~~~~~~~~~~~~~~


Radial Basis Kernels
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  rbf


Matérn Kernels
^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  matern_one_half
  matern_three_halves
  matern_five_halves


Periodic Kernels
^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  periodic


Transformed Kernels
^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  scaled
  additive
  product
  linear_truncated


boax.prediction.means
---------------------

.. currentmodule:: boax.prediction.means


Mean Types
~~~~~~~~~~

.. autoclass:: Mean


Mean Functions
~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  zero
  constant
  linear


boax.prediction.models
----------------------

.. currentmodule:: boax.prediction.models


Model Types
~~~~~~~~~~~

.. autoclass:: Model


Models
~~~~~~


Gaussian Process Models
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  gaussian_process
  gaussian_process_regression
  multi_fidelity_regression


Transformed Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  input_transformed
  outcome_transformed
  joined
  sampled
