``boax.prediction`` module
============================

.. currentmodule:: boax.prediction

Implements functionalities to fit a surrogate model to data.


boax.prediction.kernels
------------------------------

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


Mater Kernels
^^^^^^^^^^^^^

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


Transformed Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  sampled
