``boax.prediction`` module
===========================

.. currentmodule:: boax.prediction

Implements functionalities to fit Gaussian Process models to data.


boax.prediction.bijectors
-------------------------

.. currentmodule:: boax.prediction.bijectors


Bijector Types
^^^^^^^^^^^^^^

.. autoclass:: Bijector
    :members:

.. autoclass:: BijectorForwardFn
    :members:

.. autoclass:: BijectorInverseFn
    :members:


Bijectors
^^^^^^^^^

.. autosummary::
  :toctree: generated

  exp
  identity
  log
  scale
  shift
  softplus
  chain


boax.prediction.kernels
-----------------------

.. currentmodule:: boax.prediction.kernels


Kernel Types
^^^^^^^^^^^^

.. autoclass:: Kernel
    :members:


Kernels
^^^^^^^

.. autosummary::
  :toctree: generated

  rbf
  matern_one_half
  matern_three_halves
  matern_five_halves
  periodic
  combine
  scale


boax.prediction.means
----------------------

.. currentmodule:: boax.prediction.means


Mean Types
^^^^^^^^^^

.. autoclass:: Mean
    :members:


Mean Functions
^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  zero
  const
  linear


boax.prediction.processes
--------------------------

.. currentmodule:: boax.prediction.processes


Process Types
^^^^^^^^^^^^^

.. autoclass:: Process
    :members:

.. autoclass:: PriorFn
    :members:

.. autoclass:: PosteriorFn
    :members:

.. autosummary::
  :toctree: generated

  gaussian