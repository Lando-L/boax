``boax.prediction`` module
============================

.. currentmodule:: boax.prediction

Implements functionalities to fit a surrogate model to data.


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
  multi_fidelity


Transformed Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  input_transformed
  outcome_transformed
  joined
  sampled
  scaled


boax.prediction.models.means
----------------------------

.. currentmodule:: boax.prediction.models.means


Mean Types
~~~~~~~~~~

.. autoclass:: Mean


Means
~~~~~


Common Means
^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  zero
  constant
  linear


boax.prediction.models.kernels
------------------------------

.. currentmodule:: boax.prediction.models.kernels


Kernel Types
~~~~~~~~~~~~

.. autoclass:: Kernel


Kernels
~~~~~~~


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


boax.prediction.models.likelihoods
----------------------------------

.. currentmodule:: boax.prediction.models.likelihoods


Likelihood Types
~~~~~~~~~~~~~~~~

.. autoclass:: Likelihood


Likelihoods
~~~~~~~~~~~


Common Likelihoods
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  beta
  gaussian


boax.prediction.objectives
--------------------------

.. currentmodule:: boax.prediction.objectives


Objective Types
~~~~~~~~~~~~~~~

.. autoclass:: Objective


Objectives
~~~~~~~~~~


Common Objectives
~~~~~~~~~~~~~~~~~

.. autosummary::
  :toctree: generated

  negative_log_likelihood


Transformed Objectives
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  penalized
