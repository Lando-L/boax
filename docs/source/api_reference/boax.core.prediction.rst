``boax.core.prediction`` module
===============================

.. currentmodule:: boax.core.prediction

Implements functionalities to fit a surrogate model to data.


boax.core.prediction.models
---------------------------

.. currentmodule:: boax.core.prediction.models


Model Types
~~~~~~~~~~~

.. autoclass:: Model


Models
~~~~~~


Gaussian Process Models
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.prediction.models.gaussian_process

.. autosummary::
  :toctree: generated

  exact
  multi_fidelity


boax.core.prediction.models.transformations
-------------------------------------------

.. currentmodule:: boax.core.prediction.models.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  joined
  sampled
  transformed


boax.core.prediction.models.means
---------------------------------

.. currentmodule:: boax.core.prediction.models.means


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


boax.core.prediction.models.kernels
-----------------------------------

.. currentmodule:: boax.core.prediction.models.kernels


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


boax.core.prediction.models.kernels.transformations
---------------------------------------------------

.. currentmodule:: boax.core.prediction.models.kernels.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Kernels
^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  scaled
  additive
  product
  linear_truncated


boax.core.prediction.models.likelihoods
---------------------------------------

.. currentmodule:: boax.core.prediction.models.likelihoods


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


boax.core.prediction.objectives
-------------------------------

.. currentmodule:: boax.core.prediction.objectives


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


boax.core.prediction.objectives.transformations
-----------------------------------------------

.. currentmodule:: boax.core.prediction.objectives.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Objectives
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  penalized
