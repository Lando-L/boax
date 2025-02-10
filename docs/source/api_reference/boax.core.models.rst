``boax.core.models`` module
===========================

.. currentmodule:: boax.core.models

Implements functionalities to construct surrogate models.


boax.core.models
----------------

.. currentmodule:: boax.core.models


Model Types
~~~~~~~~~~~

.. autoclass:: Model


Models
~~~~~~


Gaussian Process Models
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.models.gaussian_process

.. autosummary::
  :toctree: generated

  exact


boax.core.models.transformations
--------------------------------

.. currentmodule:: boax.core.models.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Models
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  transformed
  joined


boax.core.models.means
----------------------

.. currentmodule:: boax.core.models.means


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


boax.core.models.kernels
------------------------

.. currentmodule:: boax.core.models.kernels


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


boax.core.models.kernels.transformations
----------------------------------------

.. currentmodule:: boax.core.models.kernels.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Kernels
^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  scaled
  additive
  product


boax.core.models.likelihoods
----------------------------

.. currentmodule:: boax.core.models.likelihoods


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


boax.core.objectives
-------------------------------

.. currentmodule:: boax.core.objectives


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


boax.core.objectives.transformations
-----------------------------------------------

.. currentmodule:: boax.core.objectives.transformations


Transformations
~~~~~~~~~~~~~~~


Transformed Objectives
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  penalized
