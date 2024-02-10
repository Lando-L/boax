``boax.core`` module
============================

.. currentmodule:: boax.core

Implements core functionalities.


boax.core.distributions
-----------------------

.. currentmodule:: boax.core.distributions


Distributions
~~~~~~~~~~~~~


Beta Distribution
^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.distributions.beta

.. autoclass:: Beta

.. autosummary::
  :toctree: generated

  beta
  pdf
  cdf
  sf
  logpdf
  logcdf
  logsf


Multivariate Normal Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.distributions.multivariate_normal

.. autoclass:: MultivariateNormal

.. autosummary::
  :toctree: generated

  multivariate_normal
  as_normal
  pdf
  logpdf


Normal Distribution
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.distributions.normal

.. autoclass:: Normal

.. autosummary::
  :toctree: generated

  normal
  pdf
  cdf
  sf
  logpdf
  logcdf
  logsf


Uniform Distribution
^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: boax.core.distributions.uniform

.. autoclass:: Uniform

.. autosummary::
  :toctree: generated

  uniform
  pdf
  logpdf


boax.core.samplers
--------------------------

.. currentmodule:: boax.core.samplers


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
