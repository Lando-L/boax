``boax.policies`` module
========================

.. currentmodule:: boax.policies

Implements functionalities to construct policy functions.


boax.policies
-------------------------------

.. currentmodule:: boax.policies


Policy Types
~~~~~~~~~~~~

.. autoclass:: Policy


Policies
~~~~~~~~


Action Value Policies
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  epsilon_greedy
  boltzmann
  upper_confidence_bound


Beta Policies
^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  thompson_sampling


boax.policies.believes
----------------------

.. currentmodule:: boax.policies.believes


Belief Types
~~~~~~~~~~~~

.. autoclass:: Belief


Believes
~~~~~~~~

.. autosummary::
  :toctree: generated

  binary
  continuous
