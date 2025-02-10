``boax.core.optimizers`` module
===============================

.. currentmodule:: boax.core.optimizers

Implements functionalities to optimize acquisition functions.


boax.core.optimizers
--------------------

.. currentmodule:: boax.core.optimizers


Optimizer Types
~~~~~~~~~~~~~~~

.. autoclass:: Optimizer


Optimizers
~~~~~~~~~~


Batch Optimizers
^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  batch


Sequential Optimizers
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  sequential


boax.core.optimizers.initializers
----------------------------------------------

.. currentmodule:: boax.core.optimizers.initializers


Initializer Types
~~~~~~~~~~~~~~~~~

.. autoclass:: Initializer


Initializers
~~~~~~~~~~~~


Batch Initializers
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  q_batch
  q_batch_nonnegative


boax.core.optimizers.solvers
-----------------------------------------

.. currentmodule:: boax.core.optimizers.solvers


Solver Types
~~~~~~~~~~~~

.. autoclass:: Solver


Solvers
~~~~~~~


Non-Linear Solvers
^^^^^^^^^^^^^^^^^^

.. autosummary::
  :toctree: generated

  scipy
