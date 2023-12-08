Boax: A Bayesian Optimization library for JAX.
===============================================

.. note::

   Boax is currently in early alpha and under active development!

Boax is a composable library of core components for Bayesian Optimization
that is **designed for flexibility**.

It comes with a low-level interfaces for:
   * **Fitting a Gaussian Process model to data** (`boax.prediction`): Bijectors, Kernels, Mean Functions, Gaussian Processes
   * **Constructing and optimizing acquisition functions** (`boax.optimization`) Acquisition Functions, Maximizers

----

Installation
^^^^^^^^^^^^

The latest release of Boax can be installed from
`PyPI <https://pypi.org/project/boax/>`_ using::

   pip install boax

You may also install directly from GitHub, using the following command. This
can be used to obtain the most recent version of Boax::

   pip install git+git://github.com/Lando-L/boax.git


.. toctree::
   :hidden:
   :maxdepth: 2

   guides/Getting_Started
   guides/index
   api_reference/index