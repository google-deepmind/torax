.. _installation:

Installation Guide
##################

Requirements
============

Install Python 3.10 or greater.

Make sure that tkinter is installed:

.. code-block:: console

  sudo apt-get install python3-tk

.. _how_to_install:

How to install
==============

.. _prepare_virtualenv:

Prepare a virtual environment
-----------------------------

Install virtualenv (if not already installed):

.. code-block:: console

  pip install --upgrade pip
  pip install virtualenv

Create a code directory where you will install the virtual env and other TORAX
dependencies.

.. code-block:: console

  mkdir /path/to/torax_dir && cd "$_"

Where ``/path/to/torax_dir`` should be replaced by a path of your choice.

Create a TORAX virtual env:

.. code-block:: console

  python3 -m venv toraxvenv

Activate the virtual env:

.. code-block:: console

  source toraxvenv/bin/activate

It is convenient to set up an alias for the above command.

.. install_torax:

Install TORAX from PyPI
-----------------------

This is the simplest way to install TORAX. If you don't plan to do any
development work, this is the recommended method.

.. code-block:: console

  pip install torax



Install TORAX from Github
-----------------------

If you plan to develop TORAX, we recommend installing from source.

Download and install the TORAX codebase via http:

.. code-block:: console

  git clone https://github.com/google-deepmind/torax.git

or ssh (ensure that you have the appropriate SSH key uploaded to github).

.. code-block:: console

  git clone git@github.com:google-deepmind/torax.git

Enter the TORAX directory and pip install the dependencies.

.. code-block:: console

  cd torax; pip install -e .[dev]

TORAX uses the QLKNN_7_11 transport model by default.
For an alternative transport model, see :ref:`install_qlknn_hyper`.

The following may optionally be added to ~/.bashrc and will cause jax to
store compiled programs to the filesystem, avoiding recompilation in
some cases:

.. code-block:: console

  export JAX_COMPILATION_CACHE_DIR=<path of your choice, such as ~/jax_cache>
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0

For more information see :ref:`cache`.

(Optional) GPU support
-------------------

Install additional GPU support for JAX if your machine has a GPU:
https://jax.readthedocs.io/en/latest/installation.html#supported-platforms


.. _install_qlknn_hyper:

(Optional) Install QLKNN-hyper
-------------------

An alternative to QLKNN_7_11 is to use QLKNN-hyper-10D, also known as QLKNN10D
(`K.L. van de Plassche PoP 2020 <https://doi.org/10.1063/1.5134126>`_).
QLKNN_7_11 is based on QuaLiKiz 2.8.1 which has an improved collision operator
compared to the QLKNN10D training set. QLKNN_7_11 training data includes
impurity density gradients as an input feature and has better coverage of the
near-LCFS region compared to QLKNN-hyper-10D. However, it is still widely used
in other simulators, so it can be useful for comparative studies for instance.

Download QLKNN dependencies:

.. code-block:: console

  git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git

To use this transport model, you need to set ``model_path`` in the
``transport`` section of your TORAX config to the path of the cloned repository.
