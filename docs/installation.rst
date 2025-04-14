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

Install TORAX
-------------

The following may optionally be added to ~/.bashrc and will cause jax to
store compiled programs to the filesystem, avoiding recompilation in
some cases:

.. code-block:: console

  export JAX_COMPILATION_CACHE_DIR=<path of your choice, such as ~/jax_cache>
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0

For more information see :ref:`cache`.


Download and install the TORAX codebase via http:

.. code-block:: console

  git clone https://github.com/google-deepmind/torax.git

or ssh (ensure that you have the appropriate SSH key uploaded to github).

.. code-block:: console

  git clone git@github.com:google-deepmind/torax.git

Enter the TORAX directory and pip install the dependencies.

.. code-block:: console

  cd torax; pip install .

From within the top level directory where you `pip install` from, also set the
geometry data directory.

.. code-block:: console

  export TORAX_GEOMETRY_DIR="$PWD"/torax/data/third_party/geo

We recommend automating the variable export. If using bash, run:

.. code-block:: console

  echo export TORAX_GEOMETRY_DIR="$PWD"/torax/data/third_party/geo >> ~/.bashrc

The above command only needs to be run once on a given system.

TORAX uses the QLKNN_7_11 transport model by default. It can be overridden by
specifying a QLKNN model path through the `TORAX_QLKNN_MODEL_PATH`
environment variable. To use the default transport model (recommended), keep the
`TORAX_QLKNN_MODEL_PATH` environment variable empty. Previous versions of
TORAX required the environment variable to be set. If you set this variable in
a previous TORAX installation, make sure you do not define it in your
`~/.bashrc`. You can check if the variable is defined by running:

.. code-block:: console

  echo $TORAX_QLKNN_MODEL_PATH

If the variable is defined, you can clear it by running:

.. code-block:: console

  unset TORAX_QLKNN_MODEL_PATH

For an alternative transport model, see :ref:`install_qlknn_hyper`.

.. _dev_install:

(Optional) Install TORAX in development mode
--------------------------------------------

**Recommended** for developers. Instead of the above, install optional dependencies
for (parallel) pytest and documentation generation. Also install in editable mode to
not require reinstallation for every change.

.. code-block:: console

  cd torax; pip install -e .[dev]

.. _dev_install:

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

To use this transport model, you need to set the environment variable
`TORAX_QLKNN_MODEL_PATH` to the path of the cloned repository.

.. code-block:: console

  export TORAX_QLKNN_MODEL_PATH="$PWD"/qlknn-hyper
