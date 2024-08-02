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

Download QLKNN dependencies:

.. code-block:: console

  git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git

.. code-block:: console

  export TORAX_QLKNN_MODEL_PATH="$PWD"/qlknn-hyper

It is recommended to automate the environment variable export. For example, if using bash, run:

.. code-block:: console

  echo export TORAX_QLKNN_MODEL_PATH="$PWD"/qlknn-hyper >> ~/.bashrc

The above command only needs to be run once on a given system.

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

Optional: Install additional GPU support for JAX if your machine has a GPU:
https://jax.readthedocs.io/en/latest/installation.html#supported-platforms
