.. include:: links.rst

.. _installation:

Installation Guide
##################

Requirements
============

Install Python 3.11 or greater.

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
-------------------------

If you plan to develop TORAX, we recommend installing from source. See
:ref:`contribution_tips` for an installation guide.

JAX environment variables
-------------------------

The following may optionally be added to ``~/.bashrc`` and will cause jax to
store compiled functions to the filesystem, avoiding recompilation in
some cases:

.. code-block:: console

  export JAX_COMPILATION_CACHE_DIR=<path of your choice, such as ~/jax_cache>
  export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
  export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0

For more information see :ref:`using_jax`.

(Optional) GPU support
----------------------

Install additional GPU support for JAX if your machine has a GPU:
https://jax.readthedocs.io/en/latest/installation.html#supported-platforms


(Optional) Install QLKNN-hyper
------------------------------

TORAX uses the |qlknn_7_11| transport model by default, an upgrade to the
QLKNN-hyper-10D (QLKNN10D) neural network surrogate model of QuaLiKiz
|qlknn10d|. QLKNN_7_11 is based on QuaLiKiz 2.8.1 which has an improved
collision operator compared to the QLKNN10D training set. QLKNN_7_11 training
data also includes impurity density gradients as an input feature and has better
coverage of the near-LCFS region compared to QLKNN-hyper-10D.

However, certain use-cases may require the use of QLKNN10D, such as for
comparative studies with other simulators. To install QLKNN10D, first
download the QLKNN dependencies at a location of your choice:

.. code-block:: console

  git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git

To use QLKNN10D , you then need to set ``model_path`` in the
``transport`` section of your TORAX config to the path of the cloned repository.
See :ref:`configuration` for more details.


.. _tglf_installation:

(Optional) Install TGLF wrapper
-------------------------------

TORAX supports running the TGLF quasilinear turbulent transport model directly
in memory via an ``f2py`` extension module (``tglf2py``). If you want to use
TORAX with TGLF, you can compile and install this wrapper
manually after installing core TORAX.

Prerequisites
^^^^^^^^^^^^^

Ensure you have a Fortran compiler (such as ``gfortran``), LAPACK development
libraries, OpenMP libraries, and required Python build utilities installed:

.. code-block:: console

  pip install meson ninja

Isolate the Fortran Sources
^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, ensure you have access to the GACODE repository (available through
General Atomics at https://gacode.io). After obtaining or cloning GACODE,
export the ``GACODE_ROOT`` environment variable to point to your repository
root:

.. code-block:: console

  export GACODE_ROOT=/path/to/your/gacode

Next, run the following command to isolate the required TGLF Fortran sources
while excluding MPI, driver, and unused variants (noting that GACODE filenames
like ``tglf_run.F90`` use uppercase extensions):

.. code-block:: console

  TGLF_SRCS=$(find "$GACODE_ROOT/tglf/src" -maxdepth 1 -iname "*.f90" \
    | grep -vE \
    -e '/tglf\.f90$' \
    -e '/tglf_mpi\.f90$' \
    -e '/tglf_init_mpi\.f90$' \
    -e '/tglf_driver\.f90$' \
    -e '/tglf_driver_mpi\.f90$' \
    -e '/tglf_TM_mpi\.f90$' \
    -e '/tglf_TM_driver\.f90$' \
    -e '/tglf_nn_TM\.fann\.f90$' \
    -e '\.f90_save$' \
    -e '\.cray\.f90$' \
    -e '\.ieee\.f90$')

Compile the Extension Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate directly to your installed package directory inside
``torax/_src/transport_model/tglf`` and run ``f2py``. Modern ``f2py`` will
parse the module name straight from your ``.pyf`` signature file and invoke
Meson under the hood:

.. code-block:: console

  TORAX_DIR=$(python -c \
    "import torax, os; print(os.path.dirname(torax.__file__))")
  cd "$TORAX_DIR/_src/transport_model/tglf"

  FLAGS="-fdefault-real-8 -fdefault-double-8 -ffree-line-length-256 "
  FLAGS+="-cpp -fPIC -fopenmp -Ofast -std=f2018 -fall-intrinsics -Wall -W "
  FLAGS+="-fimplicit-none -fmax-stack-var-size=65536 -frecursive "
  FLAGS+="-falign-commons"

  python -m numpy.f2py -c tglf2py.pyf \
    $TGLF_SRCS \
    --f90flags="$FLAGS" \
    --dep lapack \
    -lgomp

Once completed, a shared library binary (e.g., ``tglf2py_lib*.so`` or
``.pyd``) will be generated directly inside
``torax/_src/transport_model/tglf/``, which will be used when
``tglf_run()`` is called.

Verify the Installation
^^^^^^^^^^^^^^^^^^^^^^^

You can check if the wrapper installed correctly by executing its unit test
directly with Python:

.. code-block:: console

  python torax/torax/_src/transport_model/tglf/tests/tglf2py_test.py
