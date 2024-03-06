# What is TORAX?

TORAX is a differentiable tokamak core transport simulator aimed for fast and accurate forward modelling, pulse-design, trajectory optimization, and controller design workflows. TORAX is written in Python-JAX, with the following motivations:

- Open-source and extensible, aiding with flexible workflow coupling
- JAX provides auto-differentiation capabilities and code compilation for fast runtimes. Differentiability allows for gradient-based nonlinear PDE solvers for fast and accurate modelling, and for sensitivity analysis of simulation results to arbitrary parameter inputs, enabling applications such as trajectory optimization and data-driven parameter identification for semi-empirical models. Auto-differentiability allows for these applications to be easily extended with the addition of new physics models, or new parameter inputs, by avoiding the need to hand-derive Jacobians
- Python-JAX is a natural framework for the coupling of ML-surrogates of physics models

TORAX is in a pre-release phase with a basic physics feature set, including:

- Coupled PDEs of ion and electron heat transport, electron particle transport, and current diffusion
    - Finite-volume-method
    - Multiple solver options: linear with Pereverzev-Corrigan terms, nonlinear with Newton-Raphson, nonlinear with optimization using the jaxopt library
- Ohmic power, ion-electron heat exchange, fusion power, bootstrap current with the analytical Sauter model
- Time dependent boundary conditions and sources
- Coupling to the QLKNN10D [[van de Plassche et al, Phys. Plasmas 2020]](https://doi.org/10.1063/1.5134126) QuaLiKiz-neural-network surrogate for physics-based turbulent transport
- General geometry, provided via CHEASE equilibrium files
    - For testing and demonstration purposes, a single CHEASE equilibrium file is available in the data/geo directory. It corresponds to an ITER hybrid scenario equilibrium based on simulations in [[Citrin et al, Nucl. Fusion 2010]](https://doi.org/10.1088/0029-5515/50/11/115007), and was obtained from [PINT](https://gitlab.com/qualikiz-group/pyntegrated_model). A PINT license file is available in data/geo.

Additional heating and current drive sources can be provided by prescribed formulas, or user-provided analytical models.

Model implementation was verified through direct comparison of simulation outputs to the RAPTOR [[Felici et al, Plasma Phys. Control. Fusion 2012]](https://iopscience.iop.org/article/10.1088/0741-3335/54/2/025002) tokamak transport simulator.

This is not an officially supported Google product.

## Feature roadmap

Short term development plans include:

- Implementation of forward sensitivity calculations w.r.t. control inputs and parameters
- Implementation of persistent compilation cache for CPU
- Performance optimization and cleanup
- More extensive documentation and tutorials

Longer term desired features include:

- Sawtooth model (Porcelli + reconnection)
- Neoclassical tearing modes (modified Rutherford equation)
- Radiation sinks
    - Cyclotron radiation
    - Bremsstrahlung
    - Line radiation
- Neoclassical transport + multi-ion transport, with a focus on heavy impurities
- IMAS coupling
- Stationary-state solver
- Momentum transport

Contributions in line with the roadmap are welcome. In particular, TORAX is envisaged as a natural framework for coupling of various ML-surrogates of physics models. These could include surrogates for turbulent transport, neoclassical transport, line radiation, pedestal physics, and core-edge integration, MHD, among others.

# Installation guide

## Requirements

Install Python 3.10 or greater.

Make sure that tkinter is installed:

```shell
$ sudo apt-get install python3-tk
```

## How to install

Install virtualenv (if not already installed):

```shell
$ pip install --upgrade pip
$ pip install virtualenv
```

Create a code directory where you will install the virtual env and other TORAX
dependencies.

```shell
$ mkdir /path/to/torax_dir && cd "$_"
```

Create a TORAX virtual env:

```shell
$ python3 -m venv toraxvenv
```

Activate the virtual env:

```shell
$ source toraxvenv/bin/activate
```

Download and install QLKNN dependencies:

```shell
$ git clone https://gitlab.com/qualikiz-group/QLKNN-develop.git
$ pip install ./QLKNN-develop
$ git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git
$ export TORAX_QLKNN_MODEL_PATH="$PWD/qlknn-hyper"
```

Download and install the TORAX codebase:

```shell
$ pip install git+git://github.com/google-deepmind/torax.git
```

Optional: Install additional GPU support for JAX if your machine has a GPU:
https://jax.readthedocs.io/en/latest/installation.html#supported-platforms

## Running an example

The following command will run TORAX using the configuration file `tests/test_data/default_config.py`. TORAX configuration files overwrite the defaults in `config.py`. Comments in `config.py` provide a brief explanation of all configuration variables. More detailed documentation is on the roadmap.

```shell
$ python3 run_simulation_main.py \
$   --python_config='torax.tests.test_data.default_config'
```

Additional configuration is provided through flags which append the above run command, and environment variables:

### Set environment variables

Path to the QuaLiKiz-neural-network parameters

```shell
$ export TORAX_QLKNN_MODEL_PATH="<myqlknnmodelpath>"
```

Path to the geometry file directory

```shell
$ export TORAX_QLKNN_MODEL_PATH="<mygeodir>"
```

If true, error checking is enabled in internal routines. Used for debugging. Default is false since it is incompatible with the persistent compilation cache.

```shell
$ export TORAX_ERRORS_ENABLED=<True/False>
```

If false, JAX does not compile internal TORAX functions. Used for debugging. Default is true.

```shell
$ export TORAX_COMPILATION_ENABLED=<True/False>
```


### Set flags
Output simulation time, dt, and number of stepper iterations (dt backtracking with nonlinear solver) carried out at each timestep.

```shell
$ python3 run_simulation_main.py \
$   --python_config='torax.tests.test_data.default_config' \
$   --log_progress
```

Live plotting of simulation state and derived quantities.

```shell
$ python3 run_simulation_main.py \
$   --python_config='torax.tests.test_data.default_config' \
$   --plot_progress
```

Combination of the above.

```shell
$ python3 run_simulation_main.py \
$   --python_config='torax.tests.test_data.default_config' \
$   --log_progress --plot_progress
```

### Post-simulation

Once complete, the time history of a simulation state and derived quantities is written to `state_history.h5`. The output path is written to stdout

To take advantage of the in-memory (non-persistent) cache, the process does not end upon simulation termination. It is possible to modify the config and rerun the simulation. Only the following modifications will then trigger a recompilation:

- Grid resolution
- Evolved variables (equations being solved)
- Changing internal functions used, e.g. transport model, or time_step_calculator


## Cleaning up

You can get out of the Python virtual env by deactivating it:

```shell
$ deactivate
```

# Simulation tutorials

Under construction

# FAQ

* On MacOS, you may get the error: .. ERROR:: Could not find a local HDF5
  installation.:
* Solution: You need to tell the OS where HDF5 is, try

```shell
$ brew install hdf5
$ export HDF5_DIR="$(brew --prefix hdf5)"
$ pip install --no-binary=h5py h5py
```
