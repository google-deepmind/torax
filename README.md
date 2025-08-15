[![Unittests](https://github.com/google-deepmind/torax/actions/workflows/pytest.yml/badge.svg)](https://github.com/google-deepmind/torax/actions/workflows/pytest.yml)

## What is TORAX?

TORAX is a differentiable tokamak core transport simulator aimed for fast
and accurate forward modelling, pulse-design, trajectory optimization, and
controller design workflows. TORAX is written in Python using JAX, with
the following motivations:

- Open-source and extensible, aiding with flexible workflow coupling
- JAX provides auto-differentiation capabilities and code compilation for fast
runtimes. Differentiability allows for gradient-based nonlinear PDE solvers for
fast and accurate modelling, and for sensitivity analysis of simulation results
to arbitrary parameter inputs, enabling applications such as trajectory
optimization and data-driven parameter identification for semi-empirical models.
Auto-differentiability allows for these applications to be easily extended with
the addition of new physics models, including ML-surrogates, or new parameter
inputs, by avoiding the need to hand-derive Jacobians
- Python-JAX is a natural framework for the coupling of ML-surrogates of physics
models

For more comprehensive documentation, see our [readthedocs page](https://torax.readthedocs.io/).

TORAX, at v1.0.0, has the following physics and numerics feature set:

- Coupled PDEs of ion and electron heat transport, electron particle transport,
and current diffusion
    - Finite-volume-method discretization
    - Multiple solver options: linear with Pereverzev-Corrigan terms and the
    predictor-corrector method, nonlinear with Newton-Raphson, nonlinear with
    optimization using the jaxopt library
    - Poloidal flux boundary conditions based on either total current or loop
    voltage at the last-closed-flux-surface
- Ohmic power, ion-electron heat exchange, fusion power, Bremsstrahlung,
impurity line radiation, an [[ICRH ML-surrogate]](https://meetings.aps.org/Meeting/DPP24/Session/NP12.106) (as-yet covering limited regimes),
- Neoclassical bootstrap current and conductivity with the analytical Sauter
  model
- Coupling to the [[QLKNN_7_11]](https://github.com/google-deepmind/fusion_surrogates/)
and QLKNN10D
[[van de Plassche et al, Phys. Plasmas 2020]](https://doi.org/10.1063/1.5134126)
QuaLiKiz neural network surrogates for physics-based turbulent transport
- General geometry, provided via CHEASE, FBT, or EQDSK equilibrium files
    - For testing and demonstration purposes, a single CHEASE equilibrium file
    is available in the `data/third_party/geo` directory. It corresponds to an
    ITER hybrid scenario equilibrium based on simulations in
    [[Citrin et al, Nucl. Fusion 2010]](https://doi.org/10.1088/0029-5515/50/11/115007),
    and was obtained from [PINT](https://gitlab.com/qualikiz-group/pyntegrated_model).
    A PINT license file is available in `data/third_party/geo`
    - Time dependent geometry is supported by providing a time series of geometry files
- Simple pedestal models using a local adaptive source to set internal boundary
conditions
- Sawtooth triggering and profile redistribution

Additional heating and current drive sources can be provided by user-provided
analytical models, or user-provided prescribed data.

Model implementation was verified through direct comparison of simulation
outputs to the RAPTOR
[[Felici et al, Plasma Phys. Control. Fusion 2012]](https://iopscience.iop.org/article/10.1088/0741-3335/54/2/025002)
tokamak transport simulator.

This is not an officially supported Google product.

## Development roadmap

A [development roadmap](https://torax.readthedocs.io/en/latest/roadmap.html#development-roadmap)
is outlined in our readthedocs pages.

## Installation guide

### Requirements

Install Python 3.11 or greater.

Make sure that tkinter is installed:

```shell
sudo apt-get install python3-tk
```

### How to install

#### Prepare a virtual environment

Install Virtualenv (if not already installed):

```shell
pip install --upgrade pip
```

```shell
pip install virtualenv
```

Create and activate a virtual environment

```shell
python3 -m venv toraxvenv
source toraxvenv/bin/activate
```

#### Install from PyPI

The simplest way to use TORAX is to install it via PyPI:

```shell
pip install torax
```

You can check that everything runs as it should:

```shell
run_torax --config=examples/basic_config.py --quit
```

If you plan to help develop TORAX, you will need to clone the repository, see
[contributing](https://torax.readthedocs.io/en/latest/contributing.html) and
[contribution_tips](https://torax.readthedocs.io/en/latest/contribution_tips.html#contribution-tips)

### Running an example

The following command will run TORAX using the default configuration file
`examples/basic_config.py`.

```shell
run_torax --config='examples/basic_config.py'
```

Simulation progress is shown by a terminal progress bar indicating the current
time and percentage completed.

To run more involved, ITER-inspired simulations, run:

```shell
run_torax --config='examples/iterhybrid_rampup.py'
```

and

```shell
run_torax --config='examples/iterhybrid_predictor_corrector.py'
```

Additional configuration is provided through flags which append the above
run command, and environment variables. For example, for increased output
verbosity, you can run with the `--log_progress` flag.

```shell
run_torax  --config='examples/iterhybrid_rampup.py' --log_progress
```

#### Set environment variables

##### Error checking

If true, error checking is enabled in internal routines. Used for debugging.
Default is false since it is incompatible with the persistent compilation cache.

```shell
$ export TORAX_ERRORS_ENABLED=<True/False>
```

##### JAX Compilation and Cache

If false, JAX does not compile internal TORAX functions. Used for debugging.
Default is true.

```shell
$ export TORAX_COMPILATION_ENABLED=<True/False>
```

The following implements the JAX persistent cache and will cause jax to store
compiled programs to the filesystem, reducing recompilation time in some cases:

```shell
$ export JAX_COMPILATION_CACHE_DIR=<path of your choice, such as ~/jax_cache>
$ export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
$ export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
```

#### Set flags
`log_progress` - output simulation time, dt, and number of solver iterations
(dt backtracking with nonlinear solver) carried out at each timestep.

```shell
run_torax \
   --config=examples/iterhybrid_predictor_corrector.py \
   --log_progress
```

`output_dir` - overrides the default output directory

```shell
run_torax \
   --config=examples/iterhybrid_predictor_corrector.py \
   --output_dir=/path/to/output/dir
```

### Post-simulation

Once complete, the time history of a simulation state and derived quantities
is written to a timestamped file of the format `state_history_%Y%m%d_%H%M%S.nc`.
The output directory is user configurable, with a default `/tmp/torax_results`.

To take advantage of the in-memory (non-persistent) cache, the process does not
end upon simulation termination. It is possible to modify the runtime_params,
toggle the `log_progress` and `plot_progress` flags, and rerun the simulation.
The following modifications are examples which will trigger a recompilation:

- Grid resolution
- Evolved variables (equations being solved)
- Changing internal functions used, e.g. transport models, time_step_calculator.
  source models, solver model, pedestal model

### Cleaning up

You can exit the Python virtual env by deactivating it:

```shell
deactivate
```

#### (Optional) Install QLKNN-hyper

The default transport model in TORAX is QLKNN_7_11, installed as a TORAX
dependency. An alternative to QLKNN_7_11 is to use QLKNN-hyper-10D, also known
as QLKNN10D [K.L. van de Plassche PoP 2020](https://doi.org/10.1063/1.5134126).
QLKNN_7_11 is based on QuaLiKiz 2.8.1 which has an improved collision
operator compared to the QLKNN10D training set. QLKNN_7_11 training data
includes impurity density gradients as an input feature and has better coverage
of the near-LCFS region compared to QLKNN-hyper-10D. However, using QLKNN10D may
still be useful for specific use-cases, such as in benchmarks with other
simulators, or comparisons with QLKNN_7_11.

Download QLKNN-hyper dependencies:

```shell
git clone https://gitlab.com/qualikiz-group/qlknn-hyper.git
```

To use QLKNN10D, set `model_path` in the `transport` section of your TORAX
config to the path of the cloned repository.

## Simulation tutorials

See tutorials section in our
[readthedocs pages](https://torax.readthedocs.io/en/latest/tutorials.html).

## Citing TORAX

A TORAX paper is [available on arXiv](https://arxiv.org/abs/2406.06718). Cite
this paper to cite TORAX:

```
@article{torax2024arxiv,
  title={{TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX}},
  author={Citrin, Jonathan and Goodfellow, Ian and Raju, Akhil and Chen, Jeremy
  and Degrave, Jonas and Donner, Craig and Felici, Federico and Hamel, Philippe
  and Huber, Andrea and Nikulin, Dmitry and Pfau, David and Tracey, Brendan, and
  Riedmiller, Martin and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2406.06718},
  year={2024}
}
```
