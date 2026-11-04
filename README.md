# TORAX

[![Unittests][pytest-badge]][pytest-workflow]
[![Documentation][docs-badge]][docs]

TORAX is a differentiable 1D tokamak core plasma transport simulator written in
Python and [JAX][jax] for fast forward modelling, pulse design, trajectory
optimization, and controller design. Key design principles include:

- **Modular and extensible**: easy to register custom models for transport,
  sources, neoclassical effects, etc.
- **Fast and auto-differentiable**: End-to-end `jax.jit` compilation on CPU and
  GPU, gradient-based nonlinear PDE solvers without hand-derived Jacobians, and
  `jax.grad` / `jax.vmap` support for sensitivity analysis and parameter
  optimization.
- **ML-surrogate native**: Seamless JAX coupling to neural-network physics
  surrogates, including turbulent transport models from
  [`fusion_surrogates`][fusion-surrogates].

Full documentation and physics details are available at
[torax.readthedocs.io][docs]. Upgrading from TORAX v1? See the
[v1-to-v2 Migration Guide][migration-guide].

## Physics and numerics

- **Coupled 1D transport PDEs**
  - Finite-volume discretization of ion heat transport,
  electron heat transport, electron particle transport, and poloidal flux
  diffusion with a variety of boundary conditions.
  - Support for uniform or non-uniform grids.
  - Multiple solver options, including predictor-corrector, Newton-Raphson, and
  JAX-based optimizers.
- **Turbulent and neoclassical transport**
  - Multi-component core and pedestal transport supporting ML surrogates (QLKNN,
    TGLFNN), empirical models (Bohm/gyro-Bohm), and high-fidelity
    quasilinear models (QuaLiKiz, TGLF).
  - Fast analytic models for bootstrap current, conductivity, neoclassical
    transport, and poloidal velocity.
- **Pedestal, edge, and MHD**
  - Modular pedestal structure, with formation triggered by L-H power-threshold
  scalings and supporting both enforcing as a boundary condition or evolving
  via transport.
  - Built-in coupling to the Extended Lengyel 1D SOL/divertor model.
  - Simple sawtooth crash model.

- **Sources, sinks, and equilibria**
  - Analytical models for core physics, including fusion, impurity radiation,
  and Bremsstrahlung.
  - Basic models for actuators such as pellets, ion- and electron-cyclotron
  heating and current drive. Feedback controllers available for particle
  sources to match experimental density profiles.
  - Static and time-dependent equilibria from CHEASE, MEQ, EQDSK, IMAS and
  TokaMaker.

## Installation

TORAX requires Python 3.12 or newer. Install the latest release from PyPI in a
virtual environment:

```shell
python3 -m venv toraxvenv
source toraxvenv/bin/activate
pip install torax
```

For development (editable install with testing and linting dependencies):

```shell
git clone https://github.com/google-deepmind/torax.git
cd torax
pip install -e ".[dev]"
```

See the [Installation Guide][installation] for GPU setup and optional
`QLKNN10D` or `tglf2py` extensions.

## Quickstart

### Command line

Run an example configuration from the terminal:

```shell
run_torax --config=examples/iterhybrid_rampup.py --output_dir=/tmp/torax_results
```

View one or more saved runs with interactive Plotly plots:

```shell
plot_torax --outfile /tmp/torax_results/state_history_<timestamp>.nc
```

### Python API

```python
import torax

torax_config = torax.build_torax_config_from_file(
    'examples/iterhybrid_rampup.py'
)  # or torax.ToraxConfig.from_dict(config_dict)

state_history = torax.run_simulation(torax_config)

data_tree = state_history.simulation_output_to_xr()
data_tree.to_netcdf('state_history.nc')
```

Plot results directly in Python with `torax.plot_run_from_data_tree` (in-memory)
or `torax.plot_run` (from `.nc` files).
See [Running Programmatically][programmatic] for more details.

### JAX cache, debugging, and tests

To enable persistent JAX compilation caching, run

```shell
export JAX_COMPILATION_CACHE_DIR=~/jax_cache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.0
```

For debugging, set `JAX_DISABLE_JIT=True` to disable JIT compilation or
`TORAX_ERRORS_ENABLED=True` to enable internal runtime checks. Run the unit
test suite from a source checkout with:

```shell
TORAX_JAXTYPING=True TORAX_ERRORS_ENABLED=True pytest -n auto
```

## Citing TORAX

If you use TORAX in your research, please cite the [TORAX paper on
arXiv][paper]:

```
@article{torax2024arxiv,
  title={{TORAX: A Fast and Differentiable Tokamak Transport Simulator in JAX}},
  author={Citrin, Jonathan and Goodfellow, Ian and Raju, Akhil and Chen, Jeremy
  and Degrave, Jonas and Donner, Craig and Felici, Federico and Hamel, Philippe
  and Huber, Andrea and Nikulin, Dmitry and Pfau, David and Tracey, Brendan and
  Riedmiller, Martin and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:2406.06718},
  year={2024}
}
```

Interested in contributing? See the [Contributing][contributing] docs.

This is not an officially supported Google product.

[pytest-badge]: https://github.com/google-deepmind/torax/actions/workflows/pytest.yml/badge.svg
[pytest-workflow]: https://github.com/google-deepmind/torax/actions/workflows/pytest.yml
[docs-badge]: https://readthedocs.org/projects/torax/badge/?version=latest
[docs]: https://torax.readthedocs.io/
[jax]: https://docs.jax.dev/
[fusion-surrogates]: https://github.com/google-deepmind/fusion_surrogates
[migration-guide]: https://torax.readthedocs.io/en/latest/migration_guide.html
[installation]: https://torax.readthedocs.io/en/latest/installation.html
[programmatic]: https://torax.readthedocs.io/en/latest/running_programmatically.html
[contributing]: https://torax.readthedocs.io/en/latest/contributing.html
[paper]: https://arxiv.org/abs/2406.06718
