# Custom Pedestal Model API

## Overview

This document describes the Custom Pedestal Model API added to TORAX, which allows users to define custom pedestal scaling laws without modifying the source code. This feature addresses [Issue #1711](https://github.com/google-deepmind/torax/issues/1711).

## Motivation

Previously, implementing custom pedestal models (such as those used in STEP with power-law fits to Europed data and modified EPED scaling) required developers to create new pedestal model implementations directly in the TORAX codebase. This created barriers for users with machine-specific scaling requirements.

The Custom Pedestal Model API follows the same design pattern as the public transport model API, enabling users to couple simple custom pedestal models without modifying core source code.

## Features

The Custom Pedestal Model allows users to provide callable Python functions that compute:

- **Ion temperature at the pedestal** (`T_i_ped`) in keV
- **Electron temperature at the pedestal** (`T_e_ped`) in keV
- **Electron density at the pedestal** (`n_e_ped`) in m⁻³ or as Greenwald fraction
- **Pedestal top location** (`rho_norm_ped_top`) - optional, can be dynamic or fixed

Each function receives full access to:
- Runtime parameters (plasma current, boundary conditions, etc.)
- Geometry (magnetic field, minor radius, aspect ratio, etc.)
- Core profiles (current profiles, temperatures, densities)

## Usage

### Basic Example

```python
# Define custom scaling functions
def my_T_e_ped(runtime_params, geo, core_profiles):
    """Compute electron temperature at pedestal using EPED-like scaling."""
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6  # Convert to MA
    B_T = geo.B0  # Toroidal magnetic field
    return 0.5 * (Ip_MA ** 0.2) * (B_T ** 0.8)

def my_T_i_ped(runtime_params, geo, core_profiles):
    """Compute ion temperature using T_i/T_e ratio."""
    T_e = my_T_e_ped(runtime_params, geo, core_profiles)
    return 1.2 * T_e  # T_i = 1.2 * T_e

def my_n_e_ped(runtime_params, geo, core_profiles):
    """Compute electron density as Greenwald fraction."""
    return 0.7  # 0.7 * nGW

# Configuration
CONFIG = {
    'pedestal': {
        'model_name': 'custom',  # Use the custom pedestal model
        'set_pedestal': True,    # Enable pedestal
        # Provide the custom functions
        'T_i_ped_fn': my_T_i_ped,
        'T_e_ped_fn': my_T_e_ped,
        'n_e_ped_fn': my_n_e_ped,
        # Optional: fixed pedestal location
        'rho_norm_ped_top': 0.91,
        # Density units flag
        'n_e_ped_is_fGW': True,  # n_e_ped_fn returns Greenwald fraction
    },
    # ... other configuration
}
```

### Advanced Example with Dynamic Pedestal Width

```python
def my_rho_norm_ped_top(runtime_params, geo, core_profiles):
    """Compute pedestal width based on poloidal beta."""
    import jax.numpy as jnp

    Ip_MA = runtime_params.profile_conditions.Ip / 1e6

    # Simple scaling: higher current -> narrower pedestal
    base_rho = 0.92
    current_correction = -0.005 * (Ip_MA - 15.0)

    return jnp.clip(base_rho + current_correction, 0.85, 0.95)

CONFIG = {
    'pedestal': {
        'model_name': 'custom',
        'set_pedestal': True,
        'T_i_ped_fn': my_T_i_ped,
        'T_e_ped_fn': my_T_e_ped,
        'n_e_ped_fn': my_n_e_ped,
        'rho_norm_ped_top_fn': my_rho_norm_ped_top,  # Dynamic pedestal location
        'n_e_ped_is_fGW': True,
    },
    # ... other configuration
}
```

## API Reference

### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | str | Yes | - | Must be `'custom'` |
| `set_pedestal` | bool or TimeVaryingScalar | No | `False` | Enable/disable pedestal |
| `T_i_ped_fn` | Callable | Yes | - | Function to compute T_i at pedestal [keV] |
| `T_e_ped_fn` | Callable | Yes | - | Function to compute T_e at pedestal [keV] |
| `n_e_ped_fn` | Callable | Yes | - | Function to compute n_e at pedestal [m⁻³ or fGW] |
| `rho_norm_ped_top_fn` | Callable or None | No | `None` | Optional function to compute pedestal location |
| `rho_norm_ped_top` | float or TimeVaryingScalar | No | `0.91` | Fixed pedestal location (used if `rho_norm_ped_top_fn` is None) |
| `n_e_ped_is_fGW` | bool | No | `False` | If True, `n_e_ped_fn` returns Greenwald fraction; if False, returns absolute density in m⁻³ |

### Function Signatures

All custom functions must have the following signature:

```python
def custom_function(
    runtime_params: RuntimeParams,
    geo: Geometry,
    core_profiles: CoreProfiles
) -> FloatScalar:
    """
    Args:
        runtime_params: Runtime parameters containing Ip, boundary conditions, etc.
        geo: Geometry object with B0, a_minor, rho_norm, etc.
        core_profiles: Core plasma profiles

    Returns:
        Scalar value for the pedestal quantity
    """
    # Your implementation
    return value
```

### Available Input Data

**From `runtime_params`:**
- `runtime_params.profile_conditions.Ip` - Plasma current [A]
- `runtime_params.profile_conditions.*` - Other boundary conditions
- `runtime_params.pedestal` - Pedestal-specific runtime parameters

**From `geo`:**
- `geo.B0` - Toroidal magnetic field on axis [T]
- `geo.a_minor` - Minor radius [m]
- `geo.rho_norm` - Normalized radial coordinate array
- `geo.torax_mesh` - Mesh information
- See [geometry.py](torax/_src/geometry/geometry.py) for full list

**From `core_profiles`:**
- `core_profiles.temp_ion` - Ion temperature profile
- `core_profiles.temp_el` - Electron temperature profile
- `core_profiles.ne` - Electron density profile
- `core_profiles.Z_eff` - Effective charge profile
- See [state.py](torax/_src/state.py) for full list

## Implementation Details

### File Structure

The implementation consists of:

1. **[custom_pedestal.py](torax/_src/pedestal_model/custom_pedestal.py)** - Core implementation
   - `CustomPedestalModel` class
   - `RuntimeParams` dataclass

2. **[pydantic_model.py](torax/_src/pedestal_model/pydantic_model.py)** - Configuration
   - `CustomPedestal` Pydantic configuration class
   - Added to `PedestalConfig` union type

3. **[tests/custom_pedestal_test.py](torax/_src/pedestal_model/tests/custom_pedestal_test.py)** - Unit tests
   - Tests for simple functions
   - Tests for scaling laws
   - Tests for dynamic pedestal location
   - Tests for Greenwald fraction conversion

4. **[examples/custom_pedestal_example.py](torax/examples/custom_pedestal_example.py)** - Example configurations
   - EPED-like scaling example
   - Simple constant values example
   - Dynamic pedestal width example

### Design Pattern

The Custom Pedestal Model follows the same pattern as existing pedestal models:

1. **JAX Model Layer** (`CustomPedestalModel`):
   - Frozen dataclass inheriting from `PedestalModel`
   - Stores callable functions
   - Implements `_call_implementation()` method

2. **Pydantic Configuration Layer** (`CustomPedestal`):
   - Validates configuration
   - Builds the JAX model via `build_pedestal_model()`
   - Builds runtime params via `build_runtime_params(t)`

3. **Runtime Parameters** (`RuntimeParams`):
   - JAX pytree registered
   - Frozen dataclass
   - Contains values that can vary with time

## Comparison with Transport Model API

The Custom Pedestal Model API is inspired by the existing transport model registration pattern but simplified for pedestal use cases:

| Aspect | Transport Models | Pedestal Models |
|--------|------------------|-----------------|
| Custom functions | Supported via subclassing | Supported via callables |
| Registration | Dynamic via `register_transport_model()` | Built into union type |
| Post-processing | Extensive (clipping, smoothing, patches) | Minimal |
| Common parameters | Many (chi_min, chi_max, patches, etc.) | Few (set_pedestal, location) |
| Use case | Complex turbulent transport | Boundary condition scaling |

## Examples

### STEP-like Pedestal Model

```python
def step_T_e_ped(runtime_params, geo, core_profiles):
    """STEP pedestal model with Europed fit."""
    import jax.numpy as jnp

    # Extract parameters
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B_T = geo.B0
    kappa = 1.7  # elongation - could be extracted from geo
    delta = 0.4  # triangularity

    # Power-law fit to Europed data (example coefficients)
    C = 0.85
    alpha_Ip = 0.22
    alpha_B = 0.75
    alpha_kappa = 0.30

    T_e_ped = C * (Ip_MA ** alpha_Ip) * (B_T ** alpha_B) * (kappa ** alpha_kappa)

    return T_e_ped

def step_pedestal_width(runtime_params, geo, core_profiles):
    """Pedestal width scaling with poloidal beta."""
    import jax.numpy as jnp

    # Simple beta_p proxy
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B_T = geo.B0

    beta_p_proxy = Ip_MA / B_T

    # Width scaling
    base_width = 0.08
    beta_correction = 0.01 * jnp.sqrt(beta_p_proxy)

    width = jnp.clip(base_width + beta_correction, 0.05, 0.12)
    rho_ped = 1.0 - width

    return rho_ped
```

### Time-Varying Pedestal

You can make the pedestal behavior time-dependent by accessing time-varying parameters:

```python
def rampup_T_e_ped(runtime_params, geo, core_profiles):
    """Pedestal temperature that scales with current during ramp-up."""
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6

    # During ramp-up, Ip varies with time
    # This function automatically adapts as Ip changes
    return 0.5 * (Ip_MA ** 0.25)

CONFIG = {
    'profile_conditions': {
        'Ip': {0.0: 1e6, 10.0: 15e6},  # Ramp from 1 to 15 MA over 10s
    },
    'pedestal': {
        'model_name': 'custom',
        'set_pedestal': {0.0: False, 2.0: True},  # Enable at t=2s
        'T_e_ped_fn': rampup_T_e_ped,
        # ...
    },
}
```

## Testing

Run the unit tests:

```bash
python -m pytest torax/_src/pedestal_model/tests/custom_pedestal_test.py -v
```

Or with absltest:

```bash
python torax/_src/pedestal_model/tests/custom_pedestal_test.py
```

## Migration Guide

### From SetTpedNped

**Before:**
```python
CONFIG = {
    'pedestal': {
        'model_name': 'set_T_ped_n_ped',
        'T_i_ped': 5.0,
        'T_e_ped': 4.5,
        'n_e_ped': 0.7e20,
        'rho_norm_ped_top': 0.91,
    },
}
```

**After:**
```python
def T_i_ped_fn(runtime_params, geo, core_profiles):
    return 5.0

def T_e_ped_fn(runtime_params, geo, core_profiles):
    return 4.5

def n_e_ped_fn(runtime_params, geo, core_profiles):
    return 0.7e20

CONFIG = {
    'pedestal': {
        'model_name': 'custom',
        'T_i_ped_fn': T_i_ped_fn,
        'T_e_ped_fn': T_e_ped_fn,
        'n_e_ped_fn': n_e_ped_fn,
        'rho_norm_ped_top': 0.91,
        'n_e_ped_is_fGW': False,
    },
}
```

### Adding Scaling

You can now add dependencies on plasma parameters:

```python
def T_e_ped_fn(runtime_params, geo, core_profiles):
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B_T = geo.B0
    return 0.5 * (Ip_MA ** 0.2) * (B_T ** 0.8)  # EPED-like scaling
```

## Limitations and Future Work

### Current Limitations

1. **No registration mechanism**: Unlike transport models, custom pedestal models must be configured directly in the config file (no separate registration step)
2. **No post-processing pipeline**: Unlike transport models, there's no built-in clipping, smoothing, or domain restriction
3. **Limited validation**: Pydantic doesn't validate callable signatures, so errors will only appear at runtime

### Future Enhancements

Potential improvements for future versions:

1. Add optional registration mechanism similar to `register_transport_model()`
2. Add validation helpers to check function signatures
3. Support for multiple pedestal models with different regions (e.g., ITB + pedestal)
4. Built-in common scaling laws (EPED, EPED1, etc.) as callable factories
5. Integration with external pedestal prediction codes (Europed, EPED-NN, etc.)

## Contributing

To contribute improvements to the Custom Pedestal Model API:

1. Add new tests to [tests/custom_pedestal_test.py](torax/_src/pedestal_model/tests/custom_pedestal_test.py)
2. Update this documentation
3. Add examples to [examples/custom_pedestal_example.py](torax/examples/custom_pedestal_example.py)
4. Follow the existing code style and patterns

## References

- [Issue #1711: Simple public API for user-defined pedestal pressure scaling](https://github.com/google-deepmind/torax/issues/1711)
- Transport Model API: [torax/_src/transport_model/](torax/_src/transport_model/)
- Existing Pedestal Models: [torax/_src/pedestal_model/](torax/_src/pedestal_model/)

## Support

For questions or issues:

- Open an issue on [GitHub](https://github.com/google-deepmind/torax/issues)
- Check the [TORAX documentation](https://github.com/google-deepmind/torax)
- See [examples/custom_pedestal_example.py](torax/examples/custom_pedestal_example.py) for working examples
