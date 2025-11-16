# Quick Start: Custom Pedestal Models in TORAX

This guide shows you how to create and use custom pedestal models in TORAX using the registration API.

## Overview

TORAX allows you to define custom pedestal scaling laws without modifying the source code. This is useful for machine-specific models like those used for STEP, which use power-law fits to Europed data.

The approach follows the same pattern as custom transport models in TORAX.

## Four Simple Steps

### 1. Define Your JAX Pedestal Model

Create a class that inherits from `PedestalModel` and implements the `_call_implementation` method:

```python
import chex
import jax.numpy as jnp
from torax._src import geometry, state
from torax._src.pedestal_model import pedestal_model as pm
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params

@chex.dataclass(frozen=True)
class MyPedestalModel(pm.PedestalModel):
  """My custom pedestal model with EPED-like scaling."""

  def _call_implementation(
      self,
      runtime_params: 'MyRuntimeParams',
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pm.PedestalModelOutput:
    """Compute pedestal values."""

    # Extract plasma parameters
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B0 = geo.B0

    # Your custom scaling laws
    T_e_ped = 0.5 * (Ip_MA ** 0.2) * (B0 ** 0.8)
    T_i_ped = 1.2 * T_e_ped
    n_e_ped = 0.7e20  # or use Greenwald fraction
    rho_norm_ped_top = 0.91

    # Find mesh index
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    return pm.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )

# Define runtime parameters for your model
@chex.dataclass(frozen=True)
class MyRuntimeParams(pedestal_runtime_params.RuntimeParams):
  """Runtime parameters for my pedestal model."""
  # Add any additional parameters your model needs
  pass
```

### 2. Define Your Pydantic Configuration Class

Create a configuration class that inherits from `BasePedestal`:

```python
from typing import Annotated, Literal
import chex
from torax._src.pedestal_model import pydantic_model
from torax._src.torax_pydantic import torax_pydantic

class MyPedestal(pydantic_model.BasePedestal):
  """Configuration for my custom pedestal model."""

  model_name: Annotated[Literal['my_pedestal'], torax_pydantic.JAX_STATIC] = (
      'my_pedestal'
  )

  # Add any configuration parameters
  # Example: scaling_factor: float = 1.0

  def build_pedestal_model(self) -> MyPedestalModel:
    """Build the JAX model."""
    return MyPedestalModel()

  def build_runtime_params(self, t: chex.Numeric) -> MyRuntimeParams:
    """Build runtime parameters."""
    return MyRuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
    )
```

### 3. Register Your Model

Register your model with TORAX:

```python
from torax._src.pedestal_model import register_model

register_model.register_pedestal_model(MyPedestal)
```

### 4. Use It in Your Configuration

Now you can use your custom model in any TORAX configuration:

```python
CONFIG = {
    'pedestal': {
        'model_name': 'my_pedestal',
        'set_pedestal': True,
        # Any additional parameters you defined
    },
    # ... rest of your config
}
```

## Complete Example

Here's a complete working example:

```python
# my_custom_pedestal.py
from typing import Annotated, Literal
import chex
import jax.numpy as jnp

from torax._src import geometry, state
from torax._src.pedestal_model import pedestal_model as pm
from torax._src.pedestal_model import pydantic_model
from torax._src.pedestal_model import register_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params
from torax._src.torax_pydantic import torax_pydantic


# Step 1: JAX Model
@chex.dataclass(frozen=True)
class EPEDLikePedestalModel(pm.PedestalModel):
  def _call_implementation(
      self,
      runtime_params: 'EPEDRuntimeParams',
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pm.PedestalModelOutput:
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B0 = geo.B0

    T_e_ped = runtime_params.T_e_factor * (Ip_MA ** 0.2) * (B0 ** 0.8)
    T_i_ped = T_e_ped * runtime_params.T_i_T_e_ratio

    # Greenwald fraction
    a = geo.Rmin
    n_GW = Ip_MA / (jnp.pi * a**2) * 1e20
    n_e_ped = runtime_params.f_GW * n_GW

    rho_norm_ped_top = runtime_params.rho_norm_ped_top
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - rho_norm_ped_top)
    )

    return pm.PedestalModelOutput(
        rho_norm_ped_top=rho_norm_ped_top,
        rho_norm_ped_top_idx=rho_norm_ped_top_idx,
        T_i_ped=T_i_ped,
        T_e_ped=T_e_ped,
        n_e_ped=n_e_ped,
    )


@chex.dataclass(frozen=True)
class EPEDRuntimeParams(pedestal_runtime_params.RuntimeParams):
  T_e_factor: float = 0.5
  T_i_T_e_ratio: float = 1.2
  f_GW: float = 0.7
  rho_norm_ped_top: float = 0.91


# Step 2: Pydantic Config
class EPEDLikePedestal(pydantic_model.BasePedestal):
  model_name: Annotated[Literal['eped_like'], torax_pydantic.JAX_STATIC] = (
      'eped_like'
  )

  T_e_factor: float = 0.5
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.2)
  )
  f_GW: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7)
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(self) -> EPEDLikePedestalModel:
    return EPEDLikePedestalModel()

  def build_runtime_params(self, t: chex.Numeric) -> EPEDRuntimeParams:
    return EPEDRuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        T_e_factor=self.T_e_factor,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        f_GW=self.f_GW.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


# Step 3: Register
register_model.register_pedestal_model(EPEDLikePedestal)


# Step 4: Use in config
CONFIG = {
    'pedestal': {
        'model_name': 'eped_like',
        'set_pedestal': True,
        'T_e_factor': 0.5,
        'T_i_T_e_ratio': 1.2,
        'f_GW': 0.7,
        'rho_norm_ped_top': 0.91,
    },
    # ... rest of config
}
```

## Key Points

1. **JAX Compatibility**: Your model must be JAX-compatible (use `jax.numpy`, avoid Python loops)
2. **Frozen Dataclasses**: Use `@chex.dataclass(frozen=True)` for immutability
3. **Unique Model Name**: Choose a unique `model_name` for your model
4. **Time-Varying Parameters**: Use `torax_pydantic.TimeVaryingScalar` for parameters that can vary with time
5. **Greenwald Fraction**: You can convert Greenwald fraction to absolute density using the formula shown above

## What You Can Access

In your `_call_implementation` method, you have access to:

### Runtime Parameters (`runtime_params`)
- `runtime_params.profile_conditions.Ip` - Plasma current [A]
- `runtime_params.profile_conditions.Ip_from_parameters` - If true, Ip is derived
- `runtime_params.profile_conditions.ne_bound_right` - Boundary density
- `runtime_params.profile_conditions.kappa` - Elongation
- And more...

### Geometry (`geo`)
- `geo.B0` - Toroidal field [T]
- `geo.Rmin` - Minor radius [m]
- `geo.Rmaj` - Major radius [m]
- `geo.epsilon` - Inverse aspect ratio
- `geo.rho_norm` - Normalized radial coordinate array
- And more...

### Core Profiles (`core_profiles`)
- `core_profiles.temp_ion` - Ion temperature profile
- `core_profiles.temp_el` - Electron temperature profile
- `core_profiles.ne` - Electron density profile
- `core_profiles.psi` - Poloidal flux
- And more...

## Next Steps

- See [`custom_pedestal_example.py`](torax/examples/custom_pedestal_example.py) for a complete working example
- See [`CUSTOM_PEDESTAL_API.md`](CUSTOM_PEDESTAL_API.md) for detailed API documentation
- Compare with the transport model registration in [`transport_model/register_model.py`](torax/_src/transport_model/register_model.py)

## Need Help?

If you encounter issues:
1. Check that your model_name is unique and doesn't conflict with built-in models ('set_T_ped_n_ped', 'set_P_ped_n_ped', 'no_pedestal')
2. Ensure your JAX model is compatible with JIT compilation
3. Verify that your Pydantic class properly inherits from `BasePedestal`
4. Make sure you call `register_pedestal_model()` before using the model in a config
