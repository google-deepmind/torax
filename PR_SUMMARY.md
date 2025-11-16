# Pull Request Summary: Custom Pedestal Model Registration API

## Issue
Fixes #1711: Simple public API for user-defined pedestal pressure scaling

## Overview

This PR implements a **Pedestal Model Registration API** that allows users to define custom pedestal scaling laws without modifying the TORAX source code. The implementation follows the **same design pattern** as the transport model registration API ([`transport_model/register_model.py`](torax/_src/transport_model/register_model.py)), as requested by the maintainers.

## Motivation

Previously, implementing custom pedestal models (such as those used in STEP with power-law fits to Europed data and modified EPED scaling) required developers to create new implementations directly in the TORAX codebase. This created barriers for users with machine-specific scaling requirements.

The maintainer feedback requested:
> "So the way we would want to do this is more like the API contained here https://github.com/google-deepmind/torax/blob/main/torax/_src/transport_model/register_model.py so the user does have to implement a custom model but that model doesn't have to be part of TORAX."

This PR implements exactly that pattern for pedestal models.

## Changes

### New Files

1. **`torax/_src/pedestal_model/register_model.py`** (~90 lines)
   - `register_pedestal_model()` function - Dynamically registers user-defined pedestal models
   - Modifies Pydantic union types at runtime
   - Comprehensive docstring with usage examples
   - Follows the exact same pattern as `transport_model/register_model.py`

2. **`torax/_src/pedestal_model/tests/register_model_test.py`** (~150 lines)
   - Comprehensive unit tests for registration functionality:
     - Test single model registration
     - Test multiple model registrations
     - Verify models can be instantiated and used
   - Uses `absltest` framework consistent with existing tests

3. **`torax/examples/custom_pedestal_example.py`** (~375 lines)
   - Complete example with EPED-like scaling implementation
   - Shows how to define JAX pedestal model class
   - Shows how to define Pydantic configuration class
   - Demonstrates model registration
   - Includes two examples: complex EPED-like model and simple constant model
   - Detailed documentation and comments

4. **`QUICK_START_CUSTOM_PEDESTAL.md`** (~290 lines)
   - Quick start guide for users
   - Four-step tutorial with code examples
   - Complete working examples
   - Documentation of accessible parameters
   - Troubleshooting tips

### Modified Files

1. **`torax/_src/pedestal_model/pydantic_model.py`**
   - **Removed**: `CustomPedestal` class (callable-based approach)
   - **Removed**: Import of `custom_pedestal` module
   - **Updated**: `PedestalConfig` union now only includes built-in models
   - Users extend this via registration instead

### Removed Files

1. **`torax/_src/pedestal_model/custom_pedestal.py`**
   - Removed callable-based custom pedestal implementation
   - Replaced by user-defined model pattern

2. **`torax/_src/pedestal_model/tests/custom_pedestal_test.py`**
   - Removed tests for callable-based approach
   - Replaced by registration pattern tests

## User-Facing API

Users now follow the same pattern as transport models:

### Step 1: Define JAX Pedestal Model

```python
@chex.dataclass(frozen=True)
class MyPedestalModel(pm.PedestalModel):
  def _call_implementation(
      self,
      runtime_params: 'MyRuntimeParams',
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pm.PedestalModelOutput:
    # User's custom pedestal calculation logic
    Ip_MA = runtime_params.profile_conditions.Ip / 1e6
    B0 = geo.B0

    T_e_ped = 0.5 * (Ip_MA ** 0.2) * (B0 ** 0.8)
    T_i_ped = 1.2 * T_e_ped
    n_e_ped = 0.7e20

    return pm.PedestalModelOutput(...)
```

### Step 2: Define Pydantic Configuration

```python
class MyPedestal(pydantic_model.BasePedestal):
  model_name: Annotated[Literal['my_pedestal'], torax_pydantic.JAX_STATIC] = (
      'my_pedestal'
  )

  # Add configuration parameters
  scaling_factor: float = 1.0

  def build_pedestal_model(self) -> MyPedestalModel:
    return MyPedestalModel()

  def build_runtime_params(self, t: chex.Numeric) -> MyRuntimeParams:
    return MyRuntimeParams(...)
```

### Step 3: Register the Model

```python
from torax._src.pedestal_model import register_model

register_model.register_pedestal_model(MyPedestal)
```

### Step 4: Use in Configuration

```python
CONFIG = {
    'pedestal': {
        'model_name': 'my_pedestal',
        'set_pedestal': True,
        'scaling_factor': 1.5,
    },
}
```

## Design Pattern

The implementation follows the **exact same pattern** as transport models:

| Component | Transport Models | Pedestal Models |
|-----------|------------------|-----------------|
| **Registration Function** | `register_transport_model()` | `register_pedestal_model()` |
| **Base Pydantic Class** | `TransportBase` | `BasePedestal` |
| **Base JAX Class** | `TransportModel` | `PedestalModel` |
| **Union Type** | `TransportConfig` | `PedestalConfig` |
| **Required Methods** | `build_transport_model()`<br>`build_runtime_params()` | `build_pedestal_model()`<br>`build_runtime_params()` |
| **Example File** | Various transport examples | `custom_pedestal_example.py` |

### Key Advantages Over Callable Approach

1. **Consistency**: Same pattern as transport models - easier to learn and maintain
2. **Extensibility**: Users can define complex models with multiple parameters
3. **Type Safety**: Full Pydantic validation for user configurations
4. **JAX Compatibility**: Proper frozen dataclasses for JIT compilation
5. **Runtime Parameters**: Support for time-varying parameters
6. **No Source Modification**: Users never need to modify TORAX source code

## Backwards Compatibility

✅ **Fully backwards compatible** - existing pedestal models unchanged:
- `SetTpedNped` - unchanged
- `SetPpedTpedRatioNped` - unchanged
- `NoPedestal` - unchanged

The registration pattern extends the system without affecting existing models.

## Testing

- ✅ Registration function tests
- ✅ Multiple model registration tests
- ✅ Model instantiation and usage tests
- ✅ Example code provided and validated

The CI pipeline will run the full test suite to ensure no regressions.

## Documentation

1. **Quick Start Guide**: [`QUICK_START_CUSTOM_PEDESTAL.md`](QUICK_START_CUSTOM_PEDESTAL.md)
   - Four-step tutorial
   - Complete working examples
   - Troubleshooting guide

2. **Example Implementation**: [`torax/examples/custom_pedestal_example.py`](torax/examples/custom_pedestal_example.py)
   - EPED-like scaling model
   - Simple constant model
   - Detailed inline comments

3. **API Documentation**: In `register_model.py` docstring
   - Complete usage example
   - Parameter descriptions
   - Integration instructions

## Comparison with Original Callable Approach

The original PR used callable functions, which was simpler but less aligned with TORAX architecture:

| Aspect | Original (Callable) | New (Registration) |
|--------|-------------------|-------------------|
| **Pattern** | Custom to pedestals | Same as transport models |
| **User Implementation** | Write Python functions | Define model classes |
| **Validation** | Limited (Any type) | Full Pydantic validation |
| **Extensibility** | Limited | Highly extensible |
| **Maintainability** | Different pattern | Consistent with TORAX |
| **Architecture Alignment** | Low | High |

## Addressing Maintainer Feedback

✅ **"More like the API in transport_model/register_model.py"**
- Implemented exact same pattern

✅ **"User implements a custom model"**
- Users define model classes inheriting from `PedestalModel`

✅ **"Model doesn't have to be part of TORAX"**
- Models are registered dynamically, no source code modification needed

✅ **"Same pattern as @theo-brown meant"**
- Follows the transport model registration design

## Files Changed

### Added
- `torax/_src/pedestal_model/register_model.py` (+90 lines)
- `torax/_src/pedestal_model/tests/register_model_test.py` (+150 lines)
- `torax/examples/custom_pedestal_example.py` (+375 lines)
- `QUICK_START_CUSTOM_PEDESTAL.md` (+290 lines)

### Modified
- `torax/_src/pedestal_model/pydantic_model.py` (-82 lines, removed CustomPedestal)

### Removed
- `torax/_src/pedestal_model/custom_pedestal.py` (-163 lines)
- `torax/_src/pedestal_model/tests/custom_pedestal_test.py` (-246 lines)
- `CUSTOM_PEDESTAL_API.md` (will be replaced with updated version)

**Net Change**: ~+280 lines (more focused, better aligned with architecture)

## Acknowledgments

This implementation addresses the request from @theo-brown for a public API to define custom pedestal models for STEP and other machines with specific scaling requirements, using the pattern specified by the maintainers.
