# Pull Request: Pedestal Model Registration API

## Summary

This PR implements a Pedestal Model Registration API following the **same pattern as the transport model registration API**, as requested by the maintainers. Users can now define custom pedestal scaling laws without modifying TORAX source code.

## Fixes

Fixes #1711

## Motivation and Context

The maintainers requested that custom pedestal models follow the same registration pattern as transport models:

> "So the way we would want to do this is more like the API contained here https://github.com/google-deepmind/torax/blob/main/torax/_src/transport_model/register_model.py so the user does have to implement a custom model but that model doesn't have to be part of TORAX."

This approach enables users (e.g., STEP team) to implement machine-specific pedestal scaling laws (power-law fits to Europed data, modified EPED scaling) without modifying TORAX source code, while maintaining architectural consistency with transport models.

## Changes

### New Files
- `torax/_src/pedestal_model/register_model.py` - Registration function following transport model pattern
- `torax/_src/pedestal_model/tests/register_model_test.py` - Comprehensive tests for registration
- `torax/examples/custom_pedestal_example.py` - Complete examples (EPED-like + simple models)
- `QUICK_START_CUSTOM_PEDESTAL.md` - Quick start guide for users

### Modified Files
- `torax/_src/pedestal_model/pydantic_model.py` - Removed `CustomPedestal` callable-based class

### Removed Files
- `torax/_src/pedestal_model/custom_pedestal.py` - Replaced by registration pattern
- `torax/_src/pedestal_model/tests/custom_pedestal_test.py` - Replaced by registration tests

## Usage Example

Users follow four steps, identical to the transport model pattern:

```python
# Step 1: Define JAX model
@chex.dataclass(frozen=True)
class MyPedestalModel(pm.PedestalModel):
  def _call_implementation(self, runtime_params, geo, core_profiles):
    # Custom pedestal physics
    return pm.PedestalModelOutput(...)

# Step 2: Define Pydantic config
class MyPedestal(pydantic_model.BasePedestal):
  model_name: Annotated[Literal['my_pedestal'], ...] = 'my_pedestal'

  def build_pedestal_model(self):
    return MyPedestalModel()

  def build_runtime_params(self, t):
    return MyRuntimeParams(...)

# Step 3: Register
register_model.register_pedestal_model(MyPedestal)

# Step 4: Use
CONFIG = {'pedestal': {'model_name': 'my_pedestal', ...}}
```

## Pattern Consistency

This implementation exactly mirrors the transport model registration:

| Component | Transport | Pedestal |
|-----------|-----------|----------|
| Registration | `register_transport_model()` | `register_pedestal_model()` |
| Base class | `TransportBase` | `BasePedestal` |
| Union type | `TransportConfig` | `PedestalConfig` |

## Backwards Compatibility

✅ Fully backwards compatible - all existing pedestal models (`SetTpedNped`, `SetPpedTpedRatioNped`, `NoPedestal`) unchanged.

## Testing

- ✅ Registration functionality tested
- ✅ Multiple model registration tested
- ✅ Example code provided
- CI will run full test suite

## Documentation

- `QUICK_START_CUSTOM_PEDESTAL.md` - Tutorial with complete examples
- `custom_pedestal_example.py` - Working EPED-like and simple models
- Comprehensive docstrings in `register_model.py`

## Review Notes

This PR addresses the maintainer feedback by:
1. Following the exact transport model registration pattern
2. Requiring users to implement full model classes (not just functions)
3. Enabling external model definitions without source modifications
4. Maintaining architectural consistency across TORAX

See [`QUICK_START_CUSTOM_PEDESTAL.md`](QUICK_START_CUSTOM_PEDESTAL.md) and [`custom_pedestal_example.py`](torax/examples/custom_pedestal_example.py) for complete usage examples.
