import dataclasses

import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
import functools
from torax import jax_utils


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'use_relativistic_correction',
    ],
)
def calc_bremsstrahlung(
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    Zeff: float,
    nref: float,
    use_relativistic_correction: bool = False,
) -> jax.Array:
    """

    Args:
        geo (geometry.Geometry): _description_
        core_profiles (state.CoreProfiles): _description_
        Zeff (float): _description_
        nref (float): _description_
        use_relativistic_correction (bool, optional): _description_. Defaults to False.

    Returns:
        jax.Array: _description_
    """
    ne20 = (nref / 1e20) * core_profiles.ne.face_value()

    Te_kev = core_profiles.temp_el.face_value()

    P_brem_profile: jax.Array = 5.35e-3 * Zeff * ne20**2 * jnp.sqrt(Te_kev) # MW/m^3

    if use_relativistic_correction:
      # Apply the Stott relativistic correction.
      Tm = 511.0  # m_e * c**2 in keV
      correction = (1.0 + 2.0 * Te_kev / Tm) * (
          1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + Te_kev / Tm))
      )
      P_brem_profile *= correction

    return P_brem_profile

def bremsstrahlung_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
    del dynamic_source_runtime_params  # Unused.
    P_brem_profile = calc_bremsstrahlung(
        geo,
        core_profiles,
        dynamic_runtime_params_slice.runtime_params.plasma_composition.Zeff,
        dynamic_runtime_params_slice.numerics.nref,
    )
    # As a sink, the power is negative.
    return -1.0 * P_brem_profile


@dataclasses.dataclass(kw_only=True)
class BremsstrahlungHeatSink(source.SingleProfileTempElSource):
  """Fusion heat source for both ion and electron heat."""

  # model_func: source.SourceProfileFunction = bremsstrahlung_model_func
  model_func = bremsstrahlung_model_func

BremsstrahlungHeatSinkBuilder = source.make_source_builder(BremsstrahlungHeatSink)
