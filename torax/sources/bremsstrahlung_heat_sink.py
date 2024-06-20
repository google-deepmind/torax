import dataclasses

import jax
import chex
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import runtime_params_slice, config_args
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
import functools
from torax import jax_utils


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  use_relativistic_correction: bool


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  use_relativistic_correction: bool = False

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'use_relativistic_correction',
    ],
)
def calc_bremsstrahlung(
    core_profiles: state.CoreProfiles,
    Zeff: float,
    nref: float,
    use_relativistic_correction: bool = False,
) -> jax.Array:
  """Calculate the bremsstrahlung radiation power profile using the model from
  Wesson, John, and David J. Campbell. Tokamaks. Vol. 149. An optional correction
  for relativistic effects from Stott PPCF 2005 can be enabled with the flag
  "use_relativistic_correction".

  Args:
      core_profiles (state.CoreProfiles): core plasma profiles.
      Zeff (float): effective charge number.
      nref (float): reference density.
      use_relativistic_correction (bool, optional): Set to true to include the relativistic correction from Stott. Defaults to False.

  Returns:
      jax.Array: bremsstrahlung radiation power profile [MW/m^3]
  """
  ne20 = (nref / 1e20) * core_profiles.ne.face_value()

  Te_kev = core_profiles.temp_el.face_value()

  P_brem_profile: jax.Array = (
      5.35e-3 * Zeff * ne20**2 * jnp.sqrt(Te_kev)
  )  # MW/m^3

  if use_relativistic_correction:
    # Apply the Stott relativistic correction.
    Tm = 511.0  # m_e * c**2 in keV
    correction = (1.0 + 2.0 * Te_kev / Tm) * (
        1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + Te_kev / Tm))
    )
    P_brem_profile *= correction

  P_brem_profile_cell = geometry.face_to_cell(P_brem_profile)
  return P_brem_profile_cell


def bremsstrahlung_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
  del geo  # Unused.
  P_brem_profile = calc_bremsstrahlung(
      core_profiles,
      dynamic_runtime_params_slice.plasma_composition.Zeff,
      dynamic_runtime_params_slice.numerics.nref,
      use_relativistic_correction=dynamic_source_runtime_params.use_relativistic_correction,
  )
  # As a sink, the power is negative.
  return -1.0 * P_brem_profile


@dataclasses.dataclass(kw_only=True)
class BremsstrahlungHeatSink(source.SingleProfileTempElSource):
  """Fusion heat source for both ion and electron heat."""

  supported_modes: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.ZERO,
      runtime_params_lib.Mode.MODEL_BASED,
  )

  model_func: source.SourceProfileFunction = bremsstrahlung_model_func


BremsstrahlungHeatSinkBuilder = source.make_source_builder(
    BremsstrahlungHeatSink, runtime_params_type=RuntimeParams
)
