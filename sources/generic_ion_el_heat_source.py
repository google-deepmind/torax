"""Generic ion/electron heat source."""

# Updated with absorption_fraction parameter that allows specifying what fraction of power is absorbed

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal

import chex
from torax import array_typing
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic

def calc_generic_heat_source(
    geo: geometry.Geometry,
    rsource: float,
    w: float,
    Ptot: float,
    el_heat_fraction: float,
    absorption_fraction: float = 1.0,
) -> tuple[chex.Array, chex.Array]:
  """Computes ion/electron heat source terms.

  Args:
    geo: Geometry describing the torus
    rsource: Gaussian center in normalized r
    w: Gaussian width
    Ptot: total heating
    el_heat_fraction: fraction of heating deposited on electrons
    absorption_fraction: fraction of absorbed power

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """
  # Calculate heat profile.
  absorbed_power = Ptot * absorption_fraction
  profile = formulas.gaussian_profile(geo, center=rsource, width=w, total=absorbed_power)
  source_ion = profile * (1 - el_heat_fraction)
  source_el = profile * el_heat_fraction

  return (source_ion, source_el)

class GenericIonElHeatSourceConfig(base.SourceModelBase):
  """Configuration for the GenericIonElHeatSource.

  Attributes:
    w: Gaussian width in normalized radial coordinate
    rsource: Source Gaussian central location (in normalized r)
    Ptot: Total heating: high default based on total ITER power including alphas
    el_heat_fraction: Electron heating fraction
    absorption_fraction: Fraction of absorbed power
  """
  source_name: Literal['generic_ion_el_heat_source'] = (
      'generic_ion_el_heat_source'
  )
  w: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.25)
  rsource: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(0.0)
  Ptot: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(120e6)
  el_heat_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.66666)
  )
  # TODO(b/817): Add appropriate pydantic validation for absorption_fraction
  # to ensure it's never below a small positive value to prevent division by zero.
  absorption_fraction: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED 

def default_formula(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    core_sources: source_profiles.SourceProfiles | None = None,
) -> tuple[chex.Array, chex.Array]:
  """Default formula for the generic ion/electron heat source.

  Args:
    static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Dynamic runtime parameters.
    geo: Geometry describing the torus.
    source_name: Name of the source.
    core_profiles: Core profiles.
    core_sources: Source profiles.

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """
  del static_runtime_params_slice, core_profiles, core_sources  # Unused.
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[source_name]
  ion, el = calc_generic_heat_source(
      geo,
      dynamic_source_runtime_params.rsource,
      dynamic_source_runtime_params.w,
      dynamic_source_runtime_params.Ptot,
      dynamic_source_runtime_params.el_heat_fraction,
      dynamic_source_runtime_params.absorption_fraction,
  )
  return (ion, el)

@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime parameters for the generic heat source."""
  w: array_typing.ScalarFloat
  rsource: array_typing.ScalarFloat
  Ptot: array_typing.ScalarFloat
  el_heat_fraction: array_typing.ScalarFloat
  absorption_fraction: array_typing.ScalarFloat 

class GenericIonElectronHeatSource(source.Source):
  """Generic ion/electron heat source."""

  SOURCE_NAME: ClassVar[str] = 'generic_ion_el_heat_source'

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    """Returns the core profiles affected by this source."""
    return (
        source.AffectedCoreProfile.TEMP_ION,
        source.AffectedCoreProfile.TEMP_EL,
    ) 