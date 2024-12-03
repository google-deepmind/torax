"""Basic impurity radiation heat sink for electron heat equation.."""

import dataclasses

import chex
import jax
import jax.numpy as jnp

from torax import array_typing
from torax import geometry
from torax import math_utils
from torax import state
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib

SOURCE_NAME = "impurity_radiation_heat_sink"


def _radially_constant_fraction_of_Pin(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
    """Model function for radiation heat sink from impurities.

    This model represents a sink in the temp_el equation, whose value is a fixed
    % of the total heating power input."""
    # Based on source_models.sum_sources_temp_el and source_models.calc_and_sum_sources_psi,
    # but only summing over heating *input* sources (Pohm + Paux + Palpha + ...)
    # and summing over *both* ion and electron heating

    def get_heat_source_profile(source_name: str, source: source_lib.Source) -> jax.Array:
        # TODO: Currently this recomputes the profile for each source, which is inefficient
        # (and will be a problem if sources are slow/non-jittable)
        # A similar TODO is noted in source_models.calc_and_sum_sources_psi
        profile = source.get_value(
            dynamic_runtime_params_slice,
            dynamic_runtime_params_slice.sources[source_name],
            geo,
            core_profiles,
        )
        return source.get_source_profile_for_affected_core_profile(
            profile, source_lib.AffectedCoreProfile.TEMP_EL.value, geo
        ) + source.get_source_profile_for_affected_core_profile(
            profile, source_lib.AffectedCoreProfile.TEMP_ION.value, geo
        )

    # Calculate the total power input to the heat equations
    heat_sources_and_sinks = source_models.temp_el_sources | source_models.temp_ion_sources
    heat_sources = {k: v for k, v in heat_sources_and_sinks.items() if not "sink" in k}
    source_profiles = jax.tree.map(
        get_heat_source_profile,
        list(heat_sources.keys()),
        list(heat_sources.values()),
    )
    Qtot_in = jnp.sum(jnp.stack(source_profiles), axis=0)
    Ptot_in = math_utils.cell_integration(Qtot_in * geo.vpr, geo)
    Vtot = geo.volume_face[-1]

    # Calculate the heat sink as a fraction of the total power input
    return (
        -dynamic_source_runtime_params.fraction_of_total_power_density
        * Ptot_in / Vtot
        * jnp.ones_like(geo.rho)
    )


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    fraction_of_total_power_density: runtime_params_lib.TimeInterpolatedInput = 0.1
    mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

    def make_provider(
        self,
        torax_mesh: geometry.Grid1D | None = None,
    ) -> "RuntimeParamsProvider":
        return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    """Provides runtime parameters for a given time and geometry."""

    runtime_params_config: RuntimeParams

    def build_dynamic_params(
        self,
        t: chex.Numeric,
    ) -> "DynamicRuntimeParams":
        return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
    fraction_of_total_power_density: array_typing.ScalarFloat


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ImpurityRadiationHeatSink(source_lib.Source):
    """Impurity radiation heat sink for electron heat equation."""

    source_models: source_models_lib.SourceModels
    model_func: source_lib.SourceProfileFunction = _radially_constant_fraction_of_Pin

    @property
    def supported_modes(self) -> tuple[runtime_params_lib.Mode, ...]:
        """Returns the modes supported by this source."""
        return (
            runtime_params_lib.Mode.ZERO,
            runtime_params_lib.Mode.MODEL_BASED,
            runtime_params_lib.Mode.PRESCRIBED,
        )

    @property
    def affected_core_profiles(self) -> tuple[source_lib.AffectedCoreProfile, ...]:
        return (source_lib.AffectedCoreProfile.TEMP_EL,)
