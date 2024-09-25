from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp

from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.constants import CONSTANTS
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    """Runtime parameters for the electron-cyclotron current source."""

    # Local current drive efficiency
    dimensionless_efficiency: runtime_params_lib.TimeInterpolated

    # EC power density profile
    ec_power_density: runtime_params_lib.TimeInterpolated

    def make_provider(self, torax_mesh: geometry.Grid1D | None = None):
        if torax_mesh is None:
            raise ValueError("torax_mesh is required for RuntimeParams.make_provider.")
        return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    """Provides runtime parameters for the electron-cyclotron current source for a given time and geometry."""

    runtime_params_config: RuntimeParams
    ec_power_density: interpolated_param.InterpolatedVarTimeRho
    dimensionless_efficiency: interpolated_param.InterpolatedVarSingleAxis

    def build_dynamic_params(
        self,
        t: chex.Numeric,
    ) -> DynamicRuntimeParams:
        return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
    """Runtime parameters for the electron-cyclotron current source for a given time and geometry."""

    ec_power_density: array_typing.ArrayFloat
    dimensionless_efficiency: array_typing.ScalarFloat


def _calc_eccd_current(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jax.Array:
    """Model function for the ECRH current source."""
    assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

    # Source:
    # Electron cyclotron current drive efficiency in general tokamak geometry
    # Lin-Liu, Chan, and Prater, 2003
    # https://doi.org/10.1063/1.1610472

    ne_face_in_m3 = (
        core_profiles.ne.face_value() * dynamic_runtime_params_slice.numerics.nref
    )
    Te_face_in_eV = core_profiles.temp_el.face_value() * 1e3

    # Flux-surface averaged j profile
    return (
        2
        * jnp.pi
        * CONSTANTS.epsilon0**2
        / CONSTANTS.qe**3
        * dynamic_source_runtime_params.dimensionless_efficiency
        * dynamic_source_runtime_params.ec_power_density
        * Te_face_in_eV
        / ne_face_in_m3
    )


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElectronCyclotronCurrentSource(source.SingleProfilePsiSource):
    """Electron cyclotron current source for the Psi equation."""

    formula: source.SourceProfileFunction = _calc_eccd_current


ElectronCyclotronCurrentSourceBuilder = source.make_source_builder(
    ElectronCyclotronCurrentSource, runtime_params_type=RuntimeParams
)
