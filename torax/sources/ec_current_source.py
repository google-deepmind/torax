from __future__ import annotations

import dataclasses

import chex
import jax
from jax import numpy as jnp

from torax import constants, geometry, jax_utils, physics, state
from torax.config import config_args, runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


def ecrh_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    unused_model_func: source_models.SourceModels | None,
) -> jax.Array:
    """Model function for the ECRH current source."""
    assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

    flux_surface_averaged_ec_power_density = None

    return (
        dynamic_source_runtime_params.global_efficiency
        * flux_surface_averaged_ec_power_density
        * (
            dynamic_runtime_params_slice.profile_conditions.Te
            / dynamic_runtime_params_slice.profile_conditions.ne
        )
    )


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    # Global current drive efficiency (commonly referred to as zeta_cd)
    # Units: 10^13 A eV^-1 W^-1 m^-3
    # SI units: 6.242e31 s5 m^-7 kg^-2 A
    # Default value from Section 3.6 in Tholerus et al. (2024)
    #   doi: 10.1088/1741-4326/ad6ea2
    global_efficiency: float = 18.2

    def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
        return DynamicRuntimeParams(
            **config_args.get_init_kwargs(
                input_config=self,
                output_type=DynamicRuntimeParams,
                t=t,
            )
        )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
    global_efficiency: float


@dataclasses.dataclass(kw_only=True)
class ECRHCurrentSource(source.SingleProfilePsiSource):
    """ECRH current density source for the psi equation."""

    supported_modes: tuple[runtime_params_lib.Mode, ...] = (
        runtime_params_lib.Mode.ZERO,
        runtime_params_lib.mode.MODEL,
        runtime_params_lib.mode.PRESCRIBED,
    )
    model_func: source.SourceProfileFunction = ecrh_model_func


ECRHCurrentSourceBuilder = source.make_source_builder(
    ECRHCurrentSource, runtime_params_type=RuntimeParams
)
