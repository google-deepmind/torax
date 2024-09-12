from __future__ import annotations

import dataclasses

import chex
import jax
from jax.scipy import integrate

from torax import geometry
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models


def ecrh_model_func(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    unused_model_func: source_models.SourceModels | None,
) -> jax.Array:
    """Model function for the ECRH current source."""
    assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

    c0 = 3.27e-3
    ec_power_density = dynamic_runtime_params_slice.ec_power_density
    ec_power = integrate.trapezoid(ec_power_density * geo.vpr, geo.rho_norm)
    ec_power_density_times_rho = ec_power_density * geo.rho_norm
    form_factor = ec_power_density_times_rho / integrate.trapezoid(
        ec_power_density_times_rho / geo.spr_cell, geo.rho_norm
    )

    return (
        dynamic_source_runtime_params.dimensionless_efficiency
        / (c0 * geo.rmid)
        * ec_power
        * form_factor
        * (
            dynamic_runtime_params_slice.profile_conditions.Te
            / dynamic_runtime_params_slice.profile_conditions.ne
        )
    )


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    # Current drive efficiency
    dimensionless_efficiency: float = 0.2

    # EC power density profile
    ec_power_density: runtime_params_lib.interpolated_param.TimeInterpolated

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
    dimensionless_efficiency: float
    ec_power_density: jax.Array


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
