from __future__ import annotations

import dataclasses
from dataclasses import field

import chex
import jax
import jax.numpy as jnp

from torax import array_typing
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.constants import CONSTANTS
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source

InterpolatedVarTimeRhoInput = (
    runtime_params_lib.interpolated_param.InterpolatedVarTimeRhoInput
)

SOURCE_NAME = "electron_cyclotron_source"


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
    """Runtime parameters for the electron-cyclotron source."""

    # Local dimensionless current drive efficiency
    # Zeta from Lin-Liu, Chan, and Prater, 2003, eq 44
    # TODO: support either a scalar or a profile; if scalar axis,
    # then produce profile by ones*value
    cd_efficiency: InterpolatedVarTimeRhoInput = field(
        default_factory=lambda: {0.0: {0.0: 0.2, 1.0: 0.2}}
    )

    # Manual EC power density profile on the rho grid; units [W/m^3]
    manual_ec_power_density: InterpolatedVarTimeRhoInput = field(
        default_factory=lambda: {0.0: {0.0: 0.0, 1.0: 0.0}}
    )

    # Gaussian EC power density profile; units [W/m^3]
    gaussian_ec_power_density_width: runtime_params_lib.TimeInterpolatedInput = 0.0
    gaussian_ec_power_density_location: runtime_params_lib.TimeInterpolatedInput = 0.0
    gaussian_ec_total_power: runtime_params_lib.TimeInterpolatedInput = 0.0

    def make_provider(self, torax_mesh: geometry.Grid1D | None = None):
        if torax_mesh is None:
            raise ValueError("torax_mesh is required for RuntimeParams.make_provider.")
        return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
    """Provides runtime parameters for the electron-cyclotron source for a given time and geometry."""

    runtime_params_config: RuntimeParams
    cd_efficiency: interpolated_param.InterpolatedVarTimeRho
    manual_ec_power_density: interpolated_param.InterpolatedVarTimeRho
    gaussian_ec_power_density_width: interpolated_param.InterpolatedVarSingleAxis
    gaussian_ec_power_density_location: interpolated_param.InterpolatedVarSingleAxis
    gaussian_ec_total_power: interpolated_param.InterpolatedVarSingleAxis

    def build_dynamic_params(
        self,
        t: chex.Numeric,
    ) -> DynamicRuntimeParams:
        return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
    """Runtime parameters for the electron-cyclotron source for a given time and geometry."""

    cd_efficiency: array_typing.ArrayFloat
    manual_ec_power_density: array_typing.ArrayFloat
    gaussian_ec_power_density_width: array_typing.ScalarFloat
    gaussian_ec_power_density_location: array_typing.ScalarFloat
    gaussian_ec_total_power: array_typing.ScalarFloat


def _calc_heating_and_current(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: runtime_params_lib.SourceModels,
) -> jax.Array:
    """Model function for the electron-cyclotron source.

    Returns:
      (ec_power_density, j_parallel_ec)
    """
    # Construct the profile
    ec_power_density = (
        dynamic_source_runtime_params.manual_ec_power_density
        + formulas.gaussian_profile(
            c1=dynamic_source_runtime_params.gaussian_ec_power_density_location,
            c2=dynamic_source_runtime_params.gaussian_ec_power_density_width,
            total=dynamic_source_runtime_params.gaussian_ec_total_power,
            use_normalized_r=True,
            geo=geo,
        )
    )

    # Compute j.B via the log for numerical stability
    # This is equivalent to:
    # <j_ec.B> = (
    #     2 * pi * epsilon0**2
    #     / (qe**3 * R_maj)
    #     * F
    #     * Te [J] / ne [m^-3]
    #     * cd_efficiency
    #     * ec_power_density
    # )
    log_j_ec_dot_B = (
        jnp.log(2 * jnp.pi / geo.Rmaj) * 2 * jnp.log(CONSTANTS.epsilon0)
        - 3 * jnp.log(CONSTANTS.qe)
        + jnp.log(geo.F)  # BxR
        + jnp.log(core_profiles.temp_el.value)
        + jnp.log(CONSTANTS.keV2J)  # Convert Te to J
        - jnp.log(core_profiles.ne.value)
        - jnp.log(dynamic_runtime_params_slice.numerics.nref)  # Convert ne to m^-3
        + jnp.log(dynamic_source_runtime_params.cd_efficiency)
        + jnp.log(ec_power_density)
    )
    j_ec_dot_B = jnp.exp(log_j_ec_dot_B)

    return jnp.stack([ec_power_density, j_ec_dot_B])


def _get_ec_output_shape(geo: geometry.Geometry) -> tuple[int, ...]:
    return (2,) + source.ProfileType.CELL.get_profile_shape(geo)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElectronCyclotronSource(source.Source):
    """Electron cyclotron source for the Te and Psi equations."""

    supported_modes: tuple[runtime_params_lib.Mode, ...] = (
        runtime_params_lib.Mode.ZERO,
        runtime_params_lib.Mode.MODEL_BASED,
    )

    model_func: source.SourceProfileFunction = _calc_heating_and_current

    @property
    def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
      return (source.AffectedCoreProfile.TEMP_EL, source.AffectedCoreProfile.PSI)

    @property
    def output_shape_getter(self) -> source.SourceOutputShapeFunction:
      return _get_ec_output_shape
