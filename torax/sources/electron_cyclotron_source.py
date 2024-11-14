# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Electron cyclotron heating (prescribed Gaussian) and current drive (Lin-Liu model)."""

from __future__ import annotations

import dataclasses

import chex
import jax
import jax.numpy as jnp
from torax import array_typing
from torax import constants
from torax import geometry
from torax import interpolated_param
from torax import state
from torax.config import runtime_params_slice
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models

InterpolatedVarTimeRhoInput = (
    runtime_params_lib.interpolated_param.InterpolatedVarTimeRhoInput
)

SOURCE_NAME = "electron_cyclotron_source"


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the electron-cyclotron source."""

  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  # Local dimensionless current drive efficiency
  # Zeta from Lin-Liu, Chan, and Prater, 2003, eq 44
  cd_efficiency: InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0.0: {0.0: 0.2, 1.0: 0.2}}
  )

  # Manual EC power density profile on the rho grid; units [W/m^3]
  manual_ec_power_density: InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: {0.0: {0.0: 0.0, 1.0: 0.0}}
  )

  # Gaussian EC power density profile; dimensionless [rho_norm]
  gaussian_ec_power_density_width: runtime_params_lib.TimeInterpolatedInput = (
      0.1
  )
  gaussian_ec_power_density_location: (
      runtime_params_lib.TimeInterpolatedInput
  ) = 0.0
  gaussian_ec_total_power: runtime_params_lib.TimeInterpolatedInput = 0.0

  def make_provider(self, torax_mesh: geometry.Grid1D | None = None):
    if torax_mesh is None:
      raise ValueError(
          "torax_mesh is required for RuntimeParams.make_provider."
      )
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides runtime parameters for the electron-cyclotron source for a given time and geometry."""

  runtime_params_config: RuntimeParams
  cd_efficiency: interpolated_param.InterpolatedVarTimeRho
  manual_ec_power_density: interpolated_param.InterpolatedVarTimeRho
  gaussian_ec_power_density_width: interpolated_param.InterpolatedVarSingleAxis
  gaussian_ec_power_density_location: (
      interpolated_param.InterpolatedVarSingleAxis
  )
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
    unused_model_func: source_models.SourceModels,
) -> jax.Array:
  """Model function for the electron-cyclotron source.

  Based on Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003).
  See https://torax.readthedocs.io/en/latest/electron-cyclotron-derivation.html

  Args:
    dynamic_runtime_params_slice: Global runtime parameters
    dynamic_source_runtime_params: Specific runtime parameters for the
      electron-cyclotron source.
    geo: Magnetic geometry.
    core_profiles: CoreProfiles component of the state.
    unused_model_func: (unused) source models used in the simulation.

  Returns:
    2D array of electron cyclotron heating power density and current density.
  """
  # Helps linter understand the type of dynamic_source_runtime_params.
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  # Construct the profile
  ec_power_density = (
      dynamic_source_runtime_params.manual_ec_power_density
      + formulas.gaussian_profile(
          c1=dynamic_source_runtime_params.gaussian_ec_power_density_location,
          c2=dynamic_source_runtime_params.gaussian_ec_power_density_width,
          total=dynamic_source_runtime_params.gaussian_ec_total_power,
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
  # pylint: disable=invalid-name
  log_j_ec_dot_B = (
      jnp.log(2 * jnp.pi / geo.Rmaj)
      + 2 * jnp.log(constants.CONSTANTS.epsilon0)
      - 3 * jnp.log(constants.CONSTANTS.qe)
      + jnp.log(geo.F)
      + jnp.log(core_profiles.temp_el.value)
      + jnp.log(constants.CONSTANTS.keV2J)  # Convert Te to J
      - jnp.log(core_profiles.ne.value)
      - jnp.log(
          dynamic_runtime_params_slice.numerics.nref
      )  # Convert ne to m^-3
      + jnp.log(dynamic_source_runtime_params.cd_efficiency)
      + jnp.log(ec_power_density)
  )
  j_ec_dot_B = jnp.exp(log_j_ec_dot_B)
  # pylint: enable=invalid-name

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
