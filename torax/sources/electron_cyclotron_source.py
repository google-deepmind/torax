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
from typing import ClassVar, Literal

import chex
import jax.numpy as jnp
from torax import array_typing
from torax import constants
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import base
from torax.sources import formulas
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_profiles
from torax.torax_pydantic import torax_pydantic

InterpolatedVarTimeRhoInput = (
    runtime_params_lib.interpolated_param.InterpolatedVarTimeRhoInput
)


class ElectronCyclotronSourceConfig(base.SourceModelBase):
  """Config for the electron-cyclotron source.

  Attributes:
    cd_efficiency: Local dimensionless current drive efficiency. Zeta from
      Lin-Liu, Chan, and Prater, 2003, eq 44
    manual_ec_power_density: Manual EC power density profile on the rho grid
    gaussian_ec_power_density_width: Gaussian EC power density profile width
    gaussian_ec_power_density_location: Gaussian EC power density profile
      location
    gaussian_ec_total_power: Gaussian EC total power
  """

  source_name: Literal["electron_cyclotron_source"] = (
      "electron_cyclotron_source"
  )
  cd_efficiency: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.2, 1.0: 0.2}})
  )
  manual_ec_power_density: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.0, 1.0: 0.0}})
  )
  gaussian_ec_power_density_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  gaussian_ec_power_density_location: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  gaussian_ec_total_power: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_heating_and_current

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=self.prescribed_values.get_value(t),
        cd_efficiency=self.cd_efficiency.get_value(t),
        manual_ec_power_density=self.manual_ec_power_density.get_value(t),
        gaussian_ec_power_density_width=self.gaussian_ec_power_density_width.get_value(
            t
        ),
        gaussian_ec_power_density_location=self.gaussian_ec_power_density_location.get_value(
            t
        ),
        gaussian_ec_total_power=self.gaussian_ec_total_power.get_value(t),
    )

  def build_source(self):
    return ElectronCyclotronSource(model_func=self.model_func)


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Runtime parameters for the electron-cyclotron source for a given time and geometry."""

  cd_efficiency: array_typing.ArrayFloat
  manual_ec_power_density: array_typing.ArrayFloat
  gaussian_ec_power_density_width: array_typing.ScalarFloat
  gaussian_ec_power_density_location: array_typing.ScalarFloat
  gaussian_ec_total_power: array_typing.ScalarFloat


def calc_heating_and_current(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Model function for the electron-cyclotron source.

  Based on Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003).
  See https://torax.readthedocs.io/en/latest/electron-cyclotron-derivation.html

  Args:
    unused_static_runtime_params_slice: Static runtime parameters.
    dynamic_runtime_params_slice: Global runtime parameters
    geo: Magnetic geometry.
    source_name: Name of the source.
    core_profiles: CoreProfiles component of the state.
    unused_calculated_source_profiles: Unused.

  Returns:
    2D array of electron cyclotron heating power density and current density.
  """
  dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
      source_name
  ]
  # Helps linter understand the type of dynamic_source_runtime_params.
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  # Construct the profile
  ec_power_density = (
      dynamic_source_runtime_params.manual_ec_power_density
      + formulas.gaussian_profile(
          center=dynamic_source_runtime_params.gaussian_ec_power_density_location,
          width=dynamic_source_runtime_params.gaussian_ec_power_density_width,
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

  return ec_power_density, j_ec_dot_B


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElectronCyclotronSource(source.Source):
  """Electron cyclotron source for the Te and Psi equations."""

  SOURCE_NAME: ClassVar[str] = "electron_cyclotron_source"
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = "calc_heating_and_current"
  model_func: source.SourceProfileFunction = calc_heating_and_current

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL, source.AffectedCoreProfile.PSI)
