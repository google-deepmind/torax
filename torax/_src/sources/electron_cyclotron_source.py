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
import dataclasses
from typing import Annotated, ClassVar, Literal

import chex
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.sources import base
from torax._src.sources import formulas
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_profiles
from torax._src.torax_pydantic import torax_pydantic

# Default value for the model function to be used for the electron cyclotron
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = "gaussian_lin_liu"


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the electron-cyclotron source for a given time and geometry."""

  current_drive_efficiency: array_typing.FloatVector
  extra_prescribed_power_density: array_typing.FloatVector
  gaussian_width: array_typing.FloatScalar
  gaussian_location: array_typing.FloatScalar
  P_total: array_typing.FloatScalar


def calc_heating_and_current(
    runtime_params: runtime_params_slice.RuntimeParams,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[array_typing.FloatVectorCell, array_typing.FloatVectorCell]:
  """Model function for the electron-cyclotron source.

  Based on Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003).
  See https://torax.readthedocs.io/en/latest/electron-cyclotron-derivation.html

  Args:
    runtime_params: Global runtime parameters
    geo: Magnetic geometry.
    source_name: Name of the source.
    core_profiles: CoreProfiles component of the state.
    unused_calculated_source_profiles: Unused.

  Returns:
    2D array of electron cyclotron heating power density and current density.
  """
  source_params = runtime_params.sources[source_name]
  # Helps linter understand the type of source_params.
  assert isinstance(source_params, RuntimeParams)
  # Construct the profile
  ec_power_density = (
      source_params.extra_prescribed_power_density
      + formulas.gaussian_profile(
          center=source_params.gaussian_location,
          width=source_params.gaussian_width,
          total=source_params.P_total,
          geo=geo,
      )
  )

  # Compute j.B via the log for numerical stability
  # This is equivalent to:
  # <j_ec.B> = (
  #     2 * pi * epsilon0**2
  #     / (qe**3 * R_maj)
  #     * F
  #     * T_e [J] / n_e [m^-3]
  #     * current_drive_efficiency
  #     * ec_power_density
  # )
  # pylint: disable=invalid-name
  log_j_ec_dot_B = (
      jnp.log(2 * jnp.pi / geo.R_major)
      + 2 * jnp.log(constants.CONSTANTS.epsilon_0)
      - 3 * jnp.log(constants.CONSTANTS.q_e)
      + jnp.log(geo.F)
      + jnp.log(core_profiles.T_e.value)
      + jnp.log(constants.CONSTANTS.keV_to_J)  # Convert T_e to J
      - jnp.log(core_profiles.n_e.value)
      + jnp.log(source_params.current_drive_efficiency)
      + jnp.log(ec_power_density)
  )
  j_ec_dot_B = jnp.exp(log_j_ec_dot_B)
  # pylint: enable=invalid-name

  return ec_power_density, j_ec_dot_B


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class ElectronCyclotronSource(source.Source):
  """Electron cyclotron source for the T_e and Psi equations."""

  SOURCE_NAME: ClassVar[str] = "ecrh"
  model_func: source.SourceProfileFunction = calc_heating_and_current

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(self) -> tuple[source.AffectedCoreProfile, ...]:
    return (source.AffectedCoreProfile.TEMP_EL, source.AffectedCoreProfile.PSI)


class ElectronCyclotronSourceConfig(base.SourceModelBase):
  """Config for the electron-cyclotron source.

  Attributes:
    current_drive_efficiency: Local dimensionless current drive efficiency. Zeta
      from Lin-Liu, Chan, and Prater, 2003, eq 44
    extra_prescribed_power_density: Manual EC power density profile on the rho
      grid
    gaussian_width: Gaussian EC power density profile width
    gaussian_location: Gaussian EC power density profile location
    P_total: Gaussian EC total power
  """

  model_name: Annotated[
      Literal["gaussian_lin_liu"], torax_pydantic.JAX_STATIC
  ] = "gaussian_lin_liu"
  current_drive_efficiency: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.2, 1.0: 0.2}})
  )
  extra_prescribed_power_density: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.0, 1.0: 0.0}})
  )
  gaussian_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )
  gaussian_location: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.0
  )
  mode: Annotated[runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params_lib.Mode.MODEL_BASED
  )

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_heating_and_current

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    return RuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
        current_drive_efficiency=self.current_drive_efficiency.get_value(t),
        extra_prescribed_power_density=self.extra_prescribed_power_density.get_value(
            t
        ),
        gaussian_width=self.gaussian_width.get_value(t),
        gaussian_location=self.gaussian_location.get_value(t),
        P_total=self.P_total.get_value(t),
    )

  def build_source(self):
    return ElectronCyclotronSource(model_func=self.model_func)
