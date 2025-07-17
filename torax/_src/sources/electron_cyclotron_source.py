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
from typing import ClassVar, Literal

import chex
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import collisions
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
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Runtime parameters for the electron-cyclotron source for a given time and geometry."""

  current_drive_efficiency: array_typing.ArrayFloat
  extra_prescribed_power_density: array_typing.ArrayFloat
  gaussian_width: array_typing.ScalarFloat
  gaussian_location: array_typing.ScalarFloat
  P_total: array_typing.ScalarFloat


def calc_heating_and_current(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    unused_calculated_source_profiles: source_profiles.SourceProfiles | None,
    unused_conductivity: conductivity_base.Conductivity | None,
) -> tuple[chex.Array, ...]:
  """Model function for the electron-cyclotron source.

  Based on Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003).
  See https://torax.readthedocs.io/en/latest/electron-cyclotron-derivation.html
  for more details.

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
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)

  # Build the EC power deposition profile
  ec_power_density = (
      dynamic_source_runtime_params.extra_prescribed_power_density
      + formulas.gaussian_profile(
          center=dynamic_source_runtime_params.gaussian_location,
          width=dynamic_source_runtime_params.gaussian_width,
          total=dynamic_source_runtime_params.P_total,
          geo=geo,
      )
  )

  # pylint: disable=invalid-name

  # j_tor = dI/dA
  j_tor_ec = (
      16
      * jnp.pi
      * constants.CONSTANTS.epsilon0**2
      * core_profiles.T_e.value
      * 1e3  # T_e in eV
      * dynamic_source_runtime_params.current_drive_efficiency
      * ec_power_density
      / (
          constants.CONSTANTS.qe**2
          * collisions.calculate_log_lambda_ee(
              core_profiles.T_e.value, core_profiles.n_e.value
          )
          * core_profiles.n_e.value  # n_e in m^-3
      )
  )

  # < 1/R > on cell grid, needed to convert j_tor to < j.B >
  # TODO: add to geometry.Geometry?
  # Numerical trick to avoid division by zero: dρ/dA = dρ²/dA * 1/(2ρ)
  darea_drho2 = jnp.diff(geo.area_face) / jnp.diff(geo.rho_face**2)
  fsa_1_R = geo.F * geo.g3 * darea_drho2 / (jnp.pi * geo.B_0)

  # < j.B >
  fsa_j_dot_B = (
      geo.F
      * fsa_1_R
      * (
          1
          + geo.g2
          * geo.g3
          / (
              16 * jnp.pi**4 * geometry.face_to_cell(core_profiles.q_face) ** 2
              + constants.CONSTANTS.eps
          )
      )
      * j_tor_ec
  )

  return ec_power_density, fsa_j_dot_B


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
    current_drive_efficiency: Dimensionless current drive efficiency profile
      defined on the cell grid, :math:`\zeta = \frac{e^2\ln\Lambda_{ee}}{8
      \varepsilon_0^2} \frac{1}{\left\langle\frac{1}{R}\right\rangle} \frac{n_e
      [\mathrm{m^{-3}}]}{T_e [\mathrm{eV}]} \frac{I_{ec}
      [\mathrm{A}]}{P_{ec} [\mathrm{W}]}`.
    extra_prescribed_power_density: Manual EC power density profile on the rho
      grid.
    gaussian_width: Gaussian EC power density profile width.
    gaussian_location: Gaussian EC power density profile location.
    P_total: Gaussian EC total power.
  """

  model_name: Literal["gaussian_lin_liu"] = "gaussian_lin_liu"
  current_drive_efficiency: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.25, 1.0: 0.25}})
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
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED

  @property
  def model_func(self) -> source.SourceProfileFunction:
    return calc_heating_and_current

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        prescribed_values=tuple(
            [v.get_value(t) for v in self.prescribed_values]
        ),
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
