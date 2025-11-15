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

"""Pydantic config for Pedestal."""
import abc
from typing import Annotated, Any, Literal

import chex
from torax._src.pedestal_model import custom_pedestal
from torax._src.pedestal_model import no_pedestal
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.pedestal_model import set_tped_nped
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal. Can be
      time varying.
  """

  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )

  @abc.abstractmethod
  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the pedestal model."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params."""


class SetPpedTpedRatioNped(BasePedestal):
  """Model for direct specification of pressure, temperature ratio, and density.

  Attributes:
    P_ped: The plasma pressure at the pedestal [Pa].
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal['set_P_ped_n_ped'], torax_pydantic.JAX_STATIC
  ] = 'set_P_ped_n_ped'
  P_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(1e5)
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> (
      set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
  ):
    return (
        set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel()
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_pped_tpedratio_nped.RuntimeParams:
    return set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        P_ped=self.P_ped.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


class SetTpedNped(BasePedestal):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_ped: Ion temperature at the pedestal [keV].
    T_e_ped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal['set_T_ped_n_ped'], torax_pydantic.JAX_STATIC
  ] = 'set_T_ped_n_ped'
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  T_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
    return set_tped_nped.SetTemperatureDensityPedestalModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.RuntimeParams:
    return set_tped_nped.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


class NoPedestal(BasePedestal):
  """A pedestal model for when there is no pedestal.

  Note that setting `set_pedestal` to True with a NoPedestal model is the
  equivalent of setting it to False.
  """

  model_name: Annotated[Literal['no_pedestal'], torax_pydantic.JAX_STATIC] = (
      'no_pedestal'
  )

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
    )


class CustomPedestal(BasePedestal):
  """Custom pedestal model using user-defined callable functions.

  This configuration allows users to provide callable functions that compute
  pedestal values based on runtime parameters, geometry, and core profiles.
  This enables machine-specific scaling laws without modifying source code.

  Attributes:
    T_i_ped_fn: Callable function to compute ion temperature at pedestal [keV].
    T_e_ped_fn: Callable function to compute electron temperature at pedestal
      [keV].
    n_e_ped_fn: Callable function to compute electron density at pedestal
      [m^-3 or fGW].
    rho_norm_ped_top_fn: Optional callable function to compute pedestal top
      location. If None, uses rho_norm_ped_top value.
    rho_norm_ped_top: The location of the pedestal top (used if
      rho_norm_ped_top_fn is None).
    n_e_ped_is_fGW: Whether the electron density returned by n_e_ped_fn is in
      units of fGW.

  Example:
    ```python
    def my_T_e_ped(runtime_params, geo, core_profiles):
      Ip_MA = runtime_params.profile_conditions.Ip / 1e6
      return 0.5 * (Ip_MA ** 0.2) * (geo.B0 ** 0.8)

    def my_T_i_ped(runtime_params, geo, core_profiles):
      T_e = my_T_e_ped(runtime_params, geo, core_profiles)
      return 1.2 * T_e

    def my_n_e_ped(runtime_params, geo, core_profiles):
      return 0.7  # 0.7 * nGW

    config = {
        'pedestal': {
            'model_name': 'custom',
            'T_i_ped_fn': my_T_i_ped,
            'T_e_ped_fn': my_T_e_ped,
            'n_e_ped_fn': my_n_e_ped,
            'rho_norm_ped_top': 0.91,
            'n_e_ped_is_fGW': True,
        }
    }
    ```
  """

  model_name: Annotated[Literal['custom'], torax_pydantic.JAX_STATIC] = (
      'custom'
  )
  T_i_ped_fn: Any  # Callable - Pydantic doesn't validate callables well
  T_e_ped_fn: Any  # Callable
  n_e_ped_fn: Any  # Callable
  rho_norm_ped_top_fn: Any | None = None  # Optional callable
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )
  n_e_ped_is_fGW: bool = False

  def build_pedestal_model(
      self,
  ) -> custom_pedestal.CustomPedestalModel:
    return custom_pedestal.CustomPedestalModel(
        T_i_ped_fn=self.T_i_ped_fn,
        T_e_ped_fn=self.T_e_ped_fn,
        n_e_ped_fn=self.n_e_ped_fn,
        rho_norm_ped_top_fn=self.rho_norm_ped_top_fn,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> custom_pedestal.RuntimeParams:
    return custom_pedestal.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
    )


PedestalConfig = (
    SetPpedTpedRatioNped | SetTpedNped | CustomPedestal | NoPedestal
)
