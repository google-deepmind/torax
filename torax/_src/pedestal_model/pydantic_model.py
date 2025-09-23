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
from typing import Annotated, Any, Dict, Literal, Union

import chex
import pydantic
from torax._src.pedestal_model import no_pedestal
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.pedestal_model import set_tped_nped
from torax._src.pedestal_policy import constant
from torax._src.pedestal_policy import time_varying
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class SetPedestalTimeVarying(torax_pydantic.BaseModelFrozen):
  """Policy to control pedestal on/off state, can be constant or time-varying."""

  value: interpolated_param_1d.TimeVaryingScalar
  policy_type: Annotated[Literal['time_varying'], torax_pydantic.JAX_STATIC] = (
      'time_varying'
  )

  def build_pedestal_policy(self):
    return time_varying.TimeVarying(self.value)


# Union of all configurable pedestal policy types
SetPedestalPolicy = Annotated[
    Union[SetPedestalTimeVarying,],
    pydantic.Field(discriminator='policy_type'),
]


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models.

  """

  set_pedestal: SetPedestalPolicy = torax_pydantic.ValidatedDefault(
      {'policy_type': 'time_varying', 'value': False}
  )

  @pydantic.field_validator('set_pedestal', mode='before')
  @classmethod
  def _validate_set_pedestal_policy(cls, v: Any) -> Dict[str, Any]:
    # If someone explicitly writes a TimeVaryingScalar policy dict, we
    # support that, and it doesn't require any transformation here.
    # Not expected that users will write this way though.
    if isinstance(v, dict) and 'policy_type' in v:
      return v

    # Intercept simple (non-policy) types of set_pedestal and upgrade them
    # to policies
    if isinstance(v, bool):
      policy_dict = {'policy_type': 'time_varying', 'value': v}
    elif isinstance(v, (dict, tuple)):
      policy_dict = {'policy_type': 'time_varying', 'value': v}
    elif isinstance(v, interpolated_param_1d.TimeVaryingScalar):
      policy_dict = {'policy_type': 'time_varying', 'value': v.model_dump()}
    elif v is None:
      # Default to False if not provided
      return {'policy_type': 'time_varying', 'value': False}
    else:
      raise TypeError(
          f'Invalid type or format for set_pedestal: {type(v)}, value: {v}'
      )

    return policy_dict

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
    pedestal_policy = self.set_pedestal.build_pedestal_policy()
    return set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel(
        pedestal_policy
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_pped_tpedratio_nped.RuntimeParams:
    return set_pped_tpedratio_nped.RuntimeParams(
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
    return set_tped_nped.SetTemperatureDensityPedestalModel(
        self.set_pedestal.build_pedestal_policy()
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.RuntimeParams:
    return set_tped_nped.RuntimeParams(
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


class NoPedestal(BasePedestal):
  """A pedestal model for when there is no pedestal.

  If any pedestal policy is specified using the `set_pedestal` field,
  it is ignored.
  """

  model_name: Annotated[Literal['no_pedestal'], torax_pydantic.JAX_STATIC] = (
      'no_pedestal'
  )

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal(pedestal_policy=constant.Constant(False))

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams(
    )


PedestalConfig = SetPpedTpedRatioNped | SetTpedNped | NoPedestal
