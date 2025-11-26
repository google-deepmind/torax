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
from __future__ import annotations

import abc
from typing import Annotated, Any, Dict, Literal, Union

import chex
import jax.numpy as jnp
import pydantic
from torax._src.pedestal_model import no_pedestal
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.pedestal_model import set_tped_nped
from torax._src.pedestal_policy import constant
from torax._src.pedestal_policy import pedestal_policy as pedestal_policy_lib
from torax._src.pedestal_policy import runtime_params as pedestal_policy_runtime_params
from torax._src.pedestal_policy import time_varying
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class SetPedestalConstant(torax_pydantic.BaseModelFrozen):
  """Policy to control pedestal on/off state with constant values."""

  use_pedestal: bool
  scale_pedestal: float | None = None
  policy_type: Annotated[Literal['constant'], torax_pydantic.JAX_STATIC] = (
      'constant'
  )

  def build_pedestal_policy(self) -> constant.Constant:
    return constant.Constant()

  def build_pedestal_policy_runtime_params(
      self,
  ) -> pedestal_policy_runtime_params.ConstantRP:
    return pedestal_policy_runtime_params.ConstantRP(
        use_pedestal=jnp.array(self.use_pedestal),
        scale_pedestal=jnp.array(self.scale_pedestal)
        if self.scale_pedestal is not None
        else None,
    )


class SetPedestalTimeVarying(torax_pydantic.BaseModelFrozen):
  """Policy to control pedestal on/off state, can be constant or time-varying."""

  value: interpolated_param_1d.TimeVaryingScalar
  policy_type: Annotated[Literal['time_varying'], torax_pydantic.JAX_STATIC] = (
      'time_varying'
  )

  def build_pedestal_policy(self) -> time_varying.TimeVarying:
    return time_varying.TimeVarying()

  def build_pedestal_policy_runtime_params(
      self,
  ) -> pedestal_policy_runtime_params.TimeVaryingRP:
    if not self.value.is_bool_param:
      raise ValueError('TimeVarying policy for set_pedestal must be boolean')
    return pedestal_policy_runtime_params.TimeVaryingRP(
        time=jnp.array(self.value.time),
        value=jnp.array(self.value.value),
        is_bool_param=self.value.is_bool_param,
        interpolation_mode=self.value.interpolation_mode,
    )


# Union of all configurable pedestal policy types
SetPedestalPolicy = Annotated[
    Union[SetPedestalConstant, SetPedestalTimeVarying],
    pydantic.Field(discriminator='policy_type'),
]


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models."""

  set_pedestal: SetPedestalPolicy = torax_pydantic.ValidatedDefault(
      {'policy_type': 'constant', 'use_pedestal': False}
  )

  @pydantic.field_validator('set_pedestal', mode='before')
  @classmethod
  def _validate_set_pedestal_policy(cls, v: Any) -> Dict[str, Any]:
    if isinstance(v, dict) and 'policy_type' in v:
      return v

    # Intercept simple (non-policy) types of set_pedestal and upgrade them
    # to policies
    if isinstance(v, bool):
      return {'policy_type': 'constant', 'use_pedestal': v}
    elif isinstance(v, (dict, tuple)):
      return {'policy_type': 'time_varying', 'value': v}
    elif isinstance(v, interpolated_param_1d.TimeVaryingScalar):
      return {'policy_type': 'time_varying', 'value': v.model_dump()}
    elif v is None:
      return {'policy_type': 'constant', 'use_pedestal': False}
    else:
      raise TypeError(
          f'Invalid type or format for set_pedestal: {type(v)}, value: {v}'
      )

  def build_pedestal_policy(self) -> pedestal_policy_lib.PedestalPolicy:
    return self.set_pedestal.build_pedestal_policy()

  def build_pedestal_policy_runtime_params(
      self,
  ) -> pedestal_policy_runtime_params.PedestalPolicyRuntimeParams:
    return self.set_pedestal.build_pedestal_policy_runtime_params()

  @abc.abstractmethod
  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the pedestal model."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params for the pedestal model itself."""


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
        pedestal_policy=pedestal_policy,
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
    pedestal_policy = self.set_pedestal.build_pedestal_policy()
    return set_tped_nped.SetTemperatureDensityPedestalModel(
        pedestal_policy=pedestal_policy,
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
  # Override the default policy to ensure it's always off for this model.
  set_pedestal: SetPedestalPolicy = torax_pydantic.ValidatedDefault(
      {'policy_type': 'constant', 'use_pedestal': False}
  )

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal(pedestal_policy=constant.Constant())

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams()

  def build_pedestal_policy_runtime_params(
      self,
  ) -> pedestal_policy_runtime_params.ConstantRP:
    return pedestal_policy_runtime_params.ConstantRP(
        use_pedestal=jnp.array(False),
        scale_pedestal=None,
    )


PedestalConfig = SetPpedTpedRatioNped | SetTpedNped | NoPedestal
