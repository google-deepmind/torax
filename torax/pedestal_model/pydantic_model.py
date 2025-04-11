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
import copy
from typing import Any, Literal
import chex
import pydantic
from torax.pedestal_model import no_pedestal
from torax.pedestal_model import pedestal_model
from torax.pedestal_model import runtime_params
from torax.pedestal_model import set_pped_tpedratio_nped
from torax.pedestal_model import set_tped_nped
from torax.torax_pydantic import interpolated_param_1d
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class SetPpedTpedRatioNped(torax_pydantic.BaseModelFrozen):
  """Model for direct specification of pressure, temperature ratio, and density.

  Attributes:
    Pped: The plasma pressure at the pedestal [Pa].
    neped: The electron density at the pedestal in units of nref or fGW.
    neped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    ion_electron_temperature_ratio: Ratio of the ion and electron temperature at
      the pedestal [dimensionless].
    rho_norm_ped_top: The location of the pedestal top.
  """
  pedestal_model: Literal['set_pped_tpedratio_nped']
  Pped: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e5)
  )
  neped: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7)
  )
  neped_is_fGW: bool = False
  ion_electron_temperature_ratio: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_norm_ped_top: interpolated_param_1d.TimeVaryingScalar = (
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

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> set_pped_tpedratio_nped.DynamicRuntimeParams:
    return set_pped_tpedratio_nped.DynamicRuntimeParams(
        Pped=self.Pped.get_value(t),
        neped=self.neped.get_value(t),
        neped_is_fGW=self.neped_is_fGW,
        ion_electron_temperature_ratio=self.ion_electron_temperature_ratio.get_value(
            t
        ),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


class SetTpedNped(torax_pydantic.BaseModelFrozen):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    neped: The electron density at the pedestal in units of nref or fGW.
    neped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    Tiped: Ion temperature at the pedestal [keV].
    Teped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """

  pedestal_model: Literal['set_tped_nped']
  neped: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7)
  )
  neped_is_fGW: bool = False
  Tiped: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  Teped: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  rho_norm_ped_top: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
    return set_tped_nped.SetTemperatureDensityPedestalModel()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.DynamicRuntimeParams:
    return set_tped_nped.DynamicRuntimeParams(
        neped=self.neped.get_value(t),
        neped_is_fGW=self.neped_is_fGW,
        Tiped=self.Tiped.get_value(t),
        Teped=self.Teped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
    )


class NoPedestal(torax_pydantic.BaseModelFrozen):
  """A pedestal model for when there is no pedestal.

  This is needed as under jax compilation we have to have a valid value for
  both branches of the jax.lax.cond. This provides that value whilst being very
  explicit about the fact that there is no pedestal and simple so minimal
  compilation time.
  """
  pedestal_model: Literal['no_pedestal'] = 'no_pedestal'

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> runtime_params.DynamicRuntimeParams:
    del t  # Unused.
    return runtime_params.DynamicRuntimeParams()


class Pedestal(torax_pydantic.BaseModelFrozen):
  """Config for a pedestal model."""

  pedestal_config: SetPpedTpedRatioNped | SetTpedNped | NoPedestal = (
      pydantic.Field(
          discriminator='pedestal_model',
          default_factory=SetTpedNped,
      )
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'pedestal_config' in data:
      return data

    pedestal_config = copy.deepcopy(data)
    if 'pedestal_model' not in data:
      pedestal_config['pedestal_model'] = 'set_tped_nped'

    return {'pedestal_config': pedestal_config}

  def build_pedestal_model(
      self,
  ) -> pedestal_model.PedestalModel:
    return self.pedestal_config.build_pedestal_model()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> runtime_params.DynamicRuntimeParams:
    return self.pedestal_config.build_dynamic_params(t)
