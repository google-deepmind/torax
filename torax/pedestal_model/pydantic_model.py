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

import enum
from typing import Any

import pydantic
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@enum.unique
class PedestalType(enum.Enum):
  """Types of time step calculators."""

  SET_PPED_TPEDRATIO_NPED = 'set_pped_tpedratio_nped'
  SET_TPED_NPED = 'set_tped_nped'


class SetPpedTpedRatioNped(torax_pydantic.BaseModelMutable):
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
  Pped: torax_pydantic.Pascal = 10.0
  neped: torax_pydantic.Density = 0.7
  neped_is_fGW: bool = False
  ion_electron_temperature_ratio: torax_pydantic.OpenUnitInterval = 1.0
  rho_norm_ped_top: torax_pydantic.UnitInterval = 0.91


class SetTpedNped(torax_pydantic.BaseModelMutable):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    neped: The electron density at the pedestal in units of nref or fGW.
    neped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    Tiped: Ion temperature at the pedestal [keV].
    Teped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """
  neped: torax_pydantic.Density = 0.7
  neped_is_fGW: bool = False
  Tiped: torax_pydantic.KiloElectronVolt = 5.0
  Teped: torax_pydantic.KiloElectronVolt = 5.0
  rho_norm_ped_top: torax_pydantic.UnitInterval = 0.91


class PedestalModel(torax_pydantic.BaseModelMutable):
  """Config for a time step calculator."""

  pedestal_model: PedestalType
  pedestal_config: SetPpedTpedRatioNped | SetTpedNped

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'pedestal_model' in data and isinstance(
        data['pedestal_model'], PedestalType
    ):
      return data

    if 'pedestal_model' in data:
      pedestal_model = data['pedestal_model']
      pedestal_config = {
          k: v
          for k, v in data['pedestal_config'].items()
          if k != 'pedestal_model'
      }
    else:
      # If the pedestal model is not specified, use set_tped_nped by default.
      pedestal_model = PedestalType.SET_TPED_NPED.value
      pedestal_config = data

    constructor_args = {'pedestal_model': PedestalType(pedestal_model)}
    match pedestal_model:
      case PedestalType.SET_PPED_TPEDRATIO_NPED.value:
        constructor_args['pedestal_config'] = SetPpedTpedRatioNped.from_dict(
            pedestal_config
        )
      case PedestalType.SET_TPED_NPED.value:
        constructor_args['pedestal_config'] = SetTpedNped.from_dict(
            pedestal_config
        )
      case _:
        raise ValueError(f'Unknown pedestal model: {pedestal_model}')

    return constructor_args
