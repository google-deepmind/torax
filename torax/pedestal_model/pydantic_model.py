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
import pydantic
from torax.torax_pydantic import interpolated_param_1d
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
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


class PedestalModel(torax_pydantic.BaseModelMutable):
  """Config for a pedestal model."""
  pedestal_config: SetPpedTpedRatioNped | SetTpedNped = pydantic.Field(
      discriminator='pedestal_model', default_factory=SetTpedNped,
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
