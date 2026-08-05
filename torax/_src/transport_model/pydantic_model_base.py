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

"""Base pydantic config for Transport models."""
import abc
from typing import Annotated

import chex
import numpy as np
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params
from torax._src.transport_model import transport_model
import typing_extensions


# pylint: disable=invalid-name
class TransportBase(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base model holding parameters common to all transport models.

  Attributes:
    rho_min: normalized radius above which this model is applied.
    rho_max: normalized radius below which this model is applied.
    disable_chi_i: If True, sets the ion heat conductivity output to zero.
    disable_chi_e: If True, sets the electron heat conductivity output to zero.
    disable_D_e: If True, sets the electron diffusivity output to zero.
    disable_V_e: If True, sets the electron convection output to zero.
    fast_ion_stabilization: If True, applies fast ion stabilization.
    fast_ion_stabilization_model: Fast ion stabilization model config.
    fast_ion_stabilization_multiplier: Fast ion stabilization multiplier.
    merge_mode: Defines how transport coefficients are combined within a
      CombinedTransportModel. 'add' (default) adds to the accumulated value.
      'overwrite' overwrites the previous value in this model's valid domain
      and prevents subsequent 'add' models in the sequence from modifying this
      region.
  """

  rho_min: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  rho_max: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  disable_chi_i: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  disable_chi_e: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  disable_D_e: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  disable_V_e: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  fast_ion_stabilization: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  fast_ion_stabilization_model: Annotated[
      dict[str, str],
      torax_pydantic.JAX_STATIC,
  ] = {}
  fast_ion_stabilization_multiplier: float = 2.0
  merge_mode: Annotated[enums.MergeMode, torax_pydantic.JAX_STATIC] = (
      enums.MergeMode.ADD
  )

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    # For the time-varying parameter pair (rho_min, rho_max), we have relative
    # magnitude constraints that must hold at all times. We validate this by
    # checking the inequality at the combined time points (knots) of the pair.
    # This check is sufficient to guarantee the inequality at all times
    # provided that both parameters use the same interpolation mode (either step
    # or linear interpolation). We therefore require their interpolation modes
    # to match.
    if self.rho_max.interpolation_mode != self.rho_min.interpolation_mode:
      raise ValueError(
          'rho_max and rho_min must have the same interpolation mode.'
      )
    all_times_min_max = np.union1d(self.rho_min.time, self.rho_max.time)
    if not np.all(
        self.rho_max.get_value(all_times_min_max)
        > self.rho_min.get_value(all_times_min_max)
    ):
      raise ValueError('rho_max must be greater than rho_min for all time.')
    return self

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    return runtime_params.RuntimeParams(
        fast_ion_stabilization=self.fast_ion_stabilization.get_value(t),
        fast_ion_stabilization_model=tuple(
            sorted(
                self.fast_ion_stabilization_model.items()
            )
        ),
        fast_ion_stabilization_multiplier=self.fast_ion_stabilization_multiplier,
        rho_min=self.rho_min.get_value(t),
        rho_max=self.rho_max.get_value(t),
        disable_chi_i=self.disable_chi_i.get_value(t),
        disable_chi_e=self.disable_chi_e.get_value(t),
        disable_D_e=self.disable_D_e.get_value(t),
        disable_V_e=self.disable_V_e.get_value(t),
        merge_mode=self.merge_mode,
    )

  @abc.abstractmethod
  def build_transport_model(self) -> transport_model.TransportModel:
    """Builds a transport model from the config."""
