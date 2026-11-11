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
from typing import Annotated, Self

import chex
import numpy as np
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import component
from torax._src.transport_model import enums
from torax._src.transport_model import runtime_params


# pylint: disable=invalid-name
class ComponentTransportConfigBase(torax_pydantic.ConfigBase):
  """Base model holding parameters common to all component transport models.

  Attributes:
    rho_min: Normalized radius $rho_norm$ above which this transport model is
      applied. For `rho_min > 0`, the model is active in the range `rho_min <
      rho <= rho_max`; for `rho_min == 0`, it is active in `rho_min <= rho <=
      rho_max`. `rho_min` and `rho_max` must share the same interpolation mode
      so that `rho_min < rho_max` holds at all times.
    rho_max: Normalized radius $rho_norm$ below which this transport model is
      applied. `rho_min` and `rho_max` must share the same interpolation mode so
      that `rho_min < rho_max` holds at all times.
    disable_chi_i: If `True`, disables contribution from ion heat conductivity
      `chi_i` to the combined transport coefficients.
    disable_chi_e: If `True`, disables contribution from electron heat
      conductivity `chi_e` to the combined transport coefficients.
    disable_D_e: If `True`, disables contribution from electron diffusivity
      `D_e` to the combined transport coefficients.
    disable_V_e: If `True`, disables contribution from electron convection `V_e`
      to the combined transport coefficients.
    fast_ion_stabilization: If `True`, applies a fast ion stabilization
      correction to the $R_over_L_Ti$ input of quasilinear transport models
      (QLKNN, TGLFNN, QuaLiKiz). Predicts an ITG threshold upshift factor from
      fast ion parameters ($s_hat$, $q$, $n_fi / n_e$, $T_fi / T_e$,
      $R_over_L_Tfi$) and divides the input $R_over_L_Ti$ by this factor. For
      multiple fast ion species, the product of per-species factors is used.
      Requires `numerics.enable_fast_ions` to be `True`.
    fast_ion_stabilization_model: Dictionary mapping fast ion species names
      (e.g., `'H'`, `'He3'`) to a registered fast ion stabilization model name
      (e.g., `'fast_ion_H_v1'`) or file path to a `.fistab` model file. If empty
      or omitted for a species, the default model for that species is used.
    fast_ion_stabilization_multiplier: Multiplier applied to the correction part
      of the fast ion stabilization factor before modifying $R_over_L_Ti$. The
      default of `2.0` accounts for stronger stabilization in full gyrokinetic
      simulations compared to the QuaLiKiz-based surrogate model.
    merge_mode: Combination mode for this component inside `TransportConfig`:
      `'add'` (default, adds to the accumulated value in regions not locked by
      an overwrite model) or `'overwrite'` (replaces and locks enabled transport
      channels in this model's active radial region).
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
  def _check_fields(self) -> Self:
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
  ) -> runtime_params.ComponentRuntimeParams:
    """Builds the runtime parameters for the transport model at time `t`."""
    return runtime_params.ComponentRuntimeParams(
        fast_ion_stabilization=self.fast_ion_stabilization.get_value(t),
        fast_ion_stabilization_model=tuple(
            sorted(self.fast_ion_stabilization_model.items())
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
  def build_transport_model(self) -> component.ComponentTransportModel:
    """Builds a transport model from the config."""
