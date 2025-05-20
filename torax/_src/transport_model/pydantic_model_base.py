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

import chex
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import runtime_params
from torax._src.transport_model import transport_model
import typing_extensions


# pylint: disable=invalid-name
class TransportBase(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base model holding parameters common to all transport models.

  Attributes:
    chi_min: Lower bound on heat conductivity.
    chi_max: Upper bound on heat conductivity (can be helpful for stability).
    D_e_min: minimum electron density diffusivity.
    D_e_max: maximum electron density diffusivity.
    V_e_min: minimum electron density convection.
    V_e_max: minimum electron density convection.
    apply_inner_patch: set inner core transport coefficients (ad-hoc MHD/EM
      transport).
    D_e_inner: inner core electron density diffusivity.
    V_e_inner: inner core electron density convection.
    chi_i_inner: inner core ion heat equation diffusion term.
    chi_e_inner: inner core electron heat equation diffusion term.
    rho_inner: normalized radius below which inner patch is applied.
    apply_outer_patch: set outer core transport coefficients (ad-hoc MHD/EM
      transport). Only used when pedestal.set_pedestal = False Useful for L-mode
      near-edge region where QLKNN10D is not applicable.
    D_e_outer: outer core electron density diffusivity.
    V_e_outer: outer core electron density convection.
    chi_i_outer: outer core ion heat equation diffusion term.
    chi_e_outer: outer core electron heat equation diffusion term.
    rho_outer: normalized radius above which outer patch is applied.
    smoothing_width: Width of HWHM Gaussian smoothing kernel operating on
      transport model outputs.
    smooth_everywhere: Smooth across entire radial domain regardless of inner
      and outer patches.
  """

  chi_min: torax_pydantic.MeterSquaredPerSecond = 0.05
  chi_max: torax_pydantic.MeterSquaredPerSecond = 100.0
  D_e_min: torax_pydantic.MeterSquaredPerSecond = 0.05
  D_e_max: torax_pydantic.MeterSquaredPerSecond = 100.0
  V_e_min: torax_pydantic.MeterPerSecond = -50.0
  V_e_max: torax_pydantic.MeterPerSecond = 50.0
  apply_inner_patch: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  D_e_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  V_e_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chi_i_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_e_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_inner: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  apply_outer_patch: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  D_e_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  V_e_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chi_i_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_e_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_outer: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.9)
  )
  smoothing_width: pydantic.NonNegativeFloat = 0.0
  smooth_everywhere: bool = False

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.chi_max > self.chi_min:
      raise ValueError('chi_min must be less than chi_max.')
    if not self.D_e_min < self.D_e_max:
      raise ValueError('D_e_min must be less than D_e_max.')
    if not self.V_e_min < self.V_e_max:
      raise ValueError('V_e_min must be less than V_e_max.')
    return self

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        chi_min=self.chi_min,
        chi_max=self.chi_max,
        D_e_min=self.D_e_min,
        D_e_max=self.D_e_max,
        V_e_min=self.V_e_min,
        V_e_max=self.V_e_max,
        apply_inner_patch=self.apply_inner_patch.get_value(t),
        D_e_inner=self.D_e_inner.get_value(t),
        V_e_inner=self.V_e_inner.get_value(t),
        chi_i_inner=self.chi_i_inner.get_value(t),
        chi_e_inner=self.chi_e_inner.get_value(t),
        rho_inner=self.rho_inner.get_value(t),
        apply_outer_patch=self.apply_outer_patch.get_value(t),
        D_e_outer=self.D_e_outer.get_value(t),
        V_e_outer=self.V_e_outer.get_value(t),
        chi_i_outer=self.chi_i_outer.get_value(t),
        chi_e_outer=self.chi_e_outer.get_value(t),
        rho_outer=self.rho_outer.get_value(t),
        smoothing_width=self.smoothing_width,
        smooth_everywhere=self.smooth_everywhere,
    )

  @abc.abstractmethod
  def build_transport_model(self) -> transport_model.TransportModel:
    """Builds a transport model from the config."""
