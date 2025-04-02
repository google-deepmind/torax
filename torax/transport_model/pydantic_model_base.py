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
import chex
import pydantic
from torax.torax_pydantic import interpolated_param_1d
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import runtime_params
import typing_extensions


# pylint: disable=invalid-name
class TransportBase(torax_pydantic.BaseModelFrozen):
  """Base model holding parameters common to all transport models.

  Attributes:
    chimin: Lower bound on heat conductivity.
    chimax: Upper bound on heat conductivity (can be helpful for stability).
    Demin: minimum electron density diffusivity.
    Demax: maximum electron density diffusivity.
    Vemin: minimum electron density convection.
    Vemax: minimum electron density convection.
    apply_inner_patch: set inner core transport coefficients (ad-hoc MHD/EM
      transport).
    De_inner: inner core electron density diffusivity.
    Ve_inner: inner core electron density convection.
    chii_inner: inner core ion heat equation diffusion term.
    chie_inner: inner core electron heat equation diffusion term.
    rho_inner: normalized radius below which inner patch is applied.
    apply_outer_patch: set outer core transport coefficients (ad-hoc MHD/EM
      transport). Only used when set_pedestal = False Useful for L-mode
      near-edge region where QLKNN10D is not applicable.
    De_outer: outer core electron density diffusivity.
    Ve_outer: outer core electron density convection.
    chii_outer: outer core ion heat equation diffusion term.
    chie_outer: outer core electron heat equation diffusion term.
    rho_outer: normalized radius above which outer patch is applied.
    smoothing_sigma: Width of HWHM Gaussian smoothing kernel operating on
      transport model outputs.
    smooth_everywhere: Smooth across entire radial domain regardless of inner
      and outer patches.
  """

  chimin: torax_pydantic.MeterSquaredPerSecond = 0.05
  chimax: torax_pydantic.MeterSquaredPerSecond = 100.0
  Demin: torax_pydantic.MeterSquaredPerSecond = 0.05
  Demax: torax_pydantic.MeterSquaredPerSecond = 100.0
  Vemin: torax_pydantic.MeterPerSecond = -50.0
  Vemax: torax_pydantic.MeterPerSecond = 50.0
  apply_inner_patch: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  De_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  Ve_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chii_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_inner: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_inner: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  apply_outer_patch: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  De_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  Ve_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chii_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_outer: interpolated_param_1d.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_outer: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.9)
  )
  smoothing_sigma: pydantic.NonNegativeFloat = 0.0
  smooth_everywhere: bool = False

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.chimax > self.chimin:
      raise ValueError('chimin must be less than chimax.')
    if not self.Demin < self.Demax:
      raise ValueError('Demin must be less than Demax.')
    if not self.Vemin < self.Vemax:
      raise ValueError('Vemin must be less than Vemax.')
    return self

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> runtime_params.DynamicRuntimeParams:
    return runtime_params.DynamicRuntimeParams(
        chimin=self.chimin,
        chimax=self.chimax,
        Demin=self.Demin,
        Demax=self.Demax,
        Vemin=self.Vemin,
        Vemax=self.Vemax,
        apply_inner_patch=self.apply_inner_patch.get_value(t),
        De_inner=self.De_inner.get_value(t),
        Ve_inner=self.Ve_inner.get_value(t),
        chii_inner=self.chii_inner.get_value(t),
        chie_inner=self.chie_inner.get_value(t),
        rho_inner=self.rho_inner.get_value(t),
        apply_outer_patch=self.apply_outer_patch.get_value(t),
        De_outer=self.De_outer.get_value(t),
        Ve_outer=self.Ve_outer.get_value(t),
        chii_outer=self.chii_outer.get_value(t),
        chie_outer=self.chie_outer.get_value(t),
        rho_outer=self.rho_outer.get_value(t),
        smoothing_sigma=self.smoothing_sigma,
        smooth_everywhere=self.smooth_everywhere,
    )
