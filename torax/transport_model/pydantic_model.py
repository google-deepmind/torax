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

"""Pydantic config for Transport models."""

from typing import Any, Literal, Union

import pydantic
from torax.torax_pydantic import interpolated_param_1d
from torax.torax_pydantic import torax_pydantic
import typing_extensions


# pylint: disable=invalid-name
class TransportBase(torax_pydantic.BaseModelFrozen):
  """Base model holding parameters common to all transport models.

  Attributes:
    chimin: Lower bound on heat conductivity.
    chimax: Upper bound on heat conductivity (can be helpful for stability)
    Demin: minimum electron density diffusivity
    Demax: maximum electron density diffusivity
    Vemin: minimum electron density convection
    Vemax: minimum electron density convection
    apply_inner_patch: set inner core transport coefficients (ad-hoc MHD/EM
      transport)
    De_inner: inner core electron density diffusivity
    Ve_inner: inner core electron density convection
    chii_inner: inner core ion heat equation diffusion term
    chie_inner: inner core electron heat equation diffusion term
    rho_inner: normalized radius below which inner patch is applied
    apply_outer_patch: set outer core transport coefficients (ad-hoc MHD/EM
      transport). Only used when set_pedestal = False Useful for L-mode
      near-edge region where QLKNN10D is not applicable.
    De_outer: outer core electron density diffusivity
    Ve_outer: outer core electron density convection
    chii_outer: outer core ion heat equation diffusion term
    chie_outer: outer core electron heat equation diffusion term
    rho_outer: normalized radius above which outer patch is applied
    smoothing_sigma: Width of HWHM Gaussian smoothing kernel operating on
      transport model outputs
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
  De_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  Ve_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chii_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_inner: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  apply_outer_patch: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  De_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  Ve_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  chii_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_outer: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.9)
  )
  smoothing_sigma: float = 0.0
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


class QLKNNTransportModel(TransportBase):
  """Model for the QLKNN transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'qlknn'.
    include_ITG: Whether to include ITG modes.
    include_TEM: Whether to include TEM modes.
    include_ETG: Whether to include ETG modes.
    ITG_flux_ratio_correction: Correction factor for ITG electron heat flux.
    ETG_correction_factor: Correction factor for ETG electron heat flux.
      https://gitlab.com/qualikiz-group/QuaLiKiz/-/commit/5bcd3161c1b08e0272ab3c9412fec7f9345a2eef
    clip_inputs: Whether to clip inputs within desired margin of the QLKNN
      training set boundaries.
    clip_margin: Margin to clip inputs within desired margin of the QLKNN
      training set boundaries.
    coll_mult: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DVeff: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """
  transport_model: Literal['qlknn'] = 'qlknn'
  include_ITG: bool = True
  include_TEM: bool = True
  include_ETG: bool = True
  ITG_flux_ratio_correction: float = 1.0
  ETG_correction_factor: float = 1.0 / 3.0
  clip_inputs: bool = False
  clip_margin: float = 0.95
  coll_mult: float = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DVeff: bool = False
  An_min: pydantic.PositiveFloat = 0.05


class QualikizTransportModel(TransportBase):
  """Model for the Qualikiz transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'qualikiz'.
    maxruns: Set frequency of full QuaLiKiz contour solutions.
    numprocs: Set number of cores used QuaLiKiz calculations.
    coll_mult: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DVeff: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """
  transport_model: Literal['qualikiz'] = 'qualikiz'
  maxruns: pydantic.PositiveInt = 2
  numprocs: pydantic.PositiveInt = 8
  coll_mult: pydantic.PositiveFloat = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DVeff: bool = False
  An_min: pydantic.PositiveFloat = 0.05


class ConstantTransportModel(TransportBase):
  """Model for the Constant transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'constant'.
    chii_const: coefficient in ion heat equation diffusion term in m^2/s
    chie_const: coefficient in electron heat equation diffusion term in m^2/s
    De_const: diffusion coefficient in electron density equation in m^2/s
    Ve_const: convection coefficient in electron density equation in m^2/s
  """
  transport_model: Literal['constant'] = 'constant'
  chii_const: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_const: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  De_const: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  Ve_const: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(-0.33)
  )


class CriticalGradientTransportModel(TransportBase):
  """Model for the Critical Gradient transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'CGM'.
    alpha: Exponent of chi power law: chi ∝ (R/LTi - R/LTi_crit)^alpha
    chistiff: Stiffness parameter
    chiei_ratio: Ratio of electron to ion heat transport coefficient (ion higher
      for ITG)
    chi_D_ratio: Ratio of electron particle to ion heat transport coefficient
    VR_D_ratio: Ratio of major radius * electron particle convection, to
      electron diffusion. Sets the value of electron particle convection in the
      model.
  """
  transport_model: Literal['CGM'] = 'CGM'
  alpha: float = 2.0
  chistiff: float = 2.0
  chiei_ratio: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(2.0)
  )
  chi_D_ratio: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  VR_D_ratio: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )


class BohmGyroBohmTransportModel(TransportBase):
  """Model for the Bohm + Gyro-Bohm transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'bohm-gyrobohm'.
    chi_e_bohm_coeff: Prefactor for Bohm term for electron heat conductivity.
    chi_e_gyrobohm_coeff: Prefactor for GyroBohm term for electron heat
      conductivity.
    chi_i_bohm_coeff: Prefactor for Bohm term for ion heat conductivity.
    chi_i_gyrobohm_coeff: Prefactor for GyroBohm term for ion heat conductivity.
    d_face_c1: Constant for the electron diffusivity weighting factor.
    d_face_c2: Constant for the electron diffusivity weighting factor.
    v_face_coeff: Proportionality factor between convectivity and diffusivity.
  """
  transport_model: Literal['bohm-gyrobohm'] = 'bohm-gyrobohm'
  chi_e_bohm_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(8e-5)
  )
  chi_e_gyrobohm_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5e-6)
  )
  chi_i_bohm_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(8e-5)
  )
  chi_i_gyrobohm_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5e-6)
  )
  d_face_c1: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  d_face_c2: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  v_face_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(-0.1)
  )


TransportModelConfig = Union[
    QLKNNTransportModel,
    QualikizTransportModel,
    ConstantTransportModel,
    CriticalGradientTransportModel,
    BohmGyroBohmTransportModel,
]


class Transport(torax_pydantic.BaseModelFrozen):
  """Config for a transport model.

  The `from_dict` method of constructing this class supports the config
  described in: https://torax.readthedocs.io/en/latest/configuration.html
  """

  transport_model_config: TransportModelConfig = pydantic.Field(
      discriminator='transport_model'
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'transport_model_config' in data:
      return data

    return {'transport_model_config': data}
