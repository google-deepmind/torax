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

import copy
import dataclasses
from typing import Any, Literal, Union

import chex
import pydantic
from torax.torax_pydantic import interpolated_param_1d
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import bohm_gyrobohm
from torax.transport_model import constant
from torax.transport_model import critical_gradient
from torax.transport_model import qlknn_transport_model
from torax.transport_model import runtime_params
from torax.transport_model import transport_model as transport_model_lib
import typing_extensions


# pylint: disable=g-import-not-at-top
try:
  from torax.transport_model import qualikiz_transport_model

  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = True
except ImportError:
  _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE = False
# pylint: enable=g-import-not-at-top


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
  rho_outer: torax_pydantic.UnitIntervalTimeVaryingScalar = (
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
  model_path: str = qlknn_transport_model.get_default_model_path()
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

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    if 'model_path' in data:
      model_path = data['model_path']
    else:
      model_path = qlknn_transport_model.get_default_model_path()
      data['model_path'] = model_path

    version = qlknn_transport_model.get_model(model_path).version
    if version == '10D':
      if 'coll_mult' not in data:
        # Correction factor to a more recent QLK collision operator.
        data['coll_mult'] = 0.25
      if 'ITG_flux_ratio_correction' not in data:
        # The QLK version this specific QLKNN was trained on tends to
        # underpredict ITG electron heat flux in shaped, high-beta scenarios.
        data['ITG_flux_ratio_correction'] = 2.0
    return data

  def build_transport_model(self) -> qlknn_transport_model.QLKNNTransportModel:
    return qlknn_transport_model.QLKNNTransportModel(model_path=self.model_path)

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> qlknn_transport_model.DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return qlknn_transport_model.DynamicRuntimeParams(
        include_ITG=self.include_ITG,
        include_TEM=self.include_TEM,
        include_ETG=self.include_ETG,
        ITG_flux_ratio_correction=self.ITG_flux_ratio_correction,
        ETG_correction_factor=self.ETG_correction_factor,
        clip_inputs=self.clip_inputs,
        clip_margin=self.clip_margin,
        coll_mult=self.coll_mult,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DVeff=self.DVeff,
        An_min=self.An_min,
        **base_kwargs,
    )


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

  # pylint: disable=undefined-variable
  def build_transport_model(
      self,
  ):
    return qualikiz_transport_model.QualikizTransportModel()

  def build_dynamic_params(self, t: chex.Numeric):
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return qualikiz_transport_model.DynamicRuntimeParams(
        maxruns=self.maxruns,
        numprocs=self.numprocs,
        coll_mult=self.coll_mult,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DVeff=self.DVeff,
        An_min=self.An_min,
        **base_kwargs,
    )

  # pylint: enable=undefined-variable


class ConstantTransportModel(TransportBase):
  """Model for the Constant transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'constant'.
    chii_const: coefficient in ion heat equation diffusion term in m^2/s.
    chie_const: coefficient in electron heat equation diffusion term in m^2/s.
    De_const: diffusion coefficient in electron density equation in m^2/s.
    Ve_const: convection coefficient in electron density equation in m^2/s.
  """

  transport_model: Literal['constant'] = 'constant'
  chii_const: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chie_const: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  De_const: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  Ve_const: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(-0.33)
  )

  def build_transport_model(self) -> constant.ConstantTransportModel:
    return constant.ConstantTransportModel()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> constant.DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return constant.DynamicRuntimeParams(
        chii_const=self.chii_const.get_value(t),
        chie_const=self.chie_const.get_value(t),
        De_const=self.De_const.get_value(t),
        Ve_const=self.Ve_const.get_value(t),
        **base_kwargs,
    )


class CriticalGradientTransportModel(TransportBase):
  """Model for the Critical Gradient transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'CGM'.
    alpha: Exponent of chi power law: chi ∝ (R/LTi - R/LTi_crit)^alpha.
    chistiff: Stiffness parameter.
    chiei_ratio: Ratio of electron to ion heat transport coefficient (ion higher
      for ITG).
    chi_D_ratio: Ratio of electron particle to ion heat transport coefficient.
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
  chi_D_ratio: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  VR_D_ratio: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )

  def build_transport_model(
      self,
  ) -> critical_gradient.CriticalGradientTransportModel:
    return critical_gradient.CriticalGradientTransportModel()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> critical_gradient.DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return critical_gradient.DynamicRuntimeParams(
        alpha=self.alpha,
        chistiff=self.chistiff,
        chiei_ratio=self.chiei_ratio.get_value(t),
        chi_D_ratio=self.chi_D_ratio.get_value(t),
        VR_D_ratio=self.VR_D_ratio.get_value(t),
        **base_kwargs,
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
    chi_e_bohm_multiplier: Multiplier for the default Bohm term (default 1.0).
    chi_e_gyrobohm_multiplier: Multiplier for the default Gyro-Bohm term (default 1.0).
    chi_i_bohm_multiplier: Multiplier for the default Bohm term for ions (default 1.0).
    chi_i_gyrobohm_multiplier: Multiplier for the default Gyro-Bohm term for ions (default 1.0).
  """

  transport_model: Literal['bohm-gyrobohm'] = 'bohm-gyrobohm'
  chi_e_bohm_coeff: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(8e-5)
  )
  chi_e_gyrobohm_coeff: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5e-6)
  )
  chi_i_bohm_coeff: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(8e-5)
  )
  chi_i_gyrobohm_coeff: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5e-6)
  )
  d_face_c1: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  d_face_c2: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  v_face_coeff: interpolated_param_1d.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(-0.1)
  )
  # New fields: multipliers for the default Bohm and Gyro-Bohm terms.
  chi_e_bohm_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_e_gyrobohm_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_i_bohm_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_i_gyrobohm_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )

  def build_transport_model(
      self,
  ) -> bohm_gyrobohm.BohmGyroBohmTransportModel:
    return bohm_gyrobohm.BohmGyroBohmTransportModel()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> bohm_gyrobohm.DynamicRuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return bohm_gyrobohm.DynamicRuntimeParams(
        chi_e_bohm_coeff=self.chi_e_bohm_coeff.get_value(t),
        chi_e_gyrobohm_coeff=self.chi_e_gyrobohm_coeff.get_value(t),
        chi_i_bohm_coeff=self.chi_i_bohm_coeff.get_value(t),
        chi_i_gyrobohm_coeff=self.chi_i_gyrobohm_coeff.get_value(t),
        d_face_c1=self.d_face_c1.get_value(t),
        d_face_c2=self.d_face_c2.get_value(t),
        v_face_coeff=self.v_face_coeff.get_value(t),
        chi_e_bohm_multiplier=self.chi_e_bohm_multiplier.get_value(t),
        chi_e_gyrobohm_multiplier=self.chi_e_gyrobohm_multiplier.get_value(t),
        chi_i_bohm_multiplier=self.chi_i_bohm_multiplier.get_value(t),
        chi_i_gyrobohm_multiplier=self.chi_i_gyrobohm_multiplier.get_value(t),
        **base_kwargs,
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

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if isinstance(self.transport_model_config, QualikizTransportModel):
      if not _QUALIKIZ_TRANSPORT_MODEL_AVAILABLE:
        raise ValueError(
            'Qualikiz transport model is not available. Possible issue is that'
            ' the QuaLiKiz Pythontools are not installed.'
        )
    return self

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # If we are running with the standard class constructor we don't need to do
    # any custom validation.
    if 'transport_model_config' in data:
      return data

    if 'transport_model' not in data:
      data['transport_model'] = 'constant'

    return {'transport_model_config': data}

  def build_transport_model(self) -> transport_model_lib.TransportModel:
    """Builds a transport model from the config."""
    return self.transport_model_config.build_transport_model()

  def build_dynamic_params(
      self, t: chex.Numeric
  ) -> runtime_params.DynamicRuntimeParams:
    """Builds a dynamic runtime params from the config."""
    return self.transport_model_config.build_dynamic_params(t)
