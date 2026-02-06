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
from typing import Annotated, Any, Literal, Sequence

from absl import logging
import chex
from fusion_surrogates.qlknn.models import registry
import numpy as np
import pydantic
from torax._src.torax_pydantic import interpolated_param_1d
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import bohm_gyrobohm
from torax._src.transport_model import combined
from torax._src.transport_model import constant
from torax._src.transport_model import critical_gradient
from torax._src.transport_model import enums
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import qlknn_10d
from torax._src.transport_model import qlknn_transport_model
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
import typing_extensions


def _resolve_qlknn_model_name(model_name: str, model_path: str) -> str:
  """Resolve the model name."""
  if model_name:
    if model_name == qlknn_10d.QLKNN10D_NAME:
      if not model_path:
        raise ValueError('QLKNN10D requires a model path to be provided.')
      if model_path.endswith('.qlknn'):
        raise ValueError(
            f'Model path "{model_path}" is not a valid path for a'
            f' {qlknn_10d.QLKNN10D_NAME} model.',
        )
    return model_name

  if not model_path:
    model_name = registry.DEFAULT_MODEL_NAME
  elif not model_path.endswith('.qlknn'):
    logging.info(
        'QLKNN model path "%s"does not end with ".qlknn", we assume this is'
        ' pointing to a qlknn-hyper (QLKNN10D) directory.',
        model_path,
    )
    model_name = qlknn_10d.QLKNN10D_NAME
  else:
    # We cannot resolve the model name. We are likely using a custom model.
    model_name = ''
  return model_name


# pylint: disable=invalid-name
class QLKNNTransportModel(pydantic_model_base.TransportBase):
  """Model for the QLKNN transport model.

  To determine which model to load, TORAX uses the following logic:

  * If `model_path` is provided, then we load the model from this path.
  * Otherwise, if `model_name` is provided, we load that model from registered
    models in the `fusion_surrogates` library.
  * If `model_name` is not set either, we load the default QLKNN model from
    `fusion_surrogates` (currently `QLKNN_7_11`).

  It is recommended to not set `model_name` or `model_path` to use the default
  QLKNN model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'qlknn'.
    model_path: Path to the model. Takes precedence over `model_name`.
    qlknn_model_name: Name of the model to use. Used to select a model from the
      `fusion_surrogates` library.
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
    collisionality_multiplier: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DV_effective: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
    rotation_multiplier: Multiplier for rotation.
    rotation_mode: Mode for rotation, either HALF_RADIUS, FULL_RADIUS or OFF.
    shear_suppression_model: Shear suppression model for rotation effects.
      Either WALTZ_RULE [Waltz et al., PoP
      1998](https://doi.org/10.1063/1.872847) or VANDEPLASSCHE2020 [Van de
      Plassche et al. PoP 2020](https://doi.org/10.1063/1.5134126).
    shear_suppression_alpha: Alpha parameter for Waltz rule. Larger values
      increase the suppression effect. Only used when shear_suppression_model =
      WALTZ_RULE.
  """

  model_name: Annotated[Literal['qlknn'], torax_pydantic.JAX_STATIC] = 'qlknn'
  model_path: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  qlknn_model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  include_ITG: bool = True
  include_TEM: bool = True
  include_ETG: bool = True
  ITG_flux_ratio_correction: float = 1.0
  ETG_correction_factor: float = 1.0 / 3.0
  clip_inputs: bool = False
  clip_margin: float = 0.95
  collisionality_multiplier: float = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05
  rotation_multiplier: pydantic.NonNegativeFloat = 1.0
  rotation_mode: Annotated[
      qualikiz_based_transport_model.RotationMode, torax_pydantic.JAX_STATIC
  ] = qualikiz_based_transport_model.RotationMode.OFF
  shear_suppression_model: Annotated[
      qlknn_transport_model.ShearSuppressionModel, torax_pydantic.JAX_STATIC
  ] = qlknn_transport_model.ShearSuppressionModel.WALTZ_RULE
  shear_suppression_alpha: pydantic.NonNegativeFloat = 1.0
  output_mode_contributions: Annotated[bool, torax_pydantic.JAX_STATIC] = False

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: Any) -> Any:
    # If the input is not a dict (e.g. it is already a model instance), return
    # it as is.
    if not isinstance(data, dict):
      return data
    data = copy.deepcopy(data)

    data['qlknn_model_name'] = _resolve_qlknn_model_name(
        model_name=data.get('qlknn_model_name', ''),
        model_path=data.get('model_path', ''),
    )

    if data['qlknn_model_name'] == qlknn_10d.QLKNN10D_NAME:
      if 'collisionality_multiplier' not in data:
        # Correction factor to a more recent QLK collision operator.
        data['collisionality_multiplier'] = 0.25
      if 'ITG_flux_ratio_correction' not in data:
        # The QLK version this specific QLKNN was trained on tends to
        # underpredict ITG electron heat flux in shaped, high-beta scenarios.
        data['ITG_flux_ratio_correction'] = 2.0
    else:
      if 'smoothing_width' not in data:
        data['smoothing_width'] = 0.1
    return data

  def build_transport_model(self) -> qlknn_transport_model.QLKNNTransportModel:
    return qlknn_transport_model.QLKNNTransportModel(
        path=self.model_path, name=self.qlknn_model_name
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> qlknn_transport_model.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return qlknn_transport_model.RuntimeParams(
        include_ITG=self.include_ITG,
        include_TEM=self.include_TEM,
        include_ETG=self.include_ETG,
        ITG_flux_ratio_correction=self.ITG_flux_ratio_correction,
        ETG_correction_factor=self.ETG_correction_factor,
        clip_inputs=self.clip_inputs,
        clip_margin=self.clip_margin,
        collisionality_multiplier=self.collisionality_multiplier,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        rotation_multiplier=self.rotation_multiplier,
        rotation_mode=self.rotation_mode,
        shear_suppression_model=self.shear_suppression_model,
        shear_suppression_alpha=self.shear_suppression_alpha,
        output_mode_contributions=self.output_mode_contributions,
        **base_kwargs,
    )


class TGLFNNukaeaTransportModel(pydantic_model_base.TransportBase):
  """Model for the TGLFNN-ukaea transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'tglfnn-ukaea'.
    machine: The machine type to use. Either 'step' or 'multimachine'.
  """

  model_name: Annotated[Literal['tglfnn-ukaea'], torax_pydantic.JAX_STATIC] = (
      'tglfnn-ukaea'
  )
  machine: Annotated[
      Literal['step', 'multimachine'], torax_pydantic.JAX_STATIC
  ] = 'multimachine'
  rotation_multiplier: pydantic.NonNegativeFloat = 1.0
  use_rotation: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # Quasilinear transport options
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  def build_transport_model(
      self,
  ) -> tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel:
    return tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine=self.machine,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> tglfnn_ukaea_transport_model.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return tglfnn_ukaea_transport_model.RuntimeParams(
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        rotation_multiplier=self.rotation_multiplier,
        use_rotation=self.use_rotation,
        # From base
        **base_kwargs,
    )


class ConstantTransportModel(pydantic_model_base.TransportBase):
  """Model for the Constant transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'constant'.
    chi_i: coefficient in ion heat equation diffusion term in m^2/s.
    chi_e: coefficient in electron heat equation diffusion term in m^2/s.
    D_e: diffusion coefficient in electron density equation in m^2/s.
    V_e: convection coefficient in electron density equation in m^2/s.
  """

  model_name: Annotated[Literal['constant'], torax_pydantic.JAX_STATIC] = (
      'constant'
  )
  chi_i: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  chi_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  D_e: torax_pydantic.PositiveTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  V_e: torax_pydantic.TimeVaryingArray = torax_pydantic.ValidatedDefault(-0.33)

  def build_transport_model(self) -> constant.ConstantTransportModel:
    return constant.ConstantTransportModel()

  def build_runtime_params(self, t: chex.Numeric) -> constant.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return constant.RuntimeParams(
        chi_i=self.chi_i.get_value(t, 'face'),
        chi_e=self.chi_e.get_value(t, 'face'),
        D_e=self.D_e.get_value(t, 'face'),
        V_e=self.V_e.get_value(t, 'face'),
        **base_kwargs,
    )


class CriticalGradientTransportModel(pydantic_model_base.TransportBase):
  """Model for the Critical Gradient transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'CGM'.
    alpha: Exponent of chi power law: chi ∝ (R/LTi - R/LTi_crit)^alpha.
    chi_stiff: Stiffness parameter.
    chi_e_i_ratio: Ratio of electron to ion heat transport coefficient (ion
      higher for ITG).
    chi_D_ratio: Ratio of electron particle to ion heat transport coefficient.
    VR_D_ratio: Ratio of major radius * electron particle convection, to
      electron diffusion. Sets the value of electron particle convection in the
      model.
  """

  model_name: Annotated[Literal['CGM'], torax_pydantic.JAX_STATIC] = 'CGM'
  alpha: float = 2.0
  chi_stiff: float = 2.0
  chi_e_i_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(2.0)
  )
  chi_D_ratio: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  VR_D_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )

  def build_transport_model(
      self,
  ) -> critical_gradient.CriticalGradientTransportModel:
    return critical_gradient.CriticalGradientTransportModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> critical_gradient.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return critical_gradient.RuntimeParams(
        alpha=self.alpha,
        chi_stiff=self.chi_stiff,
        chi_e_i_ratio=self.chi_e_i_ratio.get_value(t),
        chi_D_ratio=self.chi_D_ratio.get_value(t),
        VR_D_ratio=self.VR_D_ratio.get_value(t),
        **base_kwargs,
    )


class BohmGyroBohmTransportModel(pydantic_model_base.TransportBase):
  """Model for the Bohm + Gyro-Bohm transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'bohm-gyrobohm'.
    chi_e_bohm_coeff: Prefactor for Bohm term for electron heat conductivity.
    chi_e_gyrobohm_coeff: Prefactor for GyroBohm term for electron heat
      conductivity.
    chi_i_bohm_coeff: Prefactor for Bohm term for ion heat conductivity.
    chi_i_gyrobohm_coeff: Prefactor for GyroBohm term for ion heat conductivity.
    chi_e_bohm_multiplier: Multiplier for chi_e_bohm_coeff. Intended for
      user-friendly default modification.
    chi_e_gyrobohm_multiplier: Multiplier for chi_e_gyrobohm_coeff. Intended for
      user-friendly default modification.
    chi_i_bohm_multiplier: Multiplier for chi_i_bohm_coeff. Intended for
      user-friendly default modification.
    chi_i_gyrobohm_multiplier: Multiplier for chi_i_gyrobohm_coeff. Intended for
      user-friendly default modification.
    D_face_c1: Constant for the electron diffusivity weighting factor.
    D_face_c2: Constant for the electron diffusivity weighting factor.
    V_face_coeff: Proportionality factor between convectivity and diffusivity.
  """

  model_name: Annotated[Literal['bohm-gyrobohm'], torax_pydantic.JAX_STATIC] = (
      'bohm-gyrobohm'
  )
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
  D_face_c1: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  D_face_c2: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.3)
  )
  V_face_coeff: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(-0.1)
  )

  def build_transport_model(
      self,
  ) -> bohm_gyrobohm.BohmGyroBohmTransportModel:
    return bohm_gyrobohm.BohmGyroBohmTransportModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> bohm_gyrobohm.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return bohm_gyrobohm.RuntimeParams(
        chi_e_bohm_coeff=self.chi_e_bohm_coeff.get_value(t),
        chi_e_gyrobohm_coeff=self.chi_e_gyrobohm_coeff.get_value(t),
        chi_i_bohm_coeff=self.chi_i_bohm_coeff.get_value(t),
        chi_i_gyrobohm_coeff=self.chi_i_gyrobohm_coeff.get_value(t),
        chi_e_bohm_multiplier=self.chi_e_bohm_multiplier.get_value(t),
        chi_e_gyrobohm_multiplier=self.chi_e_gyrobohm_multiplier.get_value(t),
        chi_i_bohm_multiplier=self.chi_i_bohm_multiplier.get_value(t),
        chi_i_gyrobohm_multiplier=self.chi_i_gyrobohm_multiplier.get_value(t),
        D_face_c1=self.D_face_c1.get_value(t),
        D_face_c2=self.D_face_c2.get_value(t),
        V_face_coeff=self.V_face_coeff.get_value(t),
        **base_kwargs,
    )


try:
  from torax._src.transport_model import qualikiz_transport_model  # pylint: disable=g-import-not-at-top

  # Since CombinedCompatibleTransportModel is not constant, because of the
  # try/except block, unions using this type will cause invalid-annotation
  # errors in pytype.
  CombinedCompatibleTransportModel = (
      QLKNNTransportModel
      | TGLFNNukaeaTransportModel
      | ConstantTransportModel
      | CriticalGradientTransportModel
      | BohmGyroBohmTransportModel
      | qualikiz_transport_model.QualikizTransportModelConfig
  )

except ImportError:
  CombinedCompatibleTransportModel = (
      QLKNNTransportModel
      | TGLFNNukaeaTransportModel
      | ConstantTransportModel
      | CriticalGradientTransportModel
      | BohmGyroBohmTransportModel
  )


class CombinedTransportModel(pydantic_model_base.TransportBase):
  """Model for the Combined transport model.

  Attributes:
    model_name: The transport model to use. Hardcoded to 'combined'.
    transport_models: A sequence of transport models, whose outputs will be
      summed to give the combined core transport coefficients.
    pedestal_transport_models: A sequence of models that will be combined for
      pedestal transport coefficients.
  """

  # TODO(b/434175938) V2: rename `transport_models` to `core_transport_models`
  transport_models: Sequence[CombinedCompatibleTransportModel] = pydantic.Field(
      default_factory=list
  )  # pytype: disable=invalid-annotation
  pedestal_transport_models: Sequence[
      CombinedCompatibleTransportModel
  ] = pydantic.Field(
      default_factory=list
  )  # pytype: disable=invalid-annotation
  model_name: Annotated[Literal['combined'], torax_pydantic.JAX_STATIC] = (
      'combined'
  )

  def build_transport_model(self) -> combined.CombinedTransportModel:
    transport_models = tuple(
        model.build_transport_model() for model in self.transport_models
    )
    pedestal_transport_models = tuple(
        model.build_transport_model()
        for model in self.pedestal_transport_models
    )

    return combined.CombinedTransportModel(
        transport_models=transport_models,
        pedestal_transport_models=pedestal_transport_models,
    )

  def build_runtime_params(self, t: chex.Numeric) -> combined.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    transport_model_params = [
        model.build_runtime_params(t) for model in self.transport_models
    ]
    pedestal_transport_model_params = [
        model.build_runtime_params(t)
        for model in self.pedestal_transport_models
    ]

    return combined.RuntimeParams(
        transport_model_params=transport_model_params,
        pedestal_transport_model_params=pedestal_transport_model_params,
        **base_kwargs,
    )

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    super()._check_fields()
    if (
        any([
            np.any(model.apply_inner_patch.value)
            or np.any(model.apply_outer_patch.value)
            for model in self.transport_models + self.pedestal_transport_models
        ])
        or np.any(self.apply_inner_patch.value)
        or np.any(self.apply_outer_patch.value)
    ):
      raise ValueError(
          'apply_inner_patch and apply_outer_patch not supported for'
          ' CombinedTransportModel or its component models.'
      )
    if np.any(self.rho_min.value != 0.0) or np.any(self.rho_max.value != 1.0):
      raise ValueError(
          'rho_min and rho_max should not be set for CombinedTransportModel, as'
          ' it should be applied across the whole rho domain.'
      )
    if any([
        np.any(model.rho_min.value != 0.0) or np.any(model.rho_max.value != 1.0)
        for model in self.pedestal_transport_models
    ]):
      raise ValueError(
          'rho_min and rho_max not supported for pedestal_transport_models, as '
          'their region of validity is set by the pedestal model.'
      )

    return self

  @pydantic.model_validator(mode='after')
  def _check_unique_overwrites_core(self) -> typing_extensions.Self:
    _validate_unique_overwrites(self.transport_models, 'core')
    return self

  @pydantic.model_validator(mode='after')
  def _check_unique_overwrites_pedestal(self) -> typing_extensions.Self:
    _validate_unique_overwrites(self.pedestal_transport_models, 'pedestal')
    return self


def _validate_unique_overwrites(
    models: Sequence[CombinedCompatibleTransportModel], domain: str
) -> None:  # pytype: disable=invalid-annotation
  """Validates that models overwriting the same channel do not overlap."""

  for i, model1 in enumerate(models):
    # Only check models that overwrite
    if model1.merge_mode != enums.MergeMode.OVERWRITE:
      continue

    # Avoid self-comparison and duplicated checks
    for model2 in models[i+1:]:
      # Only check models that overwrite
      if model2.merge_mode != enums.MergeMode.OVERWRITE:
        continue

      # Only check if there are shared enabled channels
      overlapping_channels = _get_overlapping_channels(model1, model2)
      if not overlapping_channels:
        continue

      # Check for spatial overlap
      if _ranges_overlap(model1, model2):
        raise ValueError(
            f'Multiple {domain} transport models are configured to'
            f' OVERWRITE the same channels {overlapping_channels} in'
            f' overlapping radial zones: '
            f"'{model1.model_name}' and '{model2.model_name}'"
        )


def _get_overlapping_channels(
    model1: CombinedCompatibleTransportModel,  # pytype: disable=invalid-annotation
    model2: CombinedCompatibleTransportModel,  # pytype: disable=invalid-annotation
) -> list[str]:
  """Returns list of channels enabled in both models."""
  channels = []
  if _is_enabled(model1.disable_chi_i) and _is_enabled(model2.disable_chi_i):
    channels.append('chi_i')
  if _is_enabled(model1.disable_chi_e) and _is_enabled(model2.disable_chi_e):
    channels.append('chi_e')
  if _is_enabled(model1.disable_D_e) and _is_enabled(model2.disable_D_e):
    channels.append('D_e')
  if _is_enabled(model1.disable_V_e) and _is_enabled(model2.disable_V_e):
    channels.append('V_e')
  return channels


def _is_enabled(
    disabled_time_series: interpolated_param_1d.TimeVaryingScalar,
) -> bool:
  # If not explicitly disabled everywhere (all True), it is enabled.
  return not np.all(disabled_time_series.value)


def _get_range_bounds(
    model: CombinedCompatibleTransportModel,  # pytype: disable=invalid-annotation
) -> tuple[float, float]:
  """Returns global min/max for a model's active range."""
  return np.min(model.rho_min.value), np.max(model.rho_max.value)


def _ranges_overlap(
    model1: CombinedCompatibleTransportModel,  # pytype: disable=invalid-annotation
    model2: CombinedCompatibleTransportModel,  # pytype: disable=invalid-annotation
) -> bool:
  """Checks if global extents of two models overlap."""
  r1_min, r1_max = _get_range_bounds(model1)
  r2_min, r2_max = _get_range_bounds(model2)

  # Overlap condition: start1 < end2 AND start2 < end1
  return (r1_min < r2_max) and (r2_min < r1_max)


TransportConfig = CombinedTransportModel | CombinedCompatibleTransportModel  # pytype: disable=invalid-annotation
