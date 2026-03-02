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

import abc
import copy
from typing import Annotated, Any, Literal, TypeAlias
import chex
import pydantic
from torax._src import array_typing
from torax._src.pedestal_model import no_pedestal
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model import set_pped_tpedratio_nped
from torax._src.pedestal_model import set_tped_nped
from torax._src.pedestal_model.formation import martin_formation_model
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


# TODO(b/323504363): Generalise to pedestal formation models based on power
# thresholds (e.g. Metal Wall scaling), not just Martin scaling.
class MartinFormation(torax_pydantic.BaseModelFrozen):
  """Configuration for Martin formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is from the Martin scaling. The reduction is a smooth sigmoid
  function of the ratio P_SOL / (P_LH * P_LH_prefactor).

  The formula is
    rescaled_P_LH = P_LH * P_LH_prefactor
      normalized_deviation = (P_SOL - rescaled_P_LH) / rescaled_P_LH - offset
      transport_multiplier = 1 - sigmoid(normalized_deviation / width)
      transport_multiplier = transport_multiplier**exponent

  The transport multiplier is later clipped to the range
  [min_transport_multiplier, max_transport_multiplier].

  Attributes:
    sigmoid_width: Dimensionless width of the sigmoid function for smoothing the
      formation window. Increase for a smoother L-H transition, but doing so may
      lead to starting the L-H transition at a power below P_LH.
    sigmoid_offset: Dimensionless offset of sigmoid function from P_LH / P_SOL =
      1. Increase to start the L-H transition at a higher P_SOL / P_LH ratio.
    sigmoid_exponent: The exponent of the transport multiplier. Increase for a
      faster reduction in transport once the L-H transition starts.
    P_LH_prefactor: Dimensionless multiplier for P_LH. Increase to scale up
      P_LH, and therefore start the L-H transition at a higher P_SOL.
  """

  model_name: Annotated[Literal["martin"], torax_pydantic.JAX_STATIC] = "martin"
  sigmoid_width: pydantic.PositiveFloat = 1e-3
  sigmoid_offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  sigmoid_exponent: pydantic.PositiveFloat = 3.0
  P_LH_prefactor: pydantic.PositiveFloat = 1.0

  def build_formation_model(
      self,
  ) -> martin_formation_model.MartinFormationModel:
    return martin_formation_model.MartinFormationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> martin_formation_model.MartinFormationRuntimeParams:
    del t
    return martin_formation_model.MartinFormationRuntimeParams(
        sigmoid_width=self.sigmoid_width,
        sigmoid_offset=self.sigmoid_offset,
        sigmoid_exponent=self.sigmoid_exponent,
        P_LH_prefactor=self.P_LH_prefactor,
    )


class ProfileValueSaturation(torax_pydantic.BaseModelFrozen):
  """Configuration for ProfileValueSaturation model.

  This saturation model triggers an increase in pedestal transport when the
  pedestal temperature and density are above the values requested by the
  pedestal model. The increase is a smooth sigmoid function of the ratio of the
  current value to the value requested by the pedestal model.

  The formula is
    normalized_deviation = (current - target) / target - offset
    transport_multiplier = 1 / (1 - sigmoid(normalized_deviation / width))
    transport_multiplier = transport_multiplier**exponent

  The transport multiplier is then clipped to the range
  [min_transport_multiplier, max_transport_multiplier].

  Attributes:
    sigmoid_width: Dimensionless width of the sigmoid function for smoothing the
      saturation window. Increase for a smoother saturation, but doing so may
      lead to starting saturation at a temperature or density below the target
      values.
    sigmoid_offset: Dimensionless offset of the saturation window. Increase to
      start saturation at a higher temperature or density.
    sigmoid_exponent: The exponent of the transport multiplier. Increase for a
      faster increase in transport once saturation starts.
  """

  model_name: Annotated[Literal["profile_value"], torax_pydantic.JAX_STATIC] = (
      "profile_value"
  )
  sigmoid_width: pydantic.PositiveFloat = 0.1
  sigmoid_offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  sigmoid_exponent: pydantic.PositiveFloat = 1.0

  def build_saturation_model(
      self,
  ) -> profile_value_saturation_model.ProfileValueSaturationModel:
    return profile_value_saturation_model.ProfileValueSaturationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.SaturationRuntimeParams:
    del t
    return runtime_params.SaturationRuntimeParams(
        sigmoid_width=self.sigmoid_width,
        sigmoid_offset=self.sigmoid_offset,
        sigmoid_exponent=self.sigmoid_exponent,
    )


# For new formation and saturation models, add to these TypeAliases via Union.
FormationConfig: TypeAlias = MartinFormation
SaturationConfig: TypeAlias = ProfileValueSaturation


class BasePedestal(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for pedestal models.

  Attributes:
    set_pedestal: Whether to use the pedestal model and set the pedestal. Can be
      time varying.
    mode: Defines how the pedestal is generated. Set to ADAPTIVE_TRANSPORT to
      set the pedestal by modifying the transport coefficients in the pedestal
      region, allowing the pedestal to self-consistently evolve. Set to
      ADAPTIVE_SOURCE to set the pedestal by adding a source/sink term at the
      pedestal top, forcing the pedestal top values to be as prescribed.
  """

  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.ADAPTIVE_SOURCE
  )
  formation_model: FormationConfig = torax_pydantic.ValidatedDefault(
      MartinFormation()
  )
  saturation_model: SaturationConfig = torax_pydantic.ValidatedDefault(
      ProfileValueSaturation()
  )
  max_transport_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(10.0)
  )
  min_transport_multiplier: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.1)
  )

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if "formation_model" not in configurable_data:
      configurable_data["formation_model"] = {"model_name": "martin"}
    if "saturation_model" not in configurable_data:
      configurable_data["saturation_model"] = {"model_name": "profile_value"}
    # Set default model names.
    if "model_name" not in configurable_data["formation_model"]:
      configurable_data["formation_model"]["model_name"] = "martin"
    if "model_name" not in configurable_data["saturation_model"]:
      configurable_data["saturation_model"]["model_name"] = "profile_value"

    return configurable_data

  @abc.abstractmethod
  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the pedestal model."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params."""
    return runtime_params.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        mode=self.mode,
        formation=self.formation_model.build_runtime_params(t),
        saturation=self.saturation_model.build_runtime_params(t),
        max_transport_multiplier=self.max_transport_multiplier.get_value(t),
        min_transport_multiplier=self.min_transport_multiplier.get_value(t),
    )


class SetPpedTpedRatioNped(BasePedestal):
  """Model for direct specification of pressure, temperature ratio, and density.

  Attributes:
    P_ped: The plasma pressure at the pedestal [Pa].
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal
      [dimensionless].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal["set_P_ped_n_ped"], torax_pydantic.JAX_STATIC
  ] = "set_P_ped_n_ped"
  P_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(1e5)
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_T_e_ratio: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> (
      set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel
  ):
    return set_pped_tpedratio_nped.SetPressureTemperatureRatioAndDensityPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_pped_tpedratio_nped.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return set_pped_tpedratio_nped.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        P_ped=self.P_ped.get_value(t),
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        max_transport_multiplier=base_runtime_params.max_transport_multiplier,
        min_transport_multiplier=base_runtime_params.min_transport_multiplier,
    )


class SetTpedNped(BasePedestal):
  """A basic version of the pedestal model that uses direct specification.

  Attributes:
    n_e_ped: The electron density at the pedestal [m^-3] or fGW.
    n_e_ped_is_fGW: Whether the electron density at the pedestal is in units of
      fGW.
    T_i_ped: Ion temperature at the pedestal [keV].
    T_e_ped: Electron temperature at the pedestal [keV].
    rho_norm_ped_top: The location of the pedestal top.
  """

  model_name: Annotated[
      Literal["set_T_ped_n_ped"], torax_pydantic.JAX_STATIC
  ] = "set_T_ped_n_ped"
  n_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      0.7e20
  )
  n_e_ped_is_fGW: bool = False
  T_i_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  T_e_ped: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      5.0
  )
  rho_norm_ped_top: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_pedestal_model(
      self,
  ) -> set_tped_nped.SetTemperatureDensityPedestalModel:
    return set_tped_nped.SetTemperatureDensityPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_tped_nped.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return set_tped_nped.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        n_e_ped=self.n_e_ped.get_value(t),
        n_e_ped_is_fGW=self.n_e_ped_is_fGW,
        T_i_ped=self.T_i_ped.get_value(t),
        T_e_ped=self.T_e_ped.get_value(t),
        rho_norm_ped_top=self.rho_norm_ped_top.get_value(t),
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        max_transport_multiplier=base_runtime_params.max_transport_multiplier,
        min_transport_multiplier=base_runtime_params.min_transport_multiplier,
    )


class NoPedestal(BasePedestal):
  """A pedestal model for when there is no pedestal.

  Note that setting `set_pedestal` to True with a NoPedestal model is the
  equivalent of setting it to False.
  """

  model_name: Annotated[Literal["no_pedestal"], torax_pydantic.JAX_STATIC] = (
      "no_pedestal"
  )

  def build_pedestal_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    base_runtime_params = super().build_runtime_params(t)
    return runtime_params.RuntimeParams(
        set_pedestal=base_runtime_params.set_pedestal,
        mode=base_runtime_params.mode,
        formation=base_runtime_params.formation,
        saturation=base_runtime_params.saturation,
        max_transport_multiplier=base_runtime_params.max_transport_multiplier,
        min_transport_multiplier=base_runtime_params.min_transport_multiplier,
    )


PedestalConfig = SetPpedTpedRatioNped | SetTpedNped | NoPedestal
