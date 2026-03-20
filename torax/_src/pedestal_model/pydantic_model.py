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
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.physics import scaling_laws
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class PowerScalingFormation(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Configuration for power scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from an appropriate scaling law.
  The reduction is a multiplicative factor between 1.0 and base_multiplier.

  The formula is
    transport_multiplier = (1.0 - alpha) * 1.0 + alpha * base_multiplier,
  where alpha is a smooth sigmoid function of
    (P_SOL - P_LH * P_LH_prefactor) / (P_LH * P_LH_prefactor)
  with given sharpness and offset, namely:
     sigmoid(x) = 1 / (1 + exp(-sharpness * [x - offset])).


  Attributes:
    sharpness: Scaling factor applied to the argument of the sigmoid function,
      setting the sharpness of the smooth formation window. Decrease for a
      smoother formation, which may be more numerically stable but may lead to
      starting formation at a temperature or density below the target values.
    offset: Bias applied to the argument of the sigmoid function, setting the
      dimensionless offset of the formation window. Increase to start formation
      at a higher P_SOL.
    base_multiplier: The base value of the transport multiplier. Increase for
      stronger decreases in transport once formation starts.
    P_LH_prefactor: Dimensionless multiplier for P_LH. Increase to scale up
      P_LH, and therefore start the L-H transition at a higher P_SOL.
  """

  sharpness: pydantic.PositiveFloat = 100.0
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  base_multiplier: Annotated[
      array_typing.FloatScalar, pydantic.Field(gt=0.0, le=1.0)
  ] = 1e-6
  P_LH_prefactor: pydantic.PositiveFloat = 1.0

  @abc.abstractmethod
  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    """Builds the formation model."""

  @abc.abstractmethod
  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    """Builds the runtime params."""


class MartinScalingFormation(PowerScalingFormation):
  """Configuration for Martin scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from the Martin scaling law. See
  `PowerScalingFormation` for more details.
  """

  model_name: Annotated[
      Literal["martin_scaling"], torax_pydantic.JAX_STATIC
  ] = "martin_scaling"

  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    return power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    del t
    return power_scaling_formation_model.PowerScalingFormationRuntimeParams(
        sharpness=self.sharpness,
        offset=self.offset,
        base_multiplier=self.base_multiplier,
        P_LH_prefactor=self.P_LH_prefactor,
    )


class DelabieScalingFormation(PowerScalingFormation):
  """Configuration for Delabie scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from the Delabie scaling law. See
  `PowerScalingFormation` for more details.
  """

  model_name: Annotated[
      Literal["delabie_scaling"], torax_pydantic.JAX_STATIC
  ] = "delabie_scaling"

  divertor_configuration: Annotated[
      scaling_laws.DivertorConfiguration, torax_pydantic.JAX_STATIC
  ] = scaling_laws.DivertorConfiguration.HT

  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    return power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.DELABIE,
        divertor_configuration=self.divertor_configuration,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    del t
    return power_scaling_formation_model.PowerScalingFormationRuntimeParams(
        sharpness=self.sharpness,
        offset=self.offset,
        base_multiplier=self.base_multiplier,
        P_LH_prefactor=self.P_LH_prefactor,
    )


class ProfileValueSaturation(torax_pydantic.BaseModelFrozen):
  """Configuration for ProfileValueSaturation model.

  This saturation model triggers an increase in pedestal transport when the
  pedestal temperature and density are above the values requested by the
  pedestal model. The increase is a smooth linear function of the ratio of the
  current value to the value requested by the pedestal model.

  The formula is
    transport_multiplier = 1 + alpha * base_multiplier,
  where alpha is a softplus function of the normalized deviation from the target
  value, with given steepness and offset:
    x = (current - target) / target - offset
    alpha = log(1 + exp(steepness * x))


  Attributes:
    steepness: Scaling factor applied to the argument of the softplus function,
      setting the steepness of the smooth saturation function. Decrease for a
      smoother saturation, which may be more numerically stable but may lead to
      starting saturation at a temperature or density below the target values.
    offset: Bias applied to the argument of the softplus function, setting the
      dimensionless offset of the saturation window. Increase to start
      saturation at a higher temperature or density.
    base_multiplier: The base value of the transport multiplier. Increase for
      stronger increases in transport once saturation starts.
  """

  model_name: Annotated[Literal["profile_value"], torax_pydantic.JAX_STATIC] = (
      "profile_value"
  )
  steepness: pydantic.PositiveFloat = 100.0
  # Default offset is > 0 as otherwise saturation starts too early. This is
  # because the softplus function is nonzero before the argument is zero.
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.1
  base_multiplier: Annotated[
      array_typing.FloatScalar, pydantic.Field(gt=1.0)
  ] = 1e6

  def build_saturation_model(
      self,
  ) -> profile_value_saturation_model.ProfileValueSaturationModel:
    return profile_value_saturation_model.ProfileValueSaturationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.SaturationRuntimeParams:
    del t
    return runtime_params.SaturationRuntimeParams(
        steepness=self.steepness,
        offset=self.offset,
        base_multiplier=self.base_multiplier,
    )


# For new formation and saturation models, add to these TypeAliases via Union.
FormationConfig: TypeAlias = DelabieScalingFormation | MartinScalingFormation
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
    formation_model: Configuration for the pedestal formation model.
    saturation_model: Configuration for the pedestal saturation model.
    chi_max: Maximum effective thermal diffusion coefficient from the core
      transport model in the pedestal region (i.e., before applying
      ADAPTIVE_TRANSPORT) [m^2/s].
    D_e_max: Maximum effective particle diffusion coefficient from the core
      transport model in the pedestal region (i.e., before applying
      ADAPTIVE_TRANSPORT) [m^2/s].
    V_e_max: Maximum effective particle pinch velocity from the core transport
      model in the pedestal region (i.e., before applying ADAPTIVE_TRANSPORT)
      [m/s].
    V_e_min: Minimum effective particle pinch velocity from the core transport
      model in the pedestal region (i.e., before applying ADAPTIVE_TRANSPORT)
      [m/s].
  """

  set_pedestal: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(False)
  )
  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.ADAPTIVE_SOURCE
  )
  formation_model: FormationConfig = torax_pydantic.ValidatedDefault(
      MartinScalingFormation()
  )
  saturation_model: SaturationConfig = torax_pydantic.ValidatedDefault(
      ProfileValueSaturation()
  )
  # TODO(b/491895183): Do a sweep across different cases to find good default
  # values for these parameters.
  chi_max: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1.0
  )
  D_e_max: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1.0
  )
  V_e_max: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1.0
  )
  V_e_min: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      -1.0
  )
  pedestal_top_smoothing_width: torax_pydantic.TimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.02)
  )

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if "formation_model" not in configurable_data:
      configurable_data["formation_model"] = {"model_name": "martin_scaling"}
    if "saturation_model" not in configurable_data:
      configurable_data["saturation_model"] = {"model_name": "profile_value"}
    # Set default model names.
    if "model_name" not in configurable_data["formation_model"]:
      configurable_data["formation_model"]["model_name"] = "martin_scaling"
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
        chi_max=self.chi_max.get_value(t),
        D_e_max=self.D_e_max.get_value(t),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
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
        chi_max=self.chi_max.get_value(t),
        D_e_max=self.D_e_max.get_value(t),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
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
        chi_max=self.chi_max.get_value(t),
        D_e_max=self.D_e_max.get_value(t),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
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
        chi_max=self.chi_max.get_value(t),
        D_e_max=self.D_e_max.get_value(t),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
    )


PedestalConfig = SetPpedTpedRatioNped | SetTpedNped | NoPedestal
