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
import dataclasses
import math
from typing import Annotated, Any, Literal, Self, TypeAlias
import chex
import pydantic
from torax._src import array_typing
from torax._src.pedestal_model import pedestal_model
from torax._src.pedestal_model import runtime_params
from torax._src.pedestal_model.formation import base as formation_base
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.pedestal_model.formation import prescribed_formation_model
from torax._src.pedestal_model.saturation import profile_value_saturation_model
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import no_pedestal
from torax._src.pedestal_model.structure import set_ptop_ttopratio_ntop
from torax._src.pedestal_model.structure import set_ttop_ntop
from torax._src.physics import scaling_laws
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class FormationConfigBase(torax_pydantic.ConfigBase):
  """Base configuration for pedestal formation models.

  Attributes:
    base_multiplier: The base value of the transport decrease multiplier used in
      `'adaptive_transport'` mode when the pedestal is active. Decrease (closer
      to 0) for stronger reductions in transport once formation starts.
    initial_confinement_mode: Optional explicit initial confinement mode
      (`'l_mode'` or `'h_mode'`) at `t_initial`. When `None` (default), the
      initial mode is inferred automatically from the formation model's
      transition conditions at `t_initial` (cannot be `'h_mode'` when
      `structure_model` is `'no_pedestal'`). Used to enable triggering L-H or
      H-L transitions from a cold start.
  """

  base_multiplier: Annotated[
      array_typing.FloatScalar, pydantic.Field(gt=0.0, le=1.0)
  ] = 1e-6
  initial_confinement_mode: Annotated[
      Literal["l_mode", "h_mode"] | None, torax_pydantic.JAX_STATIC
  ] = None

  @abc.abstractmethod
  def build_formation_model(
      self,
  ) -> formation_base.FormationModel:
    """Builds the formation model."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.FormationRuntimeParams:
    """Builds the runtime params."""
    del t
    mode = self.initial_confinement_mode
    return runtime_params.FormationRuntimeParams(
        base_multiplier=self.base_multiplier,
        initial_confinement_mode=(
            runtime_params.ConfinementMode[mode.upper()]
            if mode is not None
            else None
        ),
    )


class PowerScalingFormationConfig(FormationConfigBase):
  """Configuration for power scaling formation model.

  This formation model triggers a reduction in pedestal transport when P_SOL >
  P_LH, where P_LH is calculated from an appropriate scaling law.
  The reduction is a multiplicative factor between 1.0 and base_multiplier.

  The formula is:
  `transport_multiplier = (1.0 - alpha) * 1.0 + alpha * base_multiplier`,
  where `alpha` is a smooth sigmoid function of
  `(P_SOL - P_LH * P_LH_prefactor) / (P_LH * P_LH_prefactor)`
  with given sharpness and offset, namely:
  `sigmoid(x) = 1 / (1 + exp(-sharpness * [x - offset]))`.


  Attributes:
    sharpness: Scaling factor applied to the argument of the sigmoid function,
      setting the sharpness of the smooth formation window. Decrease for a
      smoother formation, which may be more numerically stable but may lead to
      starting formation at a temperature or density below the target values.
    offset: Bias applied to the argument of the sigmoid function, setting the
      dimensionless offset of the formation window. Increase to start formation
      at a higher P_SOL.
    P_LH_prefactor: Dimensionless multiplier for P_LH. Increase to scale up
      P_LH, and therefore start the L-H transition at a higher P_SOL.
    P_LH_hysteresis_factor: Hysteresis factor for H-L back transitions. When
      checking for an H-L transition, the L-H threshold power P_LH is multiplied
      by this factor, i.e. the back transition occurs when P_SOL < P_LH *
      P_LH_hysteresis_factor. A value less than 1 means that the plasma must
      lose more power to transition back to L-mode than was required to enter
      H-mode, which is the experimentally observed behavior. Must be in [0, 1].
    include_dW_dt_in_P_SOL: Whether to include the dW/dt term in the P_SOL
      calculation used for comparing against P_LH. When `False` (default), uses
      P_heat (total auxiliary + Ohmic power - sinks) instead of P_SOL = P_heat -
      dW/dt. Excluding dW/dt avoids unphysical dithering during transients.
  """

  sharpness: pydantic.PositiveFloat = 100.0
  offset: Annotated[
      array_typing.FloatScalar, pydantic.Field(ge=-10.0, le=10.0)
  ] = 0.0
  P_LH_prefactor: pydantic.PositiveFloat = 1.0
  P_LH_hysteresis_factor: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.8)
  )
  include_dW_dt_in_P_SOL: Annotated[bool, torax_pydantic.JAX_STATIC] = False

  @abc.abstractmethod
  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    """Builds the formation model."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> power_scaling_formation_model.PowerScalingFormationRuntimeParams:
    return power_scaling_formation_model.PowerScalingFormationRuntimeParams(
        **vars(super().build_runtime_params(t)),
        sharpness=self.sharpness,
        offset=self.offset,
        P_LH_prefactor=self.P_LH_prefactor,
        P_LH_hysteresis_factor=self.P_LH_hysteresis_factor.get_value(t),
        include_dW_dt_in_P_SOL=self.include_dW_dt_in_P_SOL,
    )


class MartinScalingFormationConfig(PowerScalingFormationConfig):
  """Triggers L-H and H-L transitions using the Martin $P_LH$ scaling law.

  Triggers L-H formation when $P_SOL > P_LH$ (scaled by `P_LH_prefactor`), with
  H-L back transitions governed by `P_LH_hysteresis_factor`. In
  `'adaptive_transport'` mode, scales pedestal transport smoothly between `1.0`
  and `base_multiplier` via a sigmoid of `(P_SOL - P_LH * P_LH_prefactor) /
  (P_LH * P_LH_prefactor)`.

  Attributes:
    model_name: Formation model name, set to `'martin_scaling'`.
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


class DelabieScalingFormationConfig(PowerScalingFormationConfig):
  """Triggers L-H and H-L transitions using the Delabie $P_LH$ scaling law.

  Triggers L-H formation when $P_SOL > P_LH$ (scaled by `P_LH_prefactor`), with
  H-L back transitions governed by `P_LH_hysteresis_factor`. In
  `'adaptive_transport'` mode, scales pedestal transport smoothly between `1.0`
  and `base_multiplier` via a sigmoid of `(P_SOL - P_LH * P_LH_prefactor) /
  (P_LH * P_LH_prefactor)`.

  Attributes:
    model_name: Formation model name, set to `'delabie_scaling'`.
    divertor_configuration: Divertor configuration (`'horizontal_target'` for
      horizontal target or `'vertical_target'` for vertical target).
  """

  model_name: Annotated[
      Literal["delabie_scaling"], torax_pydantic.JAX_STATIC
  ] = "delabie_scaling"

  divertor_configuration: Annotated[
      scaling_laws.DivertorConfiguration, torax_pydantic.JAX_STATIC
  ] = scaling_laws.DivertorConfiguration.HORIZONTAL_TARGET

  def build_formation_model(
      self,
  ) -> power_scaling_formation_model.PowerScalingFormationModel:
    return power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.DELABIE,
        divertor_configuration=self.divertor_configuration,
    )


class ProfileValueSaturationConfig(torax_pydantic.ConfigBase):
  """Increases pedestal transport when pedestal profiles exceed target values.

  Uses the current profile value at the pedestal top for saturation, smoothly
  increasing pedestal transport as a function of the normalized deviation from
  the target value requested by the structure model:

  ```
  transport_multiplier = 1 + alpha * base_multiplier
  x = (current - target) / target - offset
  alpha = log(1 + exp(steepness * x))
  ```

  Attributes:
    model_name: Saturation model name, set to `'profile_value'`.
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


class PrescribedFormationConfig(FormationConfigBase):
  """Controls pedestal activation according to a prescribed boolean schedule.

  In `'internal_boundary_condition'` mode, activates the pedestal internal
  boundary condition when `pedestal_active` is `True` (with optional ramp over
  `transition_time_width`). In `'adaptive_transport'` mode, scales pedestal
  transport by `base_multiplier` when `pedestal_active` is `True` and `1.0` when
  `False`.

  Attributes:
    model_name: Selects the pedestal formation model (`'prescribed'`,
      `'martin_scaling'`, or `'delabie_scaling'`).
    pedestal_active: Whether the pedestal is active (default interpolation is
      `'step'`; evaluated at step start $t$).
  """

  model_name: Annotated[Literal["prescribed"], torax_pydantic.JAX_STATIC] = (
      "prescribed"
  )
  pedestal_active: torax_pydantic.TimeVaryingScalarStep = (
      torax_pydantic.ValidatedDefault(True)
  )

  def build_formation_model(
      self,
  ) -> prescribed_formation_model.PrescribedFormationModel:
    return prescribed_formation_model.PrescribedFormationModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> prescribed_formation_model.PrescribedFormationRuntimeParams:
    return prescribed_formation_model.PrescribedFormationRuntimeParams(
        **vars(super().build_runtime_params(t)),
        pedestal_active=self.pedestal_active.get_value(t),
    )


# For new formation and saturation models, add to these TypeAliases via Union.
FormationConfig: TypeAlias = Annotated[
    DelabieScalingFormationConfig
    | MartinScalingFormationConfig
    | PrescribedFormationConfig,
    pydantic.Discriminator("model_name"),
]
SaturationConfig: TypeAlias = ProfileValueSaturationConfig


class StructureConfigBase(torax_pydantic.ConfigBase):
  """Base configuration for pedestal structure models.

  Attributes:
    pedestal_profile_form: Controls the shape of internal boundary conditions
      in the pedestal region. Options:

      - `'set_at_top'`: Pedestal values are pinned at a single grid
        cell nearest to `rho_norm_top`.
      - `'mtanh'`: A smooth modified-tanh (`mtanh`) profile is applied across
        the pedestal region from the pedestal top to the separatrix, following
        the `mtanh` parameterization of |snyder2004|. For simplicity, the
        core polynomial part of the full `mtanh` is neglected; only the
        `tanh` component is used in the pedestal region. The `mtanh` width
        $Delta$ is derived on-the-fly from `rho_norm_top` via the
        $psi_N(rho)$ mapping in core profiles:

        $$
        f(psi) = f_sep + a_0 * (tanh(1) -
        tanh(2 * (psi - psi_mid) / Delta))
        $$

        where $psi_top = psi_N(rho_top)$, $Delta =
        (1 - psi_top) / 1.5$, $psi_mid = 1 - Delta / 2$,
        and $a_0 = (f_top - f_sep) / (tanh(1) + tanh(2))$.
  """  # fmt: skip

  pedestal_profile_form: Annotated[
      runtime_params.PedestalProfileForm, torax_pydantic.JAX_STATIC
  ] = runtime_params.PedestalProfileForm.SET_AT_TOP

  @abc.abstractmethod
  def build_structure_model(self) -> structure_base.PedestalStructureModel:
    """Builds the pedestal structure model."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.StructureRuntimeParams:
    """Builds the runtime parameters for the pedestal structure model."""
    del t
    return runtime_params.StructureRuntimeParams(
        pedestal_profile_form=self.pedestal_profile_form,
    )


class SetPtopTtopRatioNtopConfig(StructureConfigBase):
  """Directly specifies pedestal-top pressure, temperature ratio, and density.

  Attributes:
    model_name: Pedestal structure model name, set to `'set_p_top_n_top'`.
    p_top: The plasma pressure at the pedestal top [Pa].
    p_top_multiplier: Multiplier for the pedestal-top pressure (mostly used for
      sensitivity analysis) [dimensionless].
    n_e_top: Electron density at the pedestal top [m^-3 if
      `n_e_top_is_f_GW=False`, or Greenwald fraction if `n_e_top_is_f_GW=True`].
    n_e_top_is_f_GW: Whether the electron density at the pedestal top is
      specified as a Greenwald fraction (`f_GW`) [dimensionless] instead of
      [m^-3].
    T_i_T_e_ratio: Ratio of the ion and electron temperature at the pedestal top
      [dimensionless].
    rho_norm_top: Location of the pedestal top [$rho_norm$].
  """

  model_name: Annotated[
      Literal["set_p_top_n_top"], torax_pydantic.JAX_STATIC
  ] = "set_p_top_n_top"
  p_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1e5)
  )
  p_top_multiplier: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  n_e_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7e20)
  )
  n_e_top_is_f_GW: bool = False
  T_i_T_e_ratio: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  rho_norm_top: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_structure_model(
      self,
  ) -> (
      set_ptop_ttopratio_ntop.SetPressureTemperatureRatioAndDensityPedestalModel
  ):
    return (
        set_ptop_ttopratio_ntop.SetPressureTemperatureRatioAndDensityPedestalModel()
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_ptop_ttopratio_ntop.RuntimeParams:
    return set_ptop_ttopratio_ntop.RuntimeParams(
        **vars(super().build_runtime_params(t)),
        p_top=self.p_top.get_value(t),
        p_top_multiplier=self.p_top_multiplier.get_value(t),
        n_e_top=self.n_e_top.get_value(t),
        n_e_top_is_f_GW=self.n_e_top_is_f_GW,
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        rho_norm_top=self.rho_norm_top.get_value(t),
    )


class SetTtopNtopConfig(StructureConfigBase):
  """Directly specifies pedestal-top location, density, and temperatures.

  Attributes:
    model_name: Pedestal structure model name, set to `'set_T_top_n_top'`.
    n_e_top: Electron density at the pedestal top [m^-3 if
      `n_e_top_is_f_GW=False`, or Greenwald fraction if `n_e_top_is_f_GW=True`].
    n_e_top_is_f_GW: Whether the electron density at the pedestal top is
      specified as a Greenwald fraction (`f_GW`) [dimensionless] instead of
      [m^-3].
    T_i_top: Ion temperature at the pedestal top [keV].
    T_e_top: Electron temperature at the pedestal top [keV].
    rho_norm_top: Location of the pedestal top [$rho_norm$].
  """

  model_name: Annotated[
      Literal["set_T_top_n_top"], torax_pydantic.JAX_STATIC
  ] = "set_T_top_n_top"
  n_e_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.7e20)
  )
  n_e_top_is_f_GW: bool = False
  # TODO(b/434175938): Consider extending to TimeVaryingArray
  T_i_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  T_e_top: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(5.0)
  )
  rho_norm_top: torax_pydantic.UnitIntervalTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.91)
  )

  def build_structure_model(
      self,
  ) -> set_ttop_ntop.SetTemperatureDensityPedestalModel:
    return set_ttop_ntop.SetTemperatureDensityPedestalModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> set_ttop_ntop.RuntimeParams:
    return set_ttop_ntop.RuntimeParams(
        **vars(super().build_runtime_params(t)),
        n_e_top=self.n_e_top.get_value(t),
        n_e_top_is_f_GW=self.n_e_top_is_f_GW,
        T_i_top=self.T_i_top.get_value(t),
        T_e_top=self.T_e_top.get_value(t),
        rho_norm_top=self.rho_norm_top.get_value(t),
    )


class NoPedestalConfig(StructureConfigBase):
  """A pedestal structure model for when there is no pedestal.

  Attributes:
    model_name: Selects the pedestal structure model (`'no_pedestal'`,
      `'set_T_top_n_top'`, or `'set_p_top_n_top'`). If not set, defaults to
      `'no_pedestal'`.
  """

  model_name: Annotated[Literal["no_pedestal"], torax_pydantic.JAX_STATIC] = (
      "no_pedestal"
  )

  def build_structure_model(
      self,
  ) -> no_pedestal.NoPedestal:
    return no_pedestal.NoPedestal()


StructureConfig = Annotated[
    SetPtopTtopRatioNtopConfig | SetTtopNtopConfig | NoPedestalConfig,
    pydantic.Discriminator("model_name"),
]


class PedestalConfig(torax_pydantic.ConfigBase):
  """Configuration for the pedestal model.

  Attributes:
    structure_model: Configuration for the pedestal structure model, which
      computes the H-mode target pedestal-top temperature, density, radial
      location, and profile form. Custom models (such as |epednn_mit|) can also
      be registered via `torax.pedestal.register_pedestal_structure_model` (see
      :ref:`model-integration`).
    mode: Defines how the pedestal is generated. Options:

      - `'internal_boundary_condition'`: Sets the pedestal top value by
        directly modifying the state equations to set Dirichlet internal
        boundary conditions at a cell grid point corresponding to the pedestal
        top, or for a profile (see `pedestal_profile_form`). This is the
        default mode.
      - `'adaptive_transport'`: Sets the pedestal by modifying the transport
        coefficients in the pedestal region, allowing the pedestal to
        self-consistently evolve. Transport coefficients are scaled to allow
        the temperature and density to evolve towards the prescribed pedestal
        values.
    transition_time_width: Duration of the L-H or H-L transition ramp [s] for
      `'internal_boundary_condition'` mode. During a transition, the
      confinement mode enters `'TRANSITIONING_TO_H_MODE'` or
      `'TRANSITIONING_TO_L_MODE'` and pedestal values are linearly interpolated
      between the captured L-mode baseline values and the H-mode target values
      (e.g. `T_i_top(t)`, `T_e_top(t)`, `n_e_top(t)`) over this time window.
      Must be non-negative. Defaults to `0.0` (immediate step transition between
      `'l_mode'` and `'h_mode'`) for `'prescribed'` formation, and must be
      explicitly specified when using a power-scaling formation model
      (`'martin_scaling'` or `'delabie_scaling'`) in
      `'internal_boundary_condition'` mode. Note that if `transition_time_width
      > 0.0` is used with time-varying pedestal targets, those targets should
      represent the H-mode trajectory rather than manually encoding the L-H
      ramp, as the ramp scaling is applied on top of the target values.
    explicit_pedestal: Controls whether the pedestal model is evaluated
      explicitly (once per timestep) or implicitly (every internal iteration).
      When `True` (default), the pedestal model is evaluated once in the
      pre-step phase and its output is frozen during the solver loop. When
      `False`, the pedestal model is re-evaluated every Newton iteration,
      coupling pedestal physics to the implicit solve, but increasing the
      non-linearity of the system. Note: for `'adaptive_transport'` mode,
      transport multipliers are always re-evaluated implicitly with current
      profiles regardless of this setting, since the saturation model's
      profile-based feedback loop requires implicit coupling.
    formation_model: Configuration for the pedestal formation model, which
      determines when L-H and H-L transitions occur. Defaults to `{'model_name':
      'prescribed', 'pedestal_active': True}` in `'internal_boundary_condition'`
      mode, and must be explicitly specified when `mode` is
      `'adaptive_transport'`. Note that transition conditions are evaluated in
      the pre-step phase at the start of each timestep $[t, t + dt]$; a
      transition condition met at $t_trans$ takes effect on the first
      timestep starting at $t >= t_trans$ (and not during an earlier step that
      merely crosses $t_trans$).
    saturation_model: Configuration for the pedestal saturation model, which
      determines how the pedestal values scale relative to H-mode target values
      in `'adaptive_transport'` mode.
    chi_max: Maximum effective thermal diffusion coefficient from the core
      transport model allowed in the pedestal region (before applying
      `'adaptive_transport'`) [m^2/s]. Defaults to inf (no clipping).
    D_e_max: Maximum effective particle diffusion coefficient from the core
      transport model allowed in the pedestal region (before applying
      `'adaptive_transport'`) [m^2/s]. Defaults to inf (no clipping).
    V_e_max: Maximum effective particle pinch velocity from the core transport
      model allowed in the pedestal region (before applying
      `'adaptive_transport'`) [m/s].
    V_e_min: Minimum effective particle pinch velocity from the core transport
      model allowed in the pedestal region (before applying
      `'adaptive_transport'`) [m/s].
    pedestal_top_smoothing_width: Width of the smoothing region at the pedestal
      top boundary [$rho_norm$].
    initial_L_mode_bc_multiplier: Multiplier applied to the LCFS right-boundary
      values ($T_i$, $T_e$, $n_e$) at $t_initial$ to seed the L-mode
      pedestal-top baselines when the simulation starts directly in H-mode
      without prior L-mode profile data. Must be positive.
  """  # fmt: skip

  structure_model: StructureConfig = torax_pydantic.ValidatedDefault(
      NoPedestalConfig()
  )
  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.INTERNAL_BOUNDARY_CONDITION
  )
  transition_time_width: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  explicit_pedestal: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  formation_model: FormationConfig = torax_pydantic.ValidatedDefault(
      PrescribedFormationConfig(pedestal_active=True)
  )
  saturation_model: SaturationConfig = torax_pydantic.ValidatedDefault(
      ProfileValueSaturationConfig()
  )
  # TODO(b/491895183): Do a sweep across different cases to find good default
  # values for these parameters.
  chi_max: torax_pydantic.PositiveTimeVaryingScalar | None = None
  D_e_max: torax_pydantic.PositiveTimeVaryingScalar | None = None
  V_e_max: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      1.0
  )
  V_e_min: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      -1.0
  )
  pedestal_top_smoothing_width: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.02)
  )
  initial_L_mode_bc_multiplier: pydantic.PositiveFloat = 2.0

  @pydantic.model_validator(mode="before")
  @classmethod
  def _defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
    configurable_data = copy.deepcopy(data)
    if (
        isinstance(configurable_data.get("formation_model"), dict)
        and "model_name" not in configurable_data["formation_model"]
    ):
      configurable_data["formation_model"]["model_name"] = "prescribed"
    if (
        isinstance(configurable_data.get("saturation_model"), dict)
        and "model_name" not in configurable_data["saturation_model"]
    ):
      configurable_data["saturation_model"]["model_name"] = "profile_value"

    return configurable_data

  @pydantic.model_validator(mode="after")
  def _check_source_mode(self) -> Self:
    if isinstance(self.structure_model, NoPedestalConfig):
      if not isinstance(self.formation_model, PrescribedFormationConfig):
        raise ValueError(
            "NoPedestal cannot be used with a power-scaling formation model."
        )
      if self.formation_model.initial_confinement_mode == "h_mode":
        raise ValueError(
            "NoPedestal cannot have initial_confinement_mode='h_mode'."
        )
      return self

    if (
        "formation_model" not in self.model_fields_set
        and self.mode == runtime_params.Mode.ADAPTIVE_TRANSPORT
    ):
      raise ValueError(
          "formation_model must be explicitly specified when mode is"
          " ADAPTIVE_TRANSPORT."
      )
    if (
        self.mode == runtime_params.Mode.INTERNAL_BOUNDARY_CONDITION
        and "transition_time_width" not in self.model_fields_set
        and not isinstance(self.formation_model, PrescribedFormationConfig)
    ):
      raise ValueError(
          "transition_time_width must be specified when using a"
          " power-scaling formation model in INTERNAL_BOUNDARY_CONDITION"
          " mode."
      )
    return self

  def build_pedestal_model(self) -> pedestal_model.PedestalModel:
    """Builds the mode-specific pedestal model."""
    match self.mode:
      case runtime_params.Mode.INTERNAL_BOUNDARY_CONDITION:
        return pedestal_model.InternalBoundaryConditionPedestalModel(
            structure_model=self.structure_model.build_structure_model(),
            formation_model=self.formation_model.build_formation_model(),
        )
      case runtime_params.Mode.ADAPTIVE_TRANSPORT:
        return pedestal_model.AdaptiveTransportPedestalModel(
            structure_model=self.structure_model.build_structure_model(),
            formation_model=self.formation_model.build_formation_model(),
            saturation_model=self.saturation_model.build_saturation_model(),
        )
      case _:
        raise ValueError(f"Unknown pedestal mode: {self.mode}")

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime params."""
    formation_params = self.formation_model.build_runtime_params(t)
    if isinstance(self.structure_model, NoPedestalConfig) and isinstance(
        formation_params,
        prescribed_formation_model.PrescribedFormationRuntimeParams,
    ):
      formation_params = dataclasses.replace(
          formation_params, pedestal_active=False
      )
    return runtime_params.RuntimeParams(
        mode=self.mode,
        transition_time_width=self.transition_time_width.get_value(t),
        explicit_pedestal=self.explicit_pedestal,
        structure=self.structure_model.build_runtime_params(t),
        formation=formation_params,
        saturation=self.saturation_model.build_runtime_params(t),
        chi_max=(
            self.chi_max.get_value(t) if self.chi_max is not None else math.inf
        ),
        D_e_max=(
            self.D_e_max.get_value(t) if self.D_e_max is not None else math.inf
        ),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
        initial_L_mode_bc_multiplier=self.initial_L_mode_bc_multiplier,
    )
