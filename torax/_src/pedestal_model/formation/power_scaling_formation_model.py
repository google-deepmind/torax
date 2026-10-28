# Copyright 2026 DeepMind Technologies Limited
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

"""Power scaling pedestal formation model."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base
from torax._src.physics import formulas
from torax._src.physics import scaling_laws
from torax._src.sources import source_profiles as source_profiles_lib
from typing_extensions import override

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PowerScalingFormationRuntimeParams(
    pedestal_runtime_params_lib.FormationRuntimeParams
):
  """Runtime params for power scaling pedestal formation models."""

  sharpness: array_typing.FloatScalar
  offset: array_typing.FloatScalar
  P_LH_prefactor: array_typing.FloatScalar
  P_LH_hysteresis_factor: array_typing.FloatScalar
  include_dW_dt_in_P_SOL: bool = jax.tree.static()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PowerScalingFormationState(pedestal_state_lib.FormationState):
  """State for power scaling pedestal formation models.

  Attributes:
    W_thermal_total: Total thermal stored energy [J] at time `t`, or `None` when
      `include_dW_dt_in_P_SOL` is False.
    dW_thermal_dt_smoothed: Exponentially smoothed time derivative of total
      thermal stored energy [W], or `None` when `include_dW_dt_in_P_SOL` is
      False.
    t: Simulation time [s] at which `W_thermal_total` and
      `dW_thermal_dt_smoothed` were last evaluated, or `None` when
      `include_dW_dt_in_P_SOL` is False.
  """

  W_thermal_total: array_typing.FloatScalar | None = None
  dW_thermal_dt_smoothed: array_typing.FloatScalar | None = None
  t: array_typing.FloatScalar | None = None


@jax.jit
def calculate_P_SOL_total(
    core_sources: source_profiles_lib.SourceProfiles,
    geo: geometry.Geometry,
    dW_thermal_dt_smoothed: array_typing.FloatScalar | None = None,
) -> jax.Array:
  """Calculates the total power out of the separatrix.

  Args:
    core_sources: Source profiles.
    geo: Geometry.
    dW_thermal_dt_smoothed: Smoothed time derivative of total thermal stored
      energy [W]. If None, returns P_heat (total heating power) without the
      dW/dt correction.

  Returns:
    P_SOL or P_heat total [W].
  """
  P_heat_e = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_e.values()
  )
  P_heat_i = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_i.values()
  )
  P_heat_total = jnp.asarray(P_heat_e + P_heat_i)
  if dW_thermal_dt_smoothed is None:
    return P_heat_total
  return P_heat_total - dW_thermal_dt_smoothed


@dataclasses.dataclass(frozen=True, eq=False)
class PowerScalingFormationModel(base.FormationModel):
  """Pedestal formation based on P_SOL and P_LH thresholds.

  Attributes:
    scaling_law: The scaling law to use for pedestal formation.
    divertor_configuration: The divertor configuration. Only used for the
      Delabie scaling law.
  """

  scaling_law: scaling_laws.PLHScalingLaw
  divertor_configuration: scaling_laws.DivertorConfiguration = (
      scaling_laws.DivertorConfiguration.HORIZONTAL_TARGET
  )

  @override
  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PowerScalingFormationState:
    """Returns the initial PowerScalingFormationState for this model."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )
    if not runtime_params.pedestal.formation.include_dW_dt_in_P_SOL:
      return PowerScalingFormationState()
    _, _, W_thermal_total = formulas.calculate_stored_thermal_energy(
        core_profiles.pressure_thermal_e,
        core_profiles.pressure_thermal_i,
        core_profiles.pressure_thermal_total,
        geo,
    )
    return PowerScalingFormationState(
        W_thermal_total=W_thermal_total,
        dW_thermal_dt_smoothed=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        t=jnp.asarray(runtime_params.t, dtype=jax_utils.get_dtype()),
    )

  @override
  def update_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      formation_state: pedestal_state_lib.FormationState,
  ) -> PowerScalingFormationState:
    """Updates PowerScalingFormationState for the current timestep."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )
    assert isinstance(formation_state, PowerScalingFormationState)
    if not runtime_params.pedestal.formation.include_dW_dt_in_P_SOL:
      return formation_state
    assert formation_state.W_thermal_total is not None
    assert formation_state.dW_thermal_dt_smoothed is not None
    assert formation_state.t is not None
    dt = runtime_params.t - formation_state.t
    _, _, W_thermal_total = formulas.calculate_stored_thermal_energy(
        core_profiles.pressure_thermal_e,
        core_profiles.pressure_thermal_i,
        core_profiles.pressure_thermal_total,
        geo,
    )

    def _advance_state() -> PowerScalingFormationState:
      assert formation_state.W_thermal_total is not None
      assert formation_state.dW_thermal_dt_smoothed is not None
      _, dW_thermal_dt_smoothed = (
          formulas.calculate_time_derivative_with_exponential_smoothing(
              current_value=W_thermal_total,
              previous_value=formation_state.W_thermal_total,
              previous_smoothed_derivative=(
                  formation_state.dW_thermal_dt_smoothed
              ),
              dt=jnp.where(dt > 0.0, dt, 1.0),
              smoothing_time_scale=(
                  runtime_params.numerics.dW_dt_smoothing_time_scale
              ),
          )
      )
      return PowerScalingFormationState(
          W_thermal_total=W_thermal_total,
          dW_thermal_dt_smoothed=dW_thermal_dt_smoothed,
          t=jnp.asarray(runtime_params.t, dtype=jax_utils.get_dtype()),
      )

    # When theta_implicit < 1.0, calc_coeffs evaluates explicit PDE terms at
    # start-of-step time t (where dt == 0.0); preserve formation_state to
    # avoid 0.0 / 0.0 in the raw time derivative.
    return jax.lax.cond(
        dt > 0.0,
        _advance_state,
        lambda: formation_state,
    )

  def _calculate_P_SOL_and_P_LH(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      formation_state: PowerScalingFormationState,
  ) -> tuple[jax.Array, jax.Array]:
    """Calculates P_SOL and rescaled P_LH."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )
    P_SOL_total = calculate_P_SOL_total(
        core_sources,
        geo,
        dW_thermal_dt_smoothed=formation_state.dW_thermal_dt_smoothed,
    )
    P_LH, _ = scaling_laws.calculate_P_LH(
        geo,
        core_profiles,
        scaling_law=self.scaling_law,
        divertor_configuration=self.divertor_configuration,
    )
    rescaled_P_LH = P_LH * runtime_params.pedestal.formation.P_LH_prefactor
    return P_SOL_total, rescaled_P_LH

  @override
  def evaluate_transition_conditions(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      formation_state: pedestal_state_lib.FormationState,
  ) -> tuple[
      array_typing.BoolScalar,
      array_typing.BoolScalar,
  ]:
    """Evaluates L-H and H-L transition triggers based on P_SOL vs P_LH."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )
    assert isinstance(formation_state, PowerScalingFormationState)
    P_SOL_total, rescaled_P_LH = self._calculate_P_SOL_and_P_LH(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        formation_state,
    )
    trigger_l_to_h_transition = P_SOL_total > rescaled_P_LH
    trigger_h_to_l_transition = (
        P_SOL_total
        < rescaled_P_LH
        * runtime_params.pedestal.formation.P_LH_hysteresis_factor
    )
    return (
        trigger_l_to_h_transition,
        trigger_h_to_l_transition,
    )

  @override
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      confinement_mode: (
          pedestal_runtime_params_lib.ConfinementMode | array_typing.IntScalar
      ),
      formation_state: pedestal_state_lib.FormationState,
  ) -> pedestal_model_output.TransportMultipliers:
    """Calculates transport decrease multipliers based on P_SOL and P_LH."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )
    assert isinstance(formation_state, PowerScalingFormationState)
    P_SOL_total, rescaled_P_LH = self._calculate_P_SOL_and_P_LH(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        formation_state,
    )

    # Apply hysteresis: in H-mode, use a lower effective P_LH threshold,
    # making it harder to transition back to L-mode.
    effective_P_LH = jnp.where(
        confinement_mode == pedestal_runtime_params_lib.ConfinementMode.H_MODE,
        rescaled_P_LH
        * runtime_params.pedestal.formation.P_LH_hysteresis_factor,
        rescaled_P_LH,
    )

    # Calculate transport_multiplier
    # If P_SOL > effective_P_LH, multiplier tends to 0.0
    # If P_SOL < effective_P_LH, multiplier tends to 1.0
    sharpness = runtime_params.pedestal.formation.sharpness
    offset = runtime_params.pedestal.formation.offset
    base_multiplier = runtime_params.pedestal.formation.base_multiplier
    normalized_deviation = (P_SOL_total - effective_P_LH) / effective_P_LH
    shifted_deviation = normalized_deviation - offset
    alpha = jax.nn.sigmoid(shifted_deviation * sharpness)
    transport_multiplier = (1.0 - alpha) * 1.0 + alpha * base_multiplier

    return pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=transport_multiplier,
        chi_i_multiplier=transport_multiplier,
        D_e_multiplier=transport_multiplier,
        v_e_multiplier=transport_multiplier,
    )
