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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""

import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base as formation_base
from torax._src.pedestal_model.saturation import base as saturation_base
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates temperature and density of the pedestal."""

  formation_model: formation_base.FormationModel
  saturation_model: saturation_base.SaturationModel

  def compute_transport_multipliers(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> pedestal_model_output.TransportMultipliers:
    """Computes transport multipliers from formation and saturation models."""

    transport_decrease = self.formation_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state,
    )
    transport_increase = self.saturation_model(
        runtime_params, geo, core_profiles, pedestal_output
    )

    # Combine via exp(log) for numerical stability, as multipliers can
    # be very small or large.
    return jax.tree.map(
        lambda x, y: jnp.exp(jnp.log(x) + jnp.log(y)),
        transport_decrease,
        transport_increase,
    )

  def _evaluate_pedestal(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    pedestal_output = self._call_implementation(
        runtime_params, geo, core_profiles, pedestal_transition_state,
    )

    # If in ADAPTIVE_TRANSPORT mode, calculate the transport multipliers based
    # on the formation and saturation models.
    if (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    ):
      transport_multipliers = self.compute_transport_multipliers(
          runtime_params,
          geo,
          core_profiles,
          source_profiles,
          pedestal_transition_state,
          pedestal_output,
      )
      pedestal_output = dataclasses.replace(
          pedestal_output, transport_multipliers=transport_multipliers
      )

    return pedestal_output

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        self._evaluate_pedestal,
        lambda *_: pedestal_model_output.PedestalModelOutput.no_pedestal(),
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state,
    )

  def update_transition_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: (
          pedestal_transition_state_lib.PedestalTransitionState
      ),
  ) -> pedestal_transition_state_lib.PedestalTransitionState:
    """Evaluates transition conditions and updates the transition state.

    Called once per timestep in pre_step. Evaluates the formation model's
    transition conditions, then delegates to the mode-specific update function.

    Args:
      runtime_params: Runtime parameters at time t.
      geo: Geometry at time t.
      core_profiles: Core plasma profiles at time t.
      source_profiles: Source profiles at time t.
      pedestal_transition_state: Current transition state from previous
        timestep.

    Returns:
      Updated PedestalTransitionState.
    """
    if (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
        and not runtime_params.pedestal.use_formation_model_with_internal_boundary_condition
    ):
      return pedestal_transition_state

    trigger_l_to_h_transition, trigger_h_to_l_transition = (
        self.formation_model.evaluate_transition_conditions(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            source_profiles=source_profiles,
        )
    )
    match runtime_params.pedestal.mode:
      case pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT:
        return _update_adaptive_transport(
            pedestal_transition_state,
            trigger_l_to_h_transition,
            trigger_h_to_l_transition,
        )
      case pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION:
        return _update_internal_boundary_condition(
            pedestal_transition_state,
            runtime_params,
            geo,
            core_profiles,
            trigger_l_to_h_transition,
            trigger_h_to_l_transition,
        )
      case _:
        raise ValueError(
            f'Unknown pedestal mode: {runtime_params.pedestal.mode}'
        )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.PedestalModelOutput:
    """Calculate the pedestal properties."""


def _update_adaptive_transport(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    trigger_l_to_h_transition: array_typing.BoolScalar,
    trigger_h_to_l_transition: array_typing.BoolScalar,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Updates pedestal transition state for ADAPTIVE_TRANSPORT mode.

  Simple L-mode <-> H-mode transitions with hysteresis. The sigmoid in the
  formation model handles the dynamics of the transition.

  Note that ConfinementMode reflects the discrete mode determined from the
  transition conditions evaluated at time t (the start of the timestep).

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    trigger_l_to_h_transition: Boolean condition to trigger L->H transition.
    trigger_h_to_l_transition: Boolean condition to trigger H->L
      back-transition.

  Returns:
    Updated PedestalTransitionState.
  """
  old_confinement_mode = pedestal_transition_state.confinement_mode
  new_confinement_mode = jnp.select(
      [
          # L-H transition.
          (
              old_confinement_mode
              == pedestal_transition_state_lib.ConfinementMode.L_MODE
          )
          & trigger_l_to_h_transition,
          # H-L back transition, with hysteresis.
          (
              old_confinement_mode
              == pedestal_transition_state_lib.ConfinementMode.H_MODE
          )
          & trigger_h_to_l_transition,
      ],
      [
          pedestal_transition_state_lib.ConfinementMode.H_MODE,
          pedestal_transition_state_lib.ConfinementMode.L_MODE,
      ],
      default=old_confinement_mode,
  )

  return pedestal_transition_state_lib.PedestalTransitionState(
      confinement_mode=new_confinement_mode,
      transition_start_time=pedestal_transition_state.transition_start_time,
      T_i_ped_L_mode=pedestal_transition_state.T_i_ped_L_mode,
      T_e_ped_L_mode=pedestal_transition_state.T_e_ped_L_mode,
      n_e_ped_L_mode=pedestal_transition_state.n_e_ped_L_mode,
      pedestal_model_output=pedestal_transition_state.pedestal_model_output,
      previous_pedestal_model_output=pedestal_transition_state.previous_pedestal_model_output,
  )


def _update_internal_boundary_condition(
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    trigger_l_to_h_transition: array_typing.BoolScalar,
    trigger_h_to_l_transition: array_typing.BoolScalar,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Updates pedestal transition state for INTERNAL_BOUNDARY_CONDITION mode.

  Full 4-state machine with TRANSITIONING_TO_H/L states, transition timers,
  dithering support, and L-mode value capture for ramp interpolation. When
  transition_time_width == 0.0, transitions step directly between L_MODE and
  H_MODE.

  When transitioning from L-mode to H-mode:
    - Records the current simulation time as transition_start_time
    - Saves the current kinetic profile values at the pedestal-top for the
      lower target values when setting up pedestal ramp up/down.

  When transitioning from H-mode to L-mode:
    - Records the current simulation time as transition_start_time
    - Loads the saved L-mode pedestal-top values as a target for the end of
      the transition.

  If mid-transition and then starting to revert back to the original
  confinement mode:
    - Sets the transition_start_time such that the time spent in the
      reverse transition is the same as the time spent in the original
      transition.

  Args:
    pedestal_transition_state: Current transition state from previous timestep.
    runtime_params: Runtime parameters at time t.
    geo: Geometry at time t.
    core_profiles: Core plasma profiles at time t.
    trigger_l_to_h_transition: Boolean condition to trigger L->H transition.
    trigger_h_to_l_transition: Boolean condition to trigger H->L
      back-transition.

  Returns:
    Updated PedestalTransitionState.
  """
  old_confinement_mode = pedestal_transition_state.confinement_mode

  # Has the transition time elapsed? Used for exiting transition states.
  elapsed_transition_time = (
      runtime_params.t - pedestal_transition_state.transition_start_time
  )
  transition_is_complete = (
      elapsed_transition_time >= runtime_params.pedestal.transition_time_width
  )
  instant_transition = runtime_params.pedestal.transition_time_width == 0.0
  l_to_h_transition_target_mode = jnp.where(
      instant_transition,
      pedestal_transition_state_lib.ConfinementMode.H_MODE,
      pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
  )
  h_to_l_transition_target_mode = jnp.where(
      instant_transition,
      pedestal_transition_state_lib.ConfinementMode.L_MODE,
      pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
  )

  conditions = [
      # Completed LH transition. Checked first so that completed transitions
      # take priority over starting new transitions in jnp.select.
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE
      )
      & transition_is_complete,
      # Completed HL transition.
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE
      )
      & transition_is_complete,
      # L-H transition.
      (
          old_confinement_mode
          != pedestal_transition_state_lib.ConfinementMode.H_MODE
      )
      & trigger_l_to_h_transition,
      # H-L back transition, with hysteresis.
      (
          old_confinement_mode
          != pedestal_transition_state_lib.ConfinementMode.L_MODE
      )
      & trigger_h_to_l_transition,
  ]
  new_confinement_modes = [
      pedestal_transition_state_lib.ConfinementMode.H_MODE,
      pedestal_transition_state_lib.ConfinementMode.L_MODE,
      l_to_h_transition_target_mode,
      h_to_l_transition_target_mode,
  ]
  new_confinement_mode = jnp.select(
      conditions,
      new_confinement_modes,
      old_confinement_mode,
  )

  # Update the transition start time.
  # - If we've gone from L-mode to LH transition, or from H-mode to HL
  #   transition, set the start time to the current time.
  # - If we are dithering, set the start time so that we
  #   spend the same amount of time in the back-transition as we did in the
  #   forward transition.
  # - Otherwise, keep the current start time.
  standard_transition = (
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.L_MODE
      )
      & (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE
      )
  ) | (
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.H_MODE
      )
      & (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE
      )
  )
  dithering_transition = (
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE
      )
      & (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE
      )
  ) | (
      (
          old_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE
      )
      & (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE
      )
  )
  # At time t, suppose we have been in forward transition since t0, with desired
  # transition duration w. If we now begin a back-transition, we have been in
  # forward transition for t - t0, and so we want to end the back-transition at
  # time t + (t - t0) = 2t - t0. As we have specified the transition time width
  # as w, to achieve this we set the start time to (2t - t0) - w.
  new_transition_start_time = jnp.select(
      [standard_transition, dithering_transition],
      [
          runtime_params.t,
          2.0 * runtime_params.t
          - pedestal_transition_state.transition_start_time
          - runtime_params.pedestal.transition_time_width,
      ],
      # Otherwise, preserve the current transition start time. This covers
      # both ongoing transitions (where the mode hasn't changed) and
      # non-transition states (H_MODE/L_MODE where start_time is unused).
      default=pedestal_transition_state.transition_start_time,
  )

  # Update the target values for transitions to L-mode.
  # Only needed for INTERNAL_BOUNDARY_CONDITION, which uses ramp interpolation.
  update_L_mode_values = (
      old_confinement_mode
      == pedestal_transition_state_lib.ConfinementMode.L_MODE
  ) & (
      (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE
      )
      | (
          new_confinement_mode
          == pedestal_transition_state_lib.ConfinementMode.H_MODE
      )
  )
  ped_top_idx = jnp.argmin(
      jnp.abs(
          geo.rho_norm
          - pedestal_transition_state.pedestal_model_output.rho_norm_ped_top
      )
  )
  new_T_i_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.T_i.value[ped_top_idx],
      pedestal_transition_state.T_i_ped_L_mode,
  )
  new_T_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.T_e.value[ped_top_idx],
      pedestal_transition_state.T_e_ped_L_mode,
  )
  new_n_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.n_e.value[ped_top_idx],
      pedestal_transition_state.n_e_ped_L_mode,
  )

  return pedestal_transition_state_lib.PedestalTransitionState(
      confinement_mode=new_confinement_mode,
      transition_start_time=new_transition_start_time,
      T_i_ped_L_mode=new_T_i_ped_L_mode,
      T_e_ped_L_mode=new_T_e_ped_L_mode,
      n_e_ped_L_mode=new_n_e_ped_L_mode,
      pedestal_model_output=pedestal_transition_state.pedestal_model_output,
      previous_pedestal_model_output=pedestal_transition_state.previous_pedestal_model_output,
  )
