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

"""Builder for active simulation internal boundary conditions."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name


def build_internal_boundary_conditions(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    pedestal_transition_state: (
        pedestal_transition_state_lib.PedestalTransitionState
    ),
) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
  """Builds the active internal boundary conditions for this time step.

  When a pedestal model is actively controlling the edge (either statically via
  set_pedestal=True without a formation model, or dynamically in H-mode or
  transition), configured internal boundary conditions are turned off to prevent
  conflicting boundary conditions or discontinuous cliffs at the edge.

  Args:
    runtime_params: Runtime parameters for the simulation.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles.
    pedestal_transition_state: Current state of the pedestal transition.

  Returns:
    The active InternalBoundaryConditions object.
  """
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    pedestal_model_output = pedestal_transition_state.pedestal_model_output
    if (
        runtime_params.pedestal.use_formation_model_with_internal_boundary_condition
    ):
      # Scale the pedestal output by the ramp fraction during transitions.
      # In H-mode, returns full H-mode values. In L-mode, returns L-mode values.
      # During transitions, linearly interpolates between the two.
      ramp_fraction = _compute_ramp_fraction(
          pedestal_transition_state=pedestal_transition_state,
          transition_time_width=runtime_params.pedestal.transition_time_width,
          t=runtime_params.t,
      )
      scaled_pedestal_model_output = _apply_transition_ramp_scaling(
          pedestal_transition_state=pedestal_transition_state,
          ramp_fraction=ramp_fraction,
      )
      pedestal_internal_boundary_conditions = (
          scaled_pedestal_model_output.to_internal_boundary_conditions(
              geo,
              core_profiles=core_profiles,
              pedestal_profile_form=(
                  runtime_params.pedestal.pedestal_profile_form
              ),
          )
      )
    else:
      pedestal_internal_boundary_conditions = (
          pedestal_model_output.to_internal_boundary_conditions(
              geo,
              core_profiles=core_profiles,
              pedestal_profile_form=(
                  runtime_params.pedestal.pedestal_profile_form
              ),
          )
      )
  else:
    # In ADAPTIVE_TRANSPORT mode, transport coefficients govern the edge;
    # edge internal boundary conditions are disabled.
    pedestal_internal_boundary_conditions = (
        internal_boundary_conditions_lib.InternalBoundaryConditions.empty(geo)
    )

  if runtime_params.internal_boundary_conditions is not None:
    config_internal_boundary_conditions = (
        runtime_params.internal_boundary_conditions
    )
  else:
    config_internal_boundary_conditions = (
        internal_boundary_conditions_lib.InternalBoundaryConditions.empty(geo)
    )

  is_pedestal_ibc_active = _is_pedestal_ibc_active(
      runtime_params.pedestal, pedestal_transition_state
  )
  # Use jax.tree.map with jnp.where to conditionally select between the two
  # InternalBoundaryConditions dataclass instances across all leaf arrays
  # without triggering tracer boolean evaluation during JIT compilation.
  return jax.tree.map(
      lambda p, u: jnp.where(is_pedestal_ibc_active, p, u),
      pedestal_internal_boundary_conditions,
      config_internal_boundary_conditions,
  )


def _is_pedestal_ibc_active(
    pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
    transition_state: pedestal_transition_state_lib.PedestalTransitionState,
) -> array_typing.BoolScalar:
  """Returns whether the pedestal IBC is actively controlling the edge."""
  # 1. Static check: must be in IBC mode
  if (
      pedestal_runtime_params.mode
      != pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    return False

  # 2. Dynamic check: pedestal setting must be enabled
  is_active = pedestal_runtime_params.set_pedestal

  # 3. If using formation model, dynamic check that we are not in L-mode
  if (
      pedestal_runtime_params.use_formation_model_with_internal_boundary_condition
  ):
    not_l_mode = (
        transition_state.confinement_mode
        != pedestal_transition_state_lib.ConfinementMode.L_MODE
    )
    is_active = is_active & not_l_mode

  return is_active


def _compute_ramp_fraction(
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
    transition_time_width: array_typing.FloatScalar,
    t: array_typing.FloatScalar,
) -> array_typing.FloatScalar:
  """Computes the ramp fraction for a pedestal transition.

  Returns a value in [0, 1] representing the progress of the current
  transition. 0 means the transition just started, 1 means it is complete.

  Args:
    pedestal_transition_state: Current transition state.
    transition_time_width: Duration of the transition ramp.
    t: Current simulation time (i.e. t + dt when called from the solver).

  Returns:
    Ramp fraction clipped to [0, 1].
  """
  elapsed = t - pedestal_transition_state.transition_start_time
  fraction = elapsed / transition_time_width
  return jnp.clip(fraction, 0.0, 1.0)


def _apply_transition_ramp_scaling(
    pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
    ramp_fraction: array_typing.FloatScalar,
) -> pedestal_model_output_lib.PedestalModelOutput:
  """Applies ramp scaling to internal boundary conditions during transitions.

  During an L-H transition, linearly ramps from L-mode values to the H-mode
  targets. During an H-L transition, ramps from the H-mode targets back to
  the L-mode values.

  The L-mode values are stored in the pedestal_transition_state (captured
  at the start of an L->H transition). The H-mode targets are the full
  pedestal model output.

  Args:
    pedestal_transition_state: Current transition state containing L-mode
      baseline values and the pedestal model output.
    ramp_fraction: Progress of the current transition, in [0, 1].

  Returns:
    Scaled pedestal model output.
  """

  def _interpolate_transition(l_val, h_val):
    """Interpolates between L-mode and H-mode values based on confinement mode."""
    l_to_h_ramp = l_val + ramp_fraction * (h_val - l_val)
    h_to_l_ramp = h_val + ramp_fraction * (l_val - h_val)
    confinement_mode = pedestal_transition_state.confinement_mode
    return jnp.select(
        [
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.L_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.H_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
            confinement_mode
            == pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
        ],
        [l_val, h_val, l_to_h_ramp, h_to_l_ramp],
    )

  pedestal_model_output = pedestal_transition_state.pedestal_model_output

  scaled_T_i = _interpolate_transition(
      l_val=pedestal_transition_state.T_i_ped_L_mode,
      h_val=pedestal_model_output.T_i_ped,
  )
  scaled_T_e = _interpolate_transition(
      l_val=pedestal_transition_state.T_e_ped_L_mode,
      h_val=pedestal_model_output.T_e_ped,
  )
  scaled_n_e = _interpolate_transition(
      l_val=pedestal_transition_state.n_e_ped_L_mode,
      h_val=pedestal_model_output.n_e_ped,
  )

  return dataclasses.replace(
      pedestal_model_output,
      T_i_ped=scaled_T_i,
      T_e_ped=scaled_T_e,
      n_e_ped=scaled_n_e,
  )
