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
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)
from torax._src.pedestal_model.formation import base as formation_base
from torax._src.pedestal_model.saturation import base as saturation_base
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import pedestal_structure
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates pedestal structure and transport multipliers."""

  structure_model: structure_base.PedestalStructureModel
  formation_model: formation_base.FormationModel

  @abc.abstractmethod
  def initialize(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      pedestal_model_output.PedestalModelOutput,
      pedestal_state_lib.PedestalState,
  ]:
    """Initializes the pedestal state and output."""

  @abc.abstractmethod
  def pre_step(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> tuple[
      pedestal_state_lib.PedestalState,
      pedestal_structure.PedestalStructure | None,
  ]:
    """Evaluates transition conditions and updates pedestal state."""

  @abc.abstractmethod
  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
      explicit_pedestal_structure: (
          pedestal_structure.PedestalStructure | None
      ) = None,
  ) -> tuple[
      pedestal_model_output.PedestalModelOutput,
      pedestal_state_lib.PedestalState,
  ]:
    """Calculates pedestal output and updated state."""


@dataclasses.dataclass(frozen=True, eq=False)
class InternalBoundaryConditionPedestalModel(PedestalModel):
  """Calculates pedestal structure for INTERNAL_BOUNDARY_CONDITION mode."""

  def initialize(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      pedestal_model_output.IBCPedestalModelOutput,
      pedestal_state_lib.IBCPedestalState,
  ]:
    """Initializes the pedestal state and output for IBC mode.

    Args:
      runtime_params: Runtime parameters at initial time.
      geo: Torus geometry.
      core_profiles: Initial core plasma profiles.
      source_profiles: Initial source profiles.

    Returns:
      Tuple of (IBCPedestalModelOutput, IBCPedestalState).
    """
    structure_state = self.structure_model.initial_state(
        runtime_params, geo, core_profiles
    )
    formation_state = self.formation_model.initial_state(
        runtime_params, geo, core_profiles
    )
    confinement_mode = self.formation_model.initial_confinement_mode(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        formation_state,
    )
    structure, structure_state = self.structure_model(
        runtime_params, geo, core_profiles, structure_state
    )
    ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - structure.rho_norm_ped_top)
    )
    in_l_mode = (
        confinement_mode == pedestal_runtime_params_lib.ConfinementMode.L_MODE
    )
    bc_multiplier = runtime_params.pedestal.initial_L_mode_bc_multiplier
    T_i_ped_L_mode = jnp.where(
        in_l_mode,
        core_profiles.kinetic.T_i.value[ped_top_idx],
        bc_multiplier * core_profiles.kinetic.T_i.face_value()[-1],
    )
    T_e_ped_L_mode = jnp.where(
        in_l_mode,
        core_profiles.kinetic.T_e.value[ped_top_idx],
        bc_multiplier * core_profiles.kinetic.T_e.face_value()[-1],
    )
    n_e_ped_L_mode = jnp.where(
        in_l_mode,
        core_profiles.kinetic.n_e.value[ped_top_idx],
        bc_multiplier * core_profiles.kinetic.n_e.face_value()[-1],
    )
    pedestal_state = pedestal_state_lib.IBCPedestalState(
        structure=structure_state,
        formation=formation_state,
        confinement_mode=confinement_mode,
        T_i_ped_L_mode=T_i_ped_L_mode,
        T_e_ped_L_mode=T_e_ped_L_mode,
        n_e_ped_L_mode=n_e_ped_L_mode,
        rho_norm_ped_top=structure.rho_norm_ped_top,
    )
    # No transition ramp occurs at initialization (confinement_mode is L_MODE
    # or H_MODE and transition_start_time is -inf); this only selects the
    # initial L-mode baseline vs H-mode target to match __call__.
    scaled_structure = pedestal_state.scale_structure(
        structure,
        t=runtime_params.t,
        transition_time_width=runtime_params.pedestal.transition_time_width,
    )
    return (
        pedestal_model_output.IBCPedestalModelOutput(
            structure=scaled_structure,
            confinement_mode=confinement_mode,
        ),
        pedestal_state,
    )

  def pre_step(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> tuple[
      pedestal_state_lib.IBCPedestalState,
      pedestal_structure.PedestalStructure | None,
  ]:
    """Updates IBC transition state and evaluates explicit structure.

    Args:
      runtime_params: Runtime parameters at time t.
      geo: Torus geometry at time t.
      core_profiles: Core plasma profiles at time t.
      source_profiles: Source profiles at time t.
      pedestal_state: Cross-step pedestal state from previous step.

    Returns:
      Tuple of (updated_pedestal_state, explicit_structure or None).
    """
    assert isinstance(pedestal_state, pedestal_state_lib.IBCPedestalState)
    trigger_l_to_h, trigger_h_to_l = (
        self.formation_model.evaluate_transition_conditions(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            source_profiles=source_profiles,
            formation_state=pedestal_state.formation,
        )
    )
    updated_pedestal_state = _update_ibc_pedestal_state(
        pedestal_state=pedestal_state,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        trigger_l_to_h=trigger_l_to_h,
        trigger_h_to_l=trigger_h_to_l,
    )
    if runtime_params.pedestal.explicit_pedestal:
      explicit_structure, structure_state = self.structure_model(
          runtime_params,
          geo,
          core_profiles,
          updated_pedestal_state.structure,
      )
      updated_pedestal_state = dataclasses.replace(
          updated_pedestal_state,
          structure=structure_state,
      )
    else:
      explicit_structure = None
    return updated_pedestal_state, explicit_structure

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
      explicit_pedestal_structure: (
          pedestal_structure.PedestalStructure | None
      ) = None,
  ) -> tuple[
      pedestal_model_output.IBCPedestalModelOutput,
      pedestal_state_lib.IBCPedestalState,
  ]:
    """Calculates IBC output for the current timestep or solver iteration.

    Args:
      runtime_params: Runtime parameters at time t.
      geo: Torus geometry.
      core_profiles: Current core plasma profiles.
      source_profiles: Current source profiles.
      pedestal_state: Cross-step pedestal state.
      explicit_pedestal_structure: Optional explicit structure from pre_step.

    Returns:
      Tuple of (IBCPedestalModelOutput, updated IBCPedestalState).
    """
    del source_profiles
    assert isinstance(pedestal_state, pedestal_state_lib.IBCPedestalState)
    if explicit_pedestal_structure is not None:
      structure = explicit_pedestal_structure
      structure_state = pedestal_state.structure
    else:
      structure, structure_state = self.structure_model(
          runtime_params, geo, core_profiles, pedestal_state.structure
      )
    formation_state = self.formation_model.update_state(
        runtime_params, geo, core_profiles, pedestal_state.formation
    )
    updated_state = dataclasses.replace(
        pedestal_state,
        structure=structure_state,
        formation=formation_state,
        rho_norm_ped_top=structure.rho_norm_ped_top,
    )
    scaled_structure = updated_state.scale_structure(
        structure,
        t=runtime_params.t,
        transition_time_width=runtime_params.pedestal.transition_time_width,
    )
    return (
        pedestal_model_output.IBCPedestalModelOutput(
            structure=scaled_structure,
            confinement_mode=updated_state.confinement_mode,
        ),
        updated_state,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class AdaptiveTransportPedestalModel(PedestalModel):
  """Calculates pedestal structure and multipliers for ADAPTIVE_TRANSPORT."""

  saturation_model: saturation_base.SaturationModel = dataclasses.field()

  def initialize(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[
      pedestal_model_output.AdaptiveTransportPedestalModelOutput,
      pedestal_state_lib.AdaptiveTransportPedestalState,
  ]:
    """Initializes pedestal state and output for adaptive transport mode.

    Args:
      runtime_params: Runtime parameters at initial time.
      geo: Torus geometry.
      core_profiles: Initial core plasma profiles.
      source_profiles: Initial source profiles.

    Returns:
      Tuple of (AdaptiveTransportPedestalModelOutput,
      AdaptiveTransportPedestalState).
    """
    structure_state = self.structure_model.initial_state(
        runtime_params, geo, core_profiles
    )
    formation_state = self.formation_model.initial_state(
        runtime_params, geo, core_profiles
    )
    confinement_mode = self.formation_model.initial_confinement_mode(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        formation_state,
    )
    structure, structure_state = self.structure_model(
        runtime_params, geo, core_profiles, structure_state
    )
    transport_multipliers = self._compute_transport_multipliers(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        confinement_mode,
        structure,
        formation_state,
    )
    pedestal_state = pedestal_state_lib.AdaptiveTransportPedestalState(
        structure=structure_state,
        formation=formation_state,
        confinement_mode=confinement_mode,
    )
    output = pedestal_model_output.AdaptiveTransportPedestalModelOutput(
        structure=structure,
        confinement_mode=confinement_mode,
        transport_multipliers=transport_multipliers,
    )
    return output, pedestal_state

  def pre_step(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
  ) -> tuple[
      pedestal_state_lib.AdaptiveTransportPedestalState,
      pedestal_structure.PedestalStructure | None,
  ]:
    """Evaluates transition conditions and updates confinement mode.

    Args:
      runtime_params: Runtime parameters at time t.
      geo: Torus geometry at time t.
      core_profiles: Core plasma profiles at time t.
      source_profiles: Source profiles at time t.
      pedestal_state: Cross-step pedestal state from previous step.

    Returns:
      Tuple of (updated_state, explicit_structure or None).
    """
    assert isinstance(
        pedestal_state, pedestal_state_lib.AdaptiveTransportPedestalState
    )
    trigger_l_to_h, trigger_h_to_l = (
        self.formation_model.evaluate_transition_conditions(
            runtime_params=runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            source_profiles=source_profiles,
            formation_state=pedestal_state.formation,
        )
    )
    ConfinementMode = pedestal_runtime_params_lib.ConfinementMode
    old_confinement_mode = pedestal_state.confinement_mode
    new_confinement_mode = jnp.select(
        [
            (old_confinement_mode == ConfinementMode.L_MODE) & trigger_l_to_h,
            (old_confinement_mode == ConfinementMode.H_MODE) & trigger_h_to_l,
        ],
        [
            ConfinementMode.H_MODE,
            ConfinementMode.L_MODE,
        ],
        default=old_confinement_mode,
    )
    if runtime_params.pedestal.explicit_pedestal:
      explicit_structure, structure_state = self.structure_model(
          runtime_params, geo, core_profiles, pedestal_state.structure
      )
    else:
      explicit_structure = None
      structure_state = pedestal_state.structure
    updated_state = pedestal_state_lib.AdaptiveTransportPedestalState(
        structure=structure_state,
        formation=pedestal_state.formation,
        confinement_mode=new_confinement_mode,
    )
    return updated_state, explicit_structure

  def _compute_transport_multipliers(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      confinement_mode: (
          pedestal_runtime_params_lib.ConfinementMode | array_typing.IntScalar
      ),
      structure: pedestal_structure.PedestalStructure,
      formation_state: pedestal_state_lib.FormationState,
  ) -> pedestal_model_output.TransportMultipliers:
    """Computes transport multipliers from formation and saturation models."""
    transport_decrease = self.formation_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        confinement_mode,
        formation_state,
    )
    transport_increase = self.saturation_model(
        runtime_params,
        geo,
        core_profiles,
        structure,
    )
    # Combine via exp(log) for numerical stability, as multipliers can
    # be very small or large.
    return jax.tree.map(
        lambda x, y: jnp.exp(jnp.log(x) + jnp.log(y)),
        transport_decrease,
        transport_increase,
    )

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_state: pedestal_state_lib.PedestalState,
      explicit_pedestal_structure: (
          pedestal_structure.PedestalStructure | None
      ) = None,
  ) -> tuple[
      pedestal_model_output.AdaptiveTransportPedestalModelOutput,
      pedestal_state_lib.AdaptiveTransportPedestalState,
  ]:
    """Calculates adaptive transport output for timestep or solver iteration.

    Args:
      runtime_params: Runtime parameters at time t.
      geo: Torus geometry.
      core_profiles: Core plasma profiles.
      source_profiles: Source profiles.
      pedestal_state: Cross-step pedestal state.
      explicit_pedestal_structure: Optional explicit structure from pre_step.

    Returns:
      Tuple of (AdaptiveTransportPedestalModelOutput,
      updated AdaptiveTransportPedestalState).
    """
    assert isinstance(
        pedestal_state, pedestal_state_lib.AdaptiveTransportPedestalState
    )
    if explicit_pedestal_structure is not None:
      structure = explicit_pedestal_structure
      structure_state = pedestal_state.structure
    else:
      structure, structure_state = self.structure_model(
          runtime_params, geo, core_profiles, pedestal_state.structure
      )
    formation_state = self.formation_model.update_state(
        runtime_params, geo, core_profiles, pedestal_state.formation
    )
    transport_multipliers = self._compute_transport_multipliers(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_state.confinement_mode,
        structure,
        formation_state,
    )
    updated_state = pedestal_state_lib.AdaptiveTransportPedestalState(
        structure=structure_state,
        formation=formation_state,
        confinement_mode=pedestal_state.confinement_mode,
    )
    output = pedestal_model_output.AdaptiveTransportPedestalModelOutput(
        structure=structure,
        confinement_mode=updated_state.confinement_mode,
        transport_multipliers=transport_multipliers,
    )
    return output, updated_state


def _update_ibc_pedestal_state(
    pedestal_state: pedestal_state_lib.IBCPedestalState,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    trigger_l_to_h: array_typing.BoolScalar,
    trigger_h_to_l: array_typing.BoolScalar,
) -> pedestal_state_lib.IBCPedestalState:
  """Updates IBC pedestal state machine, timers, and L-mode baselines.

  Args:
    pedestal_state: Current IBC pedestal state.
    runtime_params: Runtime parameters at time t.
    geo: Torus geometry.
    core_profiles: Core plasma profiles.
    trigger_l_to_h: Boolean condition to trigger L->H transition.
    trigger_h_to_l: Boolean condition to trigger H->L back-transition.

  Returns:
    Updated IBCPedestalState.
  """
  ConfinementMode = pedestal_runtime_params_lib.ConfinementMode
  old_confinement_mode = pedestal_state.confinement_mode

  elapsed_transition_time = (
      runtime_params.t - pedestal_state.transition_start_time
  )
  transition_is_complete = (
      elapsed_transition_time >= runtime_params.pedestal.transition_time_width
  )
  instant_transition = runtime_params.pedestal.transition_time_width == 0.0
  l_to_h_transition_target_mode = jnp.where(
      instant_transition,
      ConfinementMode.H_MODE,
      ConfinementMode.TRANSITIONING_TO_H_MODE,
  )
  h_to_l_transition_target_mode = jnp.where(
      instant_transition,
      ConfinementMode.L_MODE,
      ConfinementMode.TRANSITIONING_TO_L_MODE,
  )

  conditions = [
      # Completed L-H transition, unless we meet the threshold for an H-L
      # back-transition.
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
      & transition_is_complete
      & ~trigger_h_to_l,
      # Completed H-L back-transition, unless we meet the threshold for an L-H
      # transition.
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
      & transition_is_complete
      & ~trigger_l_to_h,
      # L-H transition.
      (old_confinement_mode != ConfinementMode.H_MODE) & trigger_l_to_h,
      # H-L back-transition, with hysteresis.
      (old_confinement_mode != ConfinementMode.L_MODE) & trigger_h_to_l,
  ]
  new_confinement_modes = [
      ConfinementMode.H_MODE,
      ConfinementMode.L_MODE,
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
      (old_confinement_mode == ConfinementMode.L_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
  ) | (
      (old_confinement_mode == ConfinementMode.H_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
  )
  dithering_transition = (
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
  ) | (
      (old_confinement_mode == ConfinementMode.TRANSITIONING_TO_L_MODE)
      & (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
  )
  # At time t, suppose we have been in forward transition since t0, with
  # desired transition duration w. If we now begin a back-transition, we have
  # been in forward transition for min(t - t0, w), and so we want the
  # back-transition to take the same duration min(t - t0, w). As the
  # transition time width is w, we set the start time to
  # min((2t - t0) - w, t).
  new_transition_start_time = jnp.select(
      [standard_transition, dithering_transition],
      [
          runtime_params.t,
          jnp.minimum(
              2.0 * runtime_params.t
              - pedestal_state.transition_start_time
              - runtime_params.pedestal.transition_time_width,
              runtime_params.t,
          ),
      ],
      default=pedestal_state.transition_start_time,
  )

  update_L_mode_values = (old_confinement_mode == ConfinementMode.L_MODE) & (
      (new_confinement_mode == ConfinementMode.TRANSITIONING_TO_H_MODE)
      | (new_confinement_mode == ConfinementMode.H_MODE)
  )
  ped_top_idx = jnp.argmin(
      jnp.abs(geo.rho_norm - pedestal_state.rho_norm_ped_top)
  )
  new_T_i_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.kinetic.T_i.value[ped_top_idx],
      pedestal_state.T_i_ped_L_mode,
  )
  new_T_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.kinetic.T_e.value[ped_top_idx],
      pedestal_state.T_e_ped_L_mode,
  )
  new_n_e_ped_L_mode = jnp.where(
      update_L_mode_values,
      core_profiles.kinetic.n_e.value[ped_top_idx],
      pedestal_state.n_e_ped_L_mode,
  )

  return dataclasses.replace(
      pedestal_state,
      confinement_mode=new_confinement_mode,
      transition_start_time=new_transition_start_time,
      T_i_ped_L_mode=new_T_i_ped_L_mode,
      T_e_ped_L_mode=new_T_e_ped_L_mode,
      n_e_ped_L_mode=new_n_e_ped_L_mode,
  )
