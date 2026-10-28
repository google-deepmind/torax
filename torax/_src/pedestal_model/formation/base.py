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

"""Base class for pedestal formation models."""

import abc
import dataclasses
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.sources import source_profiles as source_profiles_lib


@dataclasses.dataclass(frozen=True, eq=False)
class FormationModel(static_dataclass.StaticDataclass, abc.ABC):
  """Base class for pedestal formation models."""

  def initial_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_state_lib.FormationState:
    """Returns the initial FormationState for this model."""
    del runtime_params, geo, core_profiles
    return pedestal_state_lib.FormationState()

  def update_state(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      formation_state: pedestal_state_lib.FormationState,
  ) -> pedestal_state_lib.FormationState:
    """Returns the updated FormationState at the end of a timestep.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles.
      formation_state: Pedestal formation state carried over from the previous
        timestep.

    Returns:
      Updated formation state.
    """
    del runtime_params, geo, core_profiles
    return formation_state

  @abc.abstractmethod
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
    """Calculates transport decrease multipliers.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles.
      source_profiles: Source profiles.
      confinement_mode: Current confinement mode (e.g. L-mode or H-mode).
      formation_state: Pedestal formation state carried over from the previous
        timestep.

    Returns:
      Transport decrease multipliers.
    """

  @abc.abstractmethod
  def evaluate_transition_conditions(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      formation_state: pedestal_state_lib.FormationState,
  ) -> tuple[array_typing.BoolScalar, array_typing.BoolScalar]:
    """Evaluates L-H and H-L transition triggers.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry.
      core_profiles: Core profiles.
      source_profiles: Source profiles.
      formation_state: Pedestal formation state carried over from the previous
        timestep.

    Returns:
      A tuple `(trigger_l_to_h_transition, trigger_h_to_l_transition)` where the
      boolean scalars are mutually exclusive indicators of whether the
      conditions to trigger an L->H transition or an H->L back-transition are
      satisfied (`trigger_l_to_h_transition & trigger_h_to_l_transition` must be
      False).
    """

  def initial_confinement_mode(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      formation_state: pedestal_state_lib.FormationState,
  ) -> array_typing.IntScalar:
    """Returns the initial confinement mode at t_initial."""
    # Allow explicit override (e.g. to trigger an L-H or H-L transition ramp
    # from initialization) before falling back to transition conditions.
    if runtime_params.pedestal.formation.initial_confinement_mode is not None:
      return jnp.array(
          runtime_params.pedestal.formation.initial_confinement_mode,
          dtype=jax_utils.get_int_dtype(),
      )
    trigger_l_to_h_transition, _ = self.evaluate_transition_conditions(
        runtime_params, geo, core_profiles, source_profiles, formation_state
    )
    return jnp.where(
        trigger_l_to_h_transition,
        jnp.array(
            pedestal_runtime_params_lib.ConfinementMode.H_MODE,
            dtype=jax_utils.get_int_dtype(),
        ),
        jnp.array(
            pedestal_runtime_params_lib.ConfinementMode.L_MODE,
            dtype=jax_utils.get_int_dtype(),
        ),
    )
