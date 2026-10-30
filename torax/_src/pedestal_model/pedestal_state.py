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

"""State for pedestal models."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.pedestal_model import runtime_params as runtime_params_lib
from torax._src.pedestal_model.structure import pedestal_structure

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StructureState:
  """State for a pedestal structure model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FormationState:
  """State for a pedestal formation model."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalState:
  """State for a pedestal model carried over across timesteps.

  Attributes:
    structure: State for the pedestal structure model.
    formation: State for the pedestal formation model.
    confinement_mode: Current discrete confinement mode.
  """

  structure: StructureState = dataclasses.field(default_factory=StructureState)
  formation: FormationState = dataclasses.field(default_factory=FormationState)
  confinement_mode: (
      runtime_params_lib.ConfinementMode | array_typing.IntScalar
  ) = dataclasses.field(
      default_factory=lambda: jnp.asarray(
          runtime_params_lib.ConfinementMode.L_MODE,
          dtype=jax_utils.get_int_dtype(),
      )
  )

  @property
  def is_active(self) -> array_typing.BoolScalar:
    """Returns whether the pedestal is active (i.e. not in L-mode)."""
    return self.confinement_mode != runtime_params_lib.ConfinementMode.L_MODE


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveTransportPedestalState(PedestalState):
  """Cross-step pedestal state for ADAPTIVE_TRANSPORT mode."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class IBCPedestalState(PedestalState):
  """Cross-step pedestal state for INTERNAL_BOUNDARY_CONDITION mode.

  Attributes:
    transition_start_time: Simulation time at which the current L-H or H-L
      transition started (-inf when no transition has occurred; mirrored
      pseudo-start-time when dithering).
    T_i_ped_L_mode: Ion temperature at the pedestal top captured for L-mode
      ramp interpolation [keV].
    T_e_ped_L_mode: Electron temperature at the pedestal top captured for
      L-mode ramp interpolation [keV].
    n_e_ped_L_mode: Electron density at the pedestal top captured for L-mode
      ramp interpolation [m^-3].
  """

  transition_start_time: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(-jnp.inf, dtype=jax_utils.get_dtype())
  )
  T_i_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  T_e_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )
  n_e_ped_L_mode: array_typing.FloatScalar = dataclasses.field(
      default_factory=lambda: jnp.array(0.0, dtype=jax_utils.get_dtype())
  )

  def compute_ramp_fraction(
      self,
      t: array_typing.FloatScalar,
      transition_time_width: array_typing.FloatScalar,
  ) -> array_typing.FloatScalar:
    """Computes the ramp fraction for a pedestal transition.

    Returns a value in [0, 1] representing the progress of the current
    transition. 0 means the transition just started, 1 means it is complete.

    Args:
      t: Current simulation time.
      transition_time_width: Duration of the transition ramp.

    Returns:
      Ramp fraction clipped to [0, 1].
    """
    elapsed = t - self.transition_start_time
    safe_width = jnp.where(
        transition_time_width == 0.0, 1.0, transition_time_width
    )
    fraction = jnp.where(
        transition_time_width == 0.0,
        1.0,
        elapsed / safe_width,
    )
    return jnp.clip(fraction, 0.0, 1.0)

  def scale_structure(
      self,
      structure: pedestal_structure.PedestalStructure,
      t: array_typing.FloatScalar,
      transition_time_width: array_typing.FloatScalar,
  ) -> pedestal_structure.PedestalStructure:
    """Applies ramp scaling to pedestal structure during transitions.

    During an L-H transition, linearly ramps from L-mode baseline values to the
    H-mode targets. During an H-L transition, ramps from the H-mode targets back
    to the L-mode baselines.

    Args:
      structure: Pedestal structure containing target values at pedestal top.
      t: Current simulation time.
      transition_time_width: Duration of the transition ramp.

    Returns:
      Scaled pedestal structure.
    """
    ramp_fraction = self.compute_ramp_fraction(
        t=t, transition_time_width=transition_time_width
    )

    def _interpolate_transition(l_val, h_val):
      l_to_h_ramp = l_val + ramp_fraction * (h_val - l_val)
      h_to_l_ramp = h_val + ramp_fraction * (l_val - h_val)
      confinement_mode = self.confinement_mode
      return jnp.select(
          [
              confinement_mode == runtime_params_lib.ConfinementMode.L_MODE,
              confinement_mode == runtime_params_lib.ConfinementMode.H_MODE,
              confinement_mode
              == runtime_params_lib.ConfinementMode.TRANSITIONING_TO_H_MODE,
              confinement_mode
              == runtime_params_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
          ],
          [l_val, h_val, l_to_h_ramp, h_to_l_ramp],
      )

    scaled_T_i = _interpolate_transition(
        l_val=self.T_i_ped_L_mode,
        h_val=structure.T_i_ped,
    )
    scaled_T_e = _interpolate_transition(
        l_val=self.T_e_ped_L_mode,
        h_val=structure.T_e_ped,
    )
    scaled_n_e = _interpolate_transition(
        l_val=self.n_e_ped_L_mode,
        h_val=structure.n_e_ped,
    )

    return dataclasses.replace(
        structure,
        T_i_ped=scaled_T_i,
        T_e_ped=scaled_T_e,
        n_e_ped=scaled_n_e,
    )
