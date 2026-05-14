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

"""State for tracking pedestal L-H and H-L transitions."""

import dataclasses
import enum
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils

# pylint: disable=invalid-name


# Store confinement mode as an int so that it is a valid JAX dynamic type.
class ConfinementMode(enum.IntEnum):
  L_MODE = 0
  H_MODE = 1
  TRANSITIONING_TO_H_MODE = 2
  TRANSITIONING_TO_L_MODE = 3


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalTransitionState:
  """Tracks the state of pedestal L-H and H-L transitions.

  This state is used when the pedestal model is in ADAPTIVE_SOURCE mode
  with `use_formation_model_with_adaptive_source=True`. It persists across
  timesteps to track the current confinement mode and enable smooth transitions.

  Attributes:
    confinement_mode: The current confinement mode.
    transition_start_time: The simulation time at which the current transition
      started. Set to -inf when no transition is active. If dithering occurs,
      transition_start_time is a pseudo-value such that the dither has the
      desired duration.
    T_i_ped_L_mode: Ion temperature at the pedestal top captured at the start of
      the most recent L-mode to H-mode transition [keV].
    T_e_ped_L_mode: Electron temperature at the pedestal top captured at the
      start of the most recent L-mode to H-mode transition [keV].
    n_e_ped_L_mode: Electron density at the pedestal top captured at the start
      of the most recent L-mode to H-mode transition [m^-3].
  """

  confinement_mode: array_typing.IntScalar
  transition_start_time: array_typing.FloatScalar
  # TODO(b/496703290) provide a way for these to be initialized in config, to
  # avoid edge case where we start in H-mode and have no good L-mode values.
  T_i_ped_L_mode: array_typing.FloatScalar
  T_e_ped_L_mode: array_typing.FloatScalar
  n_e_ped_L_mode: array_typing.FloatScalar

  @classmethod
  def empty_L_mode(cls):
    """An L-mode transition state with no stored values.

    These will be overwritten when the first L-mode to H-mode transition begins.
    """
    return cls(
        confinement_mode=jnp.array(
            ConfinementMode.L_MODE, dtype=jax_utils.get_int_dtype()
        ),
        transition_start_time=jnp.array(-jnp.inf, dtype=jax_utils.get_dtype()),
        T_i_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_e_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_ped_L_mode=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )
