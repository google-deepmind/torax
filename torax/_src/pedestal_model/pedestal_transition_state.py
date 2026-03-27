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
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
import typing_extensions

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalTransitionState:
  """Tracks the state of pedestal L-H and H-L transitions.

  This state is used when the pedestal model is in ADAPTIVE_SOURCE mode
  with `use_formation_model_with_adaptive_source=True`. It persists across
  timesteps to track the current confinement mode and enable smooth transitions.

  Attributes:
    in_H_mode: Whether the plasma is currently in H-mode.
    transition_start_time: The simulation time at which the current transition
      started. Set to -inf when no transition is active.
    T_i_ped_L_mode: Ion temperature at the pedestal top captured at the start of
      the most recent L-mode to H-mode transition [keV].
    T_e_ped_L_mode: Electron temperature at the pedestal top captured at the
      start of the most recent L-mode to H-mode transition [keV].
    n_e_ped_L_mode: Electron density at the pedestal top captured at the start
      of the most recent L-mode to H-mode transition [m^-3].
  """

  in_H_mode: array_typing.BoolScalar
  transition_start_time: array_typing.FloatScalar
  # TODO(b/496703290) provide a way for these to be initialized in config, to
  # avoid edge case where we start in H-mode and have no good L-mode values.
  T_i_ped_L_mode: array_typing.FloatScalar
  T_e_ped_L_mode: array_typing.FloatScalar
  n_e_ped_L_mode: array_typing.FloatScalar

  @classmethod
  def initial_state(cls) -> typing_extensions.Self:
    """Creates an initial transition state starting in L-mode."""
    dtype = jax_utils.get_dtype()
    return cls(
        in_H_mode=jnp.bool_(False),
        transition_start_time=jnp.array(-jnp.inf, dtype=dtype),
        T_i_ped_L_mode=jnp.array(0.0, dtype=dtype),
        T_e_ped_L_mode=jnp.array(0.0, dtype=dtype),
        n_e_ped_L_mode=jnp.array(0.0, dtype=dtype),
    )
