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
    rho_norm_ped_top: Location of the pedestal top in normalized rho. Persisted
      across timesteps so that models which compute rho_norm_ped_top dynamically
      (e.g. EPEDNN) can propagate it to the next pre_step for L-mode baseline
      extraction.
  """

  in_H_mode: array_typing.BoolScalar
  transition_start_time: array_typing.FloatScalar
  # TODO(b/496703290) provide a way for these to be initialized in config, to
  # avoid edge case where we start in H-mode and have no good L-mode values.
  T_i_ped_L_mode: array_typing.FloatScalar
  T_e_ped_L_mode: array_typing.FloatScalar
  n_e_ped_L_mode: array_typing.FloatScalar
  rho_norm_ped_top: array_typing.FloatScalar

  @classmethod
  def initial_state(
      cls,
      T_i_ped: float = 0.0,
      T_e_ped: float = 0.0,
      n_e_ped: float = 0.0,
      rho_norm_ped_top: float = 0.9,
  ) -> typing_extensions.Self:
    """Creates an initial transition state starting in L-mode.

    Args:
      T_i_ped: Ion temperature at the pedestal top from initial conditions
        [keV].
      T_e_ped: Electron temperature at the pedestal top from initial conditions
        [keV].
      n_e_ped: Electron density at the pedestal top from initial conditions
        [m^-3].
      rho_norm_ped_top: Initial pedestal top location in normalized rho.

    Returns:
      A PedestalTransitionState initialized in L-mode with the given pedestal
      top values as L-mode baselines.
    """
    dtype = jax_utils.get_dtype()
    return cls(
        in_H_mode=jnp.bool_(False),
        transition_start_time=jnp.array(-jnp.inf, dtype=dtype),
        T_i_ped_L_mode=jnp.array(T_i_ped, dtype=dtype),
        T_e_ped_L_mode=jnp.array(T_e_ped, dtype=dtype),
        n_e_ped_L_mode=jnp.array(n_e_ped, dtype=dtype),
        rho_norm_ped_top=jnp.array(rho_norm_ped_top, dtype=dtype),
    )
