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

"""Builders for core profiles."""

from jax import numpy as jnp
from torax._src import array_typing
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import getters
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import psi_calculations


def _build_flux_profiles(
    geo: geometry.Geometry,
    psi: cell_variable.CellVariable,
    min_rho_norm: array_typing.FloatScalar,
    v_loop_lcfs: array_typing.FloatScalar | None,
) -> state.FluxProfiles:
  """Constructs FluxProfiles derived from poloidal flux and geometry."""
  j_toroidal_total, j_toroidal_total_face, i_p_profile_face = (
      psi_calculations.calc_j_toroidal_total(
          geo,
          psi,
          min_rho_norm,
      )
  )
  if v_loop_lcfs is None:
    v_loop_lcfs = jnp.asarray(0.0, dtype=jax_utils.get_dtype())
  return state.FluxProfiles(
      psi=psi,
      q_face=psi_calculations.calc_q_face(geo, psi),
      s_face=psi_calculations.calc_s_face(geo, psi),
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
      I_p_profile_face=i_p_profile_face,
      v_loop_lcfs=v_loop_lcfs,
  )


def build_base_core_profiles(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    solver_state: state.SolverState,
    v_loop_lcfs: array_typing.FloatScalar | None = None,
) -> state.BaseCoreProfiles:
  """Constructs purely algebraic BaseCoreProfiles from SolverState.

  Used by intra-step, post-step, and initialization profile builders.

  Args:
    runtime_params: Runtime parameters.
    geo: Geometry of the tokamak.
    solver_state: The current set of primary profiles.
    v_loop_lcfs: Loop voltage at the LCFS [V]. If None, defaults to 0.0.

  Returns:
    BaseCoreProfiles with kinetic, ions, flux, and toroidal angular velocity.
  """
  ion_composition, n_i, n_impurity = getters.get_updated_ions(
      runtime_params,
      geo,
      solver_state.n_e,
      solver_state.T_e,
  )
  flux = _build_flux_profiles(
      geo,
      solver_state.psi,
      runtime_params.numerics.min_rho_norm,
      v_loop_lcfs=v_loop_lcfs,
  )
  kinetic = state.KineticProfiles(
      T_i=solver_state.T_i,
      T_e=solver_state.T_e,
      n_e=solver_state.n_e,
      n_i=n_i,
      n_impurity=n_impurity,
  )
  return state.BaseCoreProfiles(
      kinetic=kinetic,
      ions=ion_composition,
      flux=flux,
      toroidal_angular_velocity=solver_state.toroidal_angular_velocity,
  )
