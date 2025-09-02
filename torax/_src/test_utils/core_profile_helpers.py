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
"""Helpers for tests using core profiles."""
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.output_tools import output


# pylint: disable=invalid-name
def make_zero_core_profiles(
    geo: geometry.Geometry,
    T_e: cell_variable.CellVariable | None = None,
    Z_impurity: jax.Array | None = None,
    Z_impurity_face: jax.Array | None = None,
    impurity_names: tuple[str, ...] = ("dummy_impurity",),
) -> state.CoreProfiles:
  """Returns a dummy CoreProfiles object."""
  zero_cell_variable = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho),
      dr=geo.drho_norm,
      right_face_constraint=jnp.ones(()),
      right_face_grad_constraint=None,
  )
  impurity_fractions_dict = {
      name: jnp.zeros_like(geo.rho) for name in impurity_names
  }
  return state.CoreProfiles(
      T_i=zero_cell_variable,
      T_e=T_e if T_e is not None else zero_cell_variable,
      psi=zero_cell_variable,
      psidot=zero_cell_variable,
      n_e=zero_cell_variable,
      n_i=zero_cell_variable,
      n_impurity=zero_cell_variable,
      impurity_fractions=impurity_fractions_dict,
      q_face=jnp.zeros_like(geo.rho_face),
      s_face=jnp.zeros_like(geo.rho_face),
      v_loop_lcfs=jnp.array(0.0),
      Z_i=jnp.zeros_like(geo.rho),
      Z_i_face=jnp.zeros_like(geo.rho_face),
      A_i=jnp.zeros(()),
      Z_impurity=Z_impurity
      if Z_impurity is not None
      else jnp.zeros_like(geo.rho),
      Z_impurity_face=Z_impurity_face
      if Z_impurity_face is not None
      else jnp.zeros_like(geo.rho_face),
      A_impurity=jnp.zeros_like(geo.rho),
      A_impurity_face=jnp.zeros_like(geo.rho_face),
      Z_eff=jnp.zeros_like(geo.rho),
      Z_eff_face=jnp.zeros_like(geo.rho_face),
      sigma=jnp.zeros_like(geo.rho),
      sigma_face=jnp.zeros_like(geo.rho_face),
      j_total=jnp.zeros_like(geo.rho),
      j_total_face=jnp.zeros_like(geo.rho_face),
      Ip_profile_face=jnp.zeros_like(geo.rho_face),
  )


def verify_core_profiles(
    ref_profiles: dict[str, array_typing.Array],
    index: int,
    core_profiles: state.CoreProfiles,
):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.T_e.value,
      ref_profiles[output.T_E][index, 1:-1],
      err_msg=(
          f"Mismatch for T_e.value. Core profile: {core_profiles.T_e.value},"
          f" Ref profile ({output.T_E}):"
          f" {ref_profiles[output.T_E][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.T_i.value,
      ref_profiles[output.T_I][index, 1:-1],
      err_msg=(
          f"Mismatch for T_i.value. Core profile: {core_profiles.T_i.value},"
          f" Ref profile ({output.T_I}):"
          f" {ref_profiles[output.T_I][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.n_e.value,
      ref_profiles[output.N_E][index, 1:-1],
      err_msg=(
          f"Mismatch for n_e.value. Core profile: {core_profiles.n_e.value},"
          f" Ref profile ({output.N_E}):"
          f" {ref_profiles[output.N_E][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.n_e.right_face_constraint,
      ref_profiles[output.N_E][index, -1],
      err_msg=(
          "Mismatch for n_e.right_face_constraint. Core profile:"
          f" {core_profiles.n_e.right_face_constraint}, Ref profile"
          f" ({output.N_E} end val): {ref_profiles[output.N_E][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.n_i.value,
      ref_profiles[output.N_I][index, 1:-1],
      err_msg=(
          f"Mismatch for n_i.value. Core profile: {core_profiles.n_i.value},"
          f" Ref profile ({output.N_I}):"
          f" {ref_profiles[output.N_I][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.n_i.right_face_constraint,
      ref_profiles[output.N_I][index, -1],
      err_msg=(
          "Mismatch for n_i.right_face_constraint. Core profile:"
          f" {core_profiles.n_i.right_face_constraint}, Ref profile"
          f" ({output.N_I} end val): {ref_profiles[output.N_I][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.psi.value,
      ref_profiles[output.PSI][index, 1:-1],
      err_msg=(
          f"Mismatch for psi.value. Core profile: {core_profiles.psi.value},"
          f" Ref profile ({output.PSI}):"
          f" {ref_profiles[output.PSI][index, 1:-1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value,
      ref_profiles[output.V_LOOP][index, 1:-1],
      err_msg=(
          "Mismatch for psidot.value. Core profile:"
          f" {core_profiles.psidot.value}, Ref profile ({output.V_LOOP}):"
          f" {ref_profiles[output.V_LOOP][index, 1:-1]}"
      ),
  )

  np.testing.assert_allclose(
      core_profiles.q_face,
      ref_profiles[output.Q][index, :],
      err_msg=(
          f"Mismatch for q_face. Core profile: {core_profiles.q_face},"
          f" Ref profile ({output.Q}): {ref_profiles[output.Q][index, :]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.s_face,
      ref_profiles[output.MAGNETIC_SHEAR][index, :],
      err_msg=(
          f"Mismatch for s_face. Core profile: {core_profiles.s_face},"
          f" Ref profile ({output.MAGNETIC_SHEAR}):"
          f" {ref_profiles[output.MAGNETIC_SHEAR][index, :]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.j_total_face[0],
      ref_profiles[output.J_TOTAL][index, 0],
      err_msg=(
          "Mismatch for j_total_face[0]. Core profile:"
          f" {core_profiles.j_total_face[0]}, Ref profile ({output.J_TOTAL}"
          f" start val): {ref_profiles[output.J_TOTAL][index, 0]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.j_total_face[-1],
      ref_profiles[output.J_TOTAL][index, -1],
      err_msg=(
          "Mismatch for j_total_face[-1]. Core profile:"
          f" {core_profiles.j_total_face[-1]}, Ref profile ({output.J_TOTAL}"
          f" end val): {ref_profiles[output.J_TOTAL][index, -1]}"
      ),
  )
  np.testing.assert_allclose(
      core_profiles.Ip_profile_face,
      ref_profiles[output.IP_PROFILE][index, :],
      err_msg=(
          "Mismatch for Ip_profile_face. Core profile:"
          f" {core_profiles.Ip_profile_face}, Ref profile"
          f" ({output.IP_PROFILE}): {ref_profiles[output.IP_PROFILE][index, :]}"
      ),
  )
