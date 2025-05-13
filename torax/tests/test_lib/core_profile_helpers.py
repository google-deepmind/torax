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
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax import state
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.output_tools import output


# pylint: disable=invalid-name
def make_zero_core_profiles(
    geo: geometry.Geometry,
    T_e: cell_variable.CellVariable | None = None,
    Z_impurity: jax.Array | None = None,
    Z_impurity_face: jax.Array | None = None,
) -> state.CoreProfiles:
  """Returns a dummy CoreProfiles object."""
  zero_cell_variable = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho),
      dr=geo.drho_norm,
      right_face_constraint=jnp.ones(()),
      right_face_grad_constraint=None,
  )
  return state.CoreProfiles(
      currents=state.Currents.zeros(geo),
      T_i=zero_cell_variable,
      T_e=T_e if T_e is not None else zero_cell_variable,
      psi=zero_cell_variable,
      psidot=zero_cell_variable,
      n_e=zero_cell_variable,
      n_i=zero_cell_variable,
      n_impurity=zero_cell_variable,
      q_face=jnp.zeros_like(geo.rho_face),
      s_face=jnp.zeros_like(geo.rho_face),
      density_reference=jnp.array(0.0),
      vloop_lcfs=jnp.array(0.0),
      Z_i=jnp.zeros_like(geo.rho),
      Z_i_face=jnp.zeros_like(geo.rho_face),
      A_i=jnp.zeros(()),
      Z_impurity=Z_impurity
      if Z_impurity is not None
      else jnp.zeros_like(geo.rho),
      Z_impurity_face=Z_impurity_face
      if Z_impurity_face is not None
      else jnp.zeros_like(geo.rho_face),
      A_impurity=jnp.zeros(()),
  )


def verify_core_profiles(
    ref_profiles: dict[str, chex.Array],
    index: int,
    core_profiles: state.CoreProfiles,
):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.T_e.value,
      ref_profiles[output.TEMPERATURE_ELECTRON][index, 1:-1],
  )
  np.testing.assert_allclose(
      core_profiles.T_i.value,
      ref_profiles[output.TEMPERATURE_ION][index, 1:-1],
  )
  np.testing.assert_allclose(
      core_profiles.n_e.value, ref_profiles[output.N_E][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_e.right_face_constraint,
      ref_profiles[output.N_E][index, -1],
  )
  np.testing.assert_allclose(
      core_profiles.psi.value, ref_profiles[output.PSI][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value, ref_profiles[output.V_LOOP][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_i.value, ref_profiles[output.N_I][index, 1:-1]
  )
  np.testing.assert_allclose(
      core_profiles.n_i.right_face_constraint,
      ref_profiles[output.N_I][index, -1],
  )

  np.testing.assert_allclose(
      core_profiles.q_face, ref_profiles[output.Q][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.s_face, ref_profiles[output.MAGNETIC_SHEAR][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_total_face[0],
      ref_profiles[output.J_TOTAL][index, 0],
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_total_face[-1],
      ref_profiles[output.J_TOTAL][index, -1],
  )
  np.testing.assert_allclose(
      core_profiles.currents.external_current_source,
      ref_profiles[output.J_EXTERNAL][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_ohmic, ref_profiles[output.J_OHMIC][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip_profile_face,
      ref_profiles[output.IP_PROFILE][index, :],
  )
