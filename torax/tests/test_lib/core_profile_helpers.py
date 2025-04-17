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
from jax import numpy as jnp
import numpy as np
from torax import output
from torax import state
from torax.fvm import cell_variable
from torax.geometry import geometry


def make_zero_core_profiles(
    geo: geometry.Geometry,
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
      temp_ion=zero_cell_variable,
      temp_el=zero_cell_variable,
      psi=zero_cell_variable,
      psidot=zero_cell_variable,
      ne=zero_cell_variable,
      ni=zero_cell_variable,
      nimp=zero_cell_variable,
      q_face=jnp.zeros_like(geo.rho_face),
      s_face=jnp.zeros_like(geo.rho_face),
      nref=jnp.array(0.0),
      vloop_lcfs=jnp.array(0.0),
      Zi=jnp.zeros_like(geo.rho),
      Zi_face=jnp.zeros_like(geo.rho_face),
      Ai=jnp.zeros(()),
      Zimp=jnp.zeros_like(geo.rho),
      Zimp_face=jnp.zeros_like(geo.rho_face),
      Aimp=jnp.zeros(()),
  )


def verify_core_profiles(
    ref_profiles: dict[str, chex.Array],
    index: int,
    core_profiles: state.CoreProfiles,
):
  """Verify core profiles matches a reference at given index."""
  np.testing.assert_allclose(
      core_profiles.temp_el.value, ref_profiles[output.TEMP_EL][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.temp_ion.value, ref_profiles[output.TEMP_ION][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.value, ref_profiles[output.NE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ne.right_face_constraint,
      ref_profiles[output.NE_RIGHT_BC][index],
  )
  np.testing.assert_allclose(
      core_profiles.psi.value, ref_profiles[output.PSI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.psidot.value, ref_profiles[output.PSIDOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.value, ref_profiles[output.NI][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.ni.right_face_constraint,
      ref_profiles[output.NI_RIGHT_BC][index],
  )

  np.testing.assert_allclose(
      core_profiles.q_face, ref_profiles[output.Q_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.s_face, ref_profiles[output.S_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.nref, ref_profiles[output.NREF][index]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap,
      ref_profiles[output.J_BOOTSTRAP][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot, ref_profiles[output.JTOT][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.jtot_face, ref_profiles[output.JTOT_FACE][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.j_bootstrap_face,
      ref_profiles[output.J_BOOTSTRAP_FACE][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.external_current_source,
      ref_profiles[output.EXTERNAL_CURRENT][index, :],
  )
  np.testing.assert_allclose(
      core_profiles.currents.johm, ref_profiles[output.JOHM][index, :]
  )
  np.testing.assert_allclose(
      core_profiles.currents.I_bootstrap,
      ref_profiles[output.I_BOOTSTRAP][index],
  )
  np.testing.assert_allclose(
      core_profiles.currents.Ip_profile_face,
      ref_profiles[output.IP_PROFILE_FACE][index, :],
  )
