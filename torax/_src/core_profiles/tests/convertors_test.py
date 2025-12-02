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
import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import numpy as jnp
import numpy as np
from torax._src import state
from torax._src.core_profiles import convertors
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry


# pylint: disable=invalid-name
class ConvertersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()

    T_i = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 1.0,
        dr=self.geo.drho_norm,
        right_face_constraint=jnp.array(0.5),
        right_face_grad_constraint=None,
    )
    T_e = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 2.0,
        dr=self.geo.drho_norm,
        right_face_constraint=jnp.array(0.6),
        right_face_grad_constraint=None,
    )
    psi = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 3.0,
        dr=self.geo.drho_norm,
        right_face_grad_constraint=jnp.array(0.7),
        right_face_constraint=None,
    )
    n_e = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 4.0,
        dr=self.geo.drho_norm,
        right_face_constraint=jnp.array(0.8),
        right_face_grad_constraint=None,
    )

    self.base_core_profiles = state.CoreProfiles(
        T_i=T_i,
        T_e=T_e,
        psi=psi,
        n_e=n_e,
        n_i=mock.ANY,
        n_impurity=mock.ANY,
        impurity_fractions=mock.ANY,
        psidot=mock.ANY,
        q_face=mock.ANY,
        s_face=mock.ANY,
        v_loop_lcfs=mock.ANY,
        Z_i=mock.ANY,
        Z_i_face=mock.ANY,
        Z_eff=mock.ANY,
        Z_eff_face=mock.ANY,
        A_i=mock.ANY,
        Z_impurity=mock.ANY,
        Z_impurity_face=mock.ANY,
        A_impurity=mock.ANY,
        A_impurity_face=mock.ANY,
        sigma=mock.ANY,
        sigma_face=mock.ANY,
        j_total=mock.ANY,
        j_total_face=mock.ANY,
        Ip_profile_face=mock.ANY,
        toroidal_velocity=mock.ANY,
    )

  def test_core_profiles_to_solver_x_tuple(self):
    evolving_names = ('T_e', 'n_e', 'psi')
    solver_x_tuple = convertors.core_profiles_to_solver_x_tuple(
        self.base_core_profiles,
        evolving_names,
    )
    self.assertLen(solver_x_tuple, 3)

    np.testing.assert_array_almost_equal(
        solver_x_tuple[0].value, self.base_core_profiles.T_e.value, decimal=10
    )
    np.testing.assert_array_almost_equal(
        solver_x_tuple[0].right_face_constraint,
        self.base_core_profiles.T_e.right_face_constraint,
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        solver_x_tuple[1].value,
        self.base_core_profiles.n_e.value / convertors.SCALING_FACTORS['n_e'],
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        solver_x_tuple[1].right_face_constraint,
        self.base_core_profiles.n_e.right_face_constraint
        / convertors.SCALING_FACTORS['n_e'],
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        solver_x_tuple[2].value, self.base_core_profiles.psi.value, decimal=10
    )
    np.testing.assert_array_almost_equal(
        solver_x_tuple[2].right_face_grad_constraint,
        self.base_core_profiles.psi.right_face_grad_constraint,
        decimal=10,
    )

  def test_solver_x_tuple_to_core_profiles(self):
    evolving_names = ('T_e', 'n_e', 'psi')

    # Simulate solver output
    x_new_T_e_val = self.base_core_profiles.T_e.value * 1.5
    x_new_n_e_val = self.base_core_profiles.n_e.value * 2.0
    x_new_psi_val = self.base_core_profiles.psi.value * 3.0

    x_new_solver_tuple = (
        dataclasses.replace(
            self.base_core_profiles.T_e,
            value=x_new_T_e_val,
        ),
        dataclasses.replace(
            self.base_core_profiles.n_e,
            value=x_new_n_e_val,
        ),
        dataclasses.replace(
            self.base_core_profiles.psi,
            value=x_new_psi_val,
        ),
    )

    updated_cp = convertors.solver_x_tuple_to_core_profiles(
        x_new_solver_tuple,
        evolving_names,
        self.base_core_profiles,
    )

    np.testing.assert_array_almost_equal(
        updated_cp.T_e.value, x_new_T_e_val, decimal=10
    )
    np.testing.assert_array_almost_equal(
        updated_cp.n_e.value,
        x_new_n_e_val * convertors.SCALING_FACTORS['n_e'],
        decimal=10,
    )
    np.testing.assert_array_almost_equal(
        updated_cp.psi.value, x_new_psi_val, decimal=10
    )

    # Check non-evolving variables are untouched from the template
    np.testing.assert_array_almost_equal(
        updated_cp.T_i.value, self.base_core_profiles.T_i.value, decimal=10
    )

  def test_core_profiles_to_x_round_trip(self):
    evolving_names = ('T_e', 'n_e', 'psi')
    checked_names = ('T_e', 'T_i', 'n_e', 'psi')
    x_solver_tuple = convertors.core_profiles_to_solver_x_tuple(
        self.base_core_profiles,
        evolving_names,
    )
    updated_cp = convertors.solver_x_tuple_to_core_profiles(
        x_solver_tuple,
        evolving_names,
        self.base_core_profiles,
    )
    for name in checked_names:
      chex.assert_trees_all_close(
          getattr(updated_cp, name),
          getattr(self.base_core_profiles, name),
      )


if __name__ == '__main__':
  absltest.main()
