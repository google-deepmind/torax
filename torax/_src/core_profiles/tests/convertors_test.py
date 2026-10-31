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
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=jnp.array(0.5),
        right_face_grad_constraint=None,
    )
    T_e = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 2.0,
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=jnp.array(0.6),
        right_face_grad_constraint=None,
    )
    psi = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 3.0,
        face_centers=self.geo.rho_face_norm,
        right_face_grad_constraint=jnp.array(0.7),
        right_face_constraint=None,
    )
    n_e = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 4.0,
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=jnp.array(0.8),
        right_face_grad_constraint=None,
    )
    toroidal_angular_velocity = cell_variable.CellVariable(
        value=jnp.ones(self.geo.rho_norm.shape) * 5.0,
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=jnp.array(0.9),
        right_face_grad_constraint=None,
    )

    self.base_core_profiles = state.CoreProfiles(
        kinetic=state.KineticProfiles(
            T_i=T_i,
            T_e=T_e,
            n_e=n_e,
            n_i=mock.ANY,
            n_impurity=mock.ANY,
        ),
        ions=state.IonComposition(
            impurity_fractions=mock.ANY,
            main_ion_fractions=mock.ANY,
            Z_i=mock.ANY,
            Z_i_face=mock.ANY,
            A_i=mock.ANY,
            Z_impurity=mock.ANY,
            Z_impurity_face=mock.ANY,
            A_impurity=mock.ANY,
            A_impurity_face=mock.ANY,
            Z_eff=mock.ANY,
            Z_eff_face=mock.ANY,
            charge_state_info=mock.ANY,
            charge_state_info_face=mock.ANY,
        ),
        flux=state.FluxProfiles(
            psi=psi,
            q_face=mock.ANY,
            s_face=mock.ANY,
            j_toroidal_total=mock.ANY,
            j_toroidal_total_face=mock.ANY,
            I_p_profile_face=mock.ANY,
            v_loop_lcfs=mock.ANY,
        ),
        toroidal_angular_velocity=toroidal_angular_velocity,
        fast_ions=mock.ANY,
        conductivity=state.Conductivity(
            sigma=mock.ANY,
            sigma_face=mock.ANY,
        ),
        poloidal_velocity=mock.ANY,
        psidot=mock.ANY,
    )
    self.base_solver_state = (
        state.SolverState.from_core_profiles(self.base_core_profiles)
    )

  def test_solver_state_to_x_round_trip(self):
    evolving_names = ('T_i', 'T_e', 'psi', 'n_e')
    x_solver_tuple = convertors.solver_state_to_solver_x_tuple(
        self.base_solver_state,
        evolving_names,
    )
    self.assertLen(x_solver_tuple, len(evolving_names))

    for name, cv in zip(evolving_names, x_solver_tuple):
      base_cv = getattr(self.base_solver_state, name)
      scaling = convertors.STATE_SCALING_FACTORS[name]
      np.testing.assert_allclose(cv.value, base_cv.value / scaling)
      if base_cv.right_face_constraint is not None:
        assert cv.right_face_constraint is not None
        np.testing.assert_allclose(
            cv.right_face_constraint,
            base_cv.right_face_constraint / scaling,
        )
      if base_cv.right_face_grad_constraint is not None:
        assert cv.right_face_grad_constraint is not None
        np.testing.assert_allclose(
            cv.right_face_grad_constraint,
            base_cv.right_face_grad_constraint / scaling,
        )

    result = convertors.solver_x_tuple_to_solver_state(
        x_solver_tuple,
        evolving_names,
        self.base_solver_state,
    )
    for name in (
        'T_i',
        'T_e',
        'psi',
        'n_e',
        'toroidal_angular_velocity',
    ):
      chex.assert_trees_all_close(
          getattr(result, name),
          getattr(self.base_solver_state, name),
      )

  @parameterized.parameters(
      (('T_e',),),
      (('T_i', 'T_e', 'n_e'),),
      (('T_i', 'T_e', 'psi', 'n_e'),),
  )
  def test_solver_x_tuple_to_solver_state_updates_evolving(
      self, evolving_names: tuple[str, ...]
  ):
    x_new_solver_tuple = tuple(
        dataclasses.replace(
            getattr(self.base_solver_state, name),
            value=getattr(self.base_solver_state, name).value
            / convertors.STATE_SCALING_FACTORS[name]
            * 2.5,
        )
        for name in evolving_names
    )
    result = convertors.solver_x_tuple_to_solver_state(
        x_new_solver_tuple,
        evolving_names,
        self.base_solver_state,
    )
    for name in (
        'T_i',
        'T_e',
        'psi',
        'n_e',
        'toroidal_angular_velocity',
    ):
      base_cv = getattr(self.base_solver_state, name)
      updated_cv = getattr(result, name)
      if name in evolving_names:
        np.testing.assert_allclose(updated_cv.value, base_cv.value * 2.5)
      else:
        chex.assert_trees_all_close(updated_cv, base_cv)

  @parameterized.parameters('T_i', 'T_e', 'n_e', 'psi')
  def test_compute_channel_residual_scale(self, channel: str):
    with self.subTest('scale_high'):
      x_high = jnp.array([2.0, 4.0])
      scale_high = convertors._compute_channel_residual_scale(channel, x_high)
      # psi uses span (max - min = 2.0); temperatures and density use mean (3.0)
      expected_high = 2.0 if channel == 'psi' else 3.0
      np.testing.assert_allclose(scale_high, expected_high)

    with self.subTest('scale_low'):
      x_low = jnp.array([0.001, 0.002])
      scale_low = convertors._compute_channel_residual_scale(channel, x_low)
      np.testing.assert_allclose(
          scale_low, convertors.RESIDUAL_SCALE_FLOORS[channel]
      )

  def test_compute_channel_residual_scale_unknown_channel(self):
    x_unknown = jnp.array([0.05, 0.05])
    with self.assertRaises(KeyError):
      convertors._compute_channel_residual_scale('unknown', x_unknown)

  def test_compute_residual_scaling_vector(self):
    evolving_names = ('T_e', 'n_e', 'psi')
    n_cells = len(self.base_solver_state.T_e.value)
    psi = dataclasses.replace(
        self.base_solver_state.psi,
        value=jnp.linspace(0.0, 5.0, n_cells),
    )
    x_old = (
        self.base_solver_state.T_e,
        self.base_solver_state.n_e,
        psi,
    )
    scales = convertors.compute_residual_scaling_vector(evolving_names, x_old)

    self.assertEqual(scales.shape, (len(evolving_names) * n_cells,))

    # Precomputed expected scales:
    # T_e: mean(ones * 2.0) = 2.0
    # n_e: mean(ones * 4.0) = 4.0
    # psi: max(linspace(0, 5)) - min(linspace(0, 5)) = 5.0
    expected = jnp.concatenate([
        jnp.full(n_cells, 2.0),
        jnp.full(n_cells, 4.0),
        jnp.full(n_cells, 5.0),
    ])
    np.testing.assert_allclose(scales, expected)


if __name__ == '__main__':
  absltest.main()
