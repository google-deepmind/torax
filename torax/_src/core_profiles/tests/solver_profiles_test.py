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

"""Tests verifying construction, validation, and slicing of SolverProfiles."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import jax_utils
from torax._src.core_profiles import solver_profiles
from torax._src.core_profiles import solver_state
from torax._src.fvm import cell_variable
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.physics import psi_calculations
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class SolverProfilesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    self.core_profiles_t0, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(self.torax_config)
    )
    self.solver_state = self.core_profiles_t0.to_solver_state()

  def test_from_solver_state_structure_and_values(self):
    result = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )

    self.assertIsInstance(result, solver_profiles.SolverProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )
    np.testing.assert_allclose(
        result.kinetic.T_e.value,
        self.solver_state.T_e.value,
    )
    np.testing.assert_allclose(
        result.kinetic.n_e.value,
        self.solver_state.n_e.value,
    )
    np.testing.assert_allclose(
        result.poloidal_flux.psi.value,
        self.solver_state.psi.value,
    )
    np.testing.assert_allclose(
        result.kinetic.toroidal_angular_velocity.value,
        self.solver_state.toroidal_angular_velocity.value,
    )
    np.testing.assert_allclose(
        result.poloidal_flux.v_loop_lcfs,
        jnp.asarray(0.0, dtype=jax_utils.get_dtype()),
    )

    expected_q_face = psi_calculations.calc_q_face(
        self.geo, self.solver_state.psi
    )
    np.testing.assert_allclose(result.poloidal_flux.q_face, expected_q_face)
    self.assertTrue(result.quasineutrality_satisfied())

  def test_from_solver_state_custom_v_loop_lcfs(self):
    custom_v_loop = jnp.asarray(1.75, dtype=jax_utils.get_dtype())
    result = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(result.poloidal_flux.v_loop_lcfs, custom_v_loop)

  def test_from_solver_state_jittable(self):
    @jax.jit
    def jitted_builder(state):
      return solver_profiles.SolverProfiles.from_solver_state(
          runtime_params=self.runtime_params,
          geo=self.geo,
          solver_state=state,
      )

    result = jitted_builder(self.solver_state)
    self.assertIsInstance(result, solver_profiles.SolverProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )

  def test_to_solver_state(self):
    profiles = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )
    result = profiles.to_solver_state()
    self.assertIsInstance(result, solver_state.SolverState)
    np.testing.assert_allclose(result.T_i.value, self.solver_state.T_i.value)
    np.testing.assert_allclose(result.T_e.value, self.solver_state.T_e.value)
    np.testing.assert_allclose(result.psi.value, self.solver_state.psi.value)
    np.testing.assert_allclose(result.n_e.value, self.solver_state.n_e.value)
    np.testing.assert_allclose(
        result.toroidal_angular_velocity.value,
        self.solver_state.toroidal_angular_velocity.value,
    )

  def test_thermal_calculations_empty_fast_ions(self):
    profiles = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )
    n_imp = profiles.calculate_n_impurity_thermal(())
    np.testing.assert_allclose(n_imp.value, profiles.kinetic.n_impurity.value)

    n_i_face = profiles.calculate_n_i_total_thermal_face(())
    expected_n_i_face = (
        profiles.kinetic.n_i.face_value()
        + profiles.kinetic.n_impurity.face_value()
        * profiles.impurity_density_scaling_face
    )
    np.testing.assert_allclose(n_i_face, expected_n_i_face)

    p_i = profiles.calculate_pressure_thermal_i(())
    expected_p_i_val = (
        profiles.kinetic.T_i.value
        * constants.CONSTANTS.keV_to_J
        * (profiles.kinetic.n_i.value + profiles.kinetic.n_impurity.value)
    )
    np.testing.assert_allclose(p_i.value, expected_p_i_val)

  def test_thermal_calculations_with_fast_ions(self):
    profiles = solver_profiles.SolverProfiles.from_solver_state(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )
    fast_ion_n = cell_variable.CellVariable(
        value=0.1 * profiles.kinetic.n_impurity.value,
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=0.1 * profiles.kinetic.n_impurity.get_right_bc(),
        right_face_grad_constraint=None,
    )
    fast_ion_temp = cell_variable.CellVariable(
        value=100.0 * profiles.kinetic.T_i.value,
        face_centers=self.geo.rho_face_norm,
        right_face_constraint=100.0 * profiles.kinetic.T_i.get_right_bc(),
        right_face_grad_constraint=None,
    )
    fast_ion = fast_ion_lib.FastIon(
        species='He3',
        source='ICRH',
        n=fast_ion_n,
        T=fast_ion_temp,
    )
    impurity_fractions = dict(profiles.ions.impurity_fractions)
    impurity_fractions['He3'] = jnp.ones_like(profiles.kinetic.n_impurity.value)
    profiles = dataclasses.replace(
        profiles,
        ions=dataclasses.replace(
            profiles.ions,
            impurity_fractions=impurity_fractions,
        ),
    )

    n_imp = profiles.calculate_n_impurity_thermal((fast_ion,))
    np.testing.assert_allclose(
        n_imp.value, 0.9 * profiles.kinetic.n_impurity.value
    )
    np.testing.assert_allclose(
        n_imp.get_right_bc(), 0.9 * profiles.kinetic.n_impurity.get_right_bc()
    )

    n_i_face = profiles.calculate_n_i_total_thermal_face((fast_ion,))
    expected_n_i_face = (
        profiles.kinetic.n_i.face_value()
        + n_imp.face_value() * profiles.impurity_density_scaling_face
    )
    np.testing.assert_allclose(n_i_face, expected_n_i_face)

    p_i = profiles.calculate_pressure_thermal_i((fast_ion,))
    expected_p_i_val = (
        profiles.kinetic.T_i.value
        * constants.CONSTANTS.keV_to_J
        * (profiles.kinetic.n_i.value + n_imp.value)
    )
    np.testing.assert_allclose(p_i.value, expected_p_i_val)


if __name__ == '__main__':
  absltest.main()
