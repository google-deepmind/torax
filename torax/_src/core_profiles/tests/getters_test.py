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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src.config import numerics as numerics_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import runtime_params as core_profile_runtime_params
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.torax_pydantic import torax_pydantic

SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class GettersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    # TODO(b/512741582): Remove cache clearing after finding underlying issue.
    jax.clear_caches()
    self.geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()

  def test_updated_ion_temperature(self):
    bound = np.array(42.0)
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        core_profile_runtime_params.RuntimeParams,
        instance=True,
        T_i_right_bc=bound,
        T_i=value,
    )
    result = getters.get_updated_ion_temperature(profile_conditions, self.geo)
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  def test_only_updating_boundary_condition(self):
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        core_profile_runtime_params.RuntimeParams,
        instance=True,
        T_i_right_bc=0.5,
        T_i=value,
    )
    original_T_i = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 3.0, 4.0]),
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        face_centers=self.geo.rho_face_norm,
    )
    result = getters.get_updated_ion_temperature(
        profile_conditions,
        self.geo,
        only_boundary_condition=True,
        original_T_i_value=original_T_i,
    )
    np.testing.assert_allclose(result.value, original_T_i.value)
    np.testing.assert_equal(result.right_face_constraint, 0.5)

  def test_updated_electron_temperature(self):
    bound = np.array(42.0)
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        core_profile_runtime_params.RuntimeParams,
        instance=True,
        T_e_right_bc=bound,
        T_e=value,
    )
    result = getters.get_updated_electron_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  def test_only_updating_boundary_condition_electron_temperature(self):
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        core_profile_runtime_params.RuntimeParams,
        instance=True,
        T_e_right_bc=0.5,
        T_e=value,
    )
    original_T_e = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 3.0, 4.0]),
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        face_centers=self.geo.rho_face_norm,
    )
    result = getters.get_updated_electron_temperature(
        profile_conditions,
        self.geo,
        only_boundary_condition=True,
        original_T_e_value=original_T_e,
    )
    np.testing.assert_allclose(result.value, original_T_e.value)
    np.testing.assert_equal(result.right_face_constraint, 0.5)

  def test_n_e_core_profile_setter(self):
    """Tests that setting n_e works."""
    expected_value = np.array([1.4375e20, 1.3125e20, 1.1875e20, 1.0625e20])
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict(
        {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': {0: {0: 1.5e20, 1: 1e20}},
            'n_e_nbar_is_fGW': False,
            'n_e_right_bc_is_fGW': False,
            'nbar': 1e20,
            'normalize_n_e_to_nbar': False,
        },
    )
    numerics = numerics_lib.NumericsConfig.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    n_e = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )
    np.testing.assert_allclose(
        n_e.value,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_only_updating_boundary_condition_n_e(self):
    """Tests that only updating the boundary condition works."""
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        core_profile_runtime_params.RuntimeParams,
        instance=True,
        n_e_right_bc=0.5,
        n_e=value,
        I_p=15e6,
        n_e_right_bc_is_fGW=False,
        n_e_nbar_is_fGW=False,
        normalize_n_e_to_nbar=False,
    )
    original_n_e = cell_variable.CellVariable(
        value=jnp.array([1.0, 2.0, 3.0, 4.0]),
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        face_centers=self.geo.rho_face_norm,
    )
    n_e = getters.get_updated_electron_density(
        profile_conditions,
        self.geo,
        only_boundary_condition=True,
        original_n_e_value=original_n_e,
    )
    np.testing.assert_allclose(n_e.value, original_n_e.value)
    np.testing.assert_allclose(n_e.right_face_constraint, 0.5)  # pyrefly: ignore[no-matching-overload]

  @parameterized.named_parameters(
      dict(
          testcase_name='Set from n_e',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=1.0e20,
      ),
      dict(
          testcase_name='Set and normalize from n_e',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.8050314e20,
      ),
      dict(
          testcase_name='Set and normalize from n_e in fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=0.9609356e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc',
          n_e_right_bc=0.85e20,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.85e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc absolute, ignore normalize',
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
      dict(
          testcase_name='Set from n_e in fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=1.19366207319e20,
      ),
      dict(
          testcase_name='Set from n_e, ignore n_e_right_bc_is_fGW',
          n_e_right_bc=None,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=1.0e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc, which is in fGW',
          n_e_right_bc=0.5,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=False,
          n_e_right_bc_is_fGW=True,
          expected_n_e_right_bc=0.59683103659e20,
      ),
      dict(
          testcase_name='Set from n_e_right_bc, ignore n_e_nbar_is_fGW',
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=False,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
      dict(
          testcase_name=(
              'Set from n_e_right_bc, ignore n_e_nbar_is_fGW, ignore normalize'
          ),
          n_e_right_bc=0.5e20,
          normalize_n_e_to_nbar=True,
          n_e_nbar_is_fGW=True,
          n_e_right_bc_is_fGW=False,
          expected_n_e_right_bc=0.5e20,
      ),
  )
  def test_density_boundary_condition_override(
      self,
      n_e_right_bc: float | None,
      normalize_n_e_to_nbar: bool,
      n_e_nbar_is_fGW: bool,
      n_e_right_bc_is_fGW: bool,
      expected_n_e_right_bc: float,
  ):
    """Tests that setting n_e right boundary works."""
    if n_e_nbar_is_fGW:
      nbar = 1.0
      n_e = {0: {0: 1.5, 1: 1}}
    else:
      nbar = 1.0e20
      n_e = {0: {0: 1.5e20, 1: 1e20}}

    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'n_e': n_e,
        'n_e_nbar_is_fGW': n_e_nbar_is_fGW,
        'n_e_right_bc_is_fGW': n_e_right_bc_is_fGW,
        'nbar': nbar,
        'n_e_right_bc': n_e_right_bc,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
    })
    numerics = numerics_lib.NumericsConfig.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    n_e = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        n_e.right_face_constraint,
        expected_n_e_right_bc,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_n_e_core_profile_setter_with_normalization(
      self,
  ):
    """Tests that normalizing vs. not by nbar gives consistent results."""
    nbar = 1e20
    numerics = numerics_lib.NumericsConfig.from_dict({})
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'n_e': {0: {0: 1.5e20, 1: 1e20}},
        'n_e_nbar_is_fGW': False,
        'nbar': nbar,
        'normalize_n_e_to_nbar': True,
        'n_e_right_bc': 0.5e20,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    n_e_normalized = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )

    np.testing.assert_allclose(np.mean(n_e_normalized.value), nbar, rtol=1e-1)

    profile_conditions._update_fields({'normalize_n_e_to_nbar': False})
    n_e_unnormalized = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )

    ratio = n_e_unnormalized.value / n_e_normalized.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  @parameterized.parameters(
      True,
      False,
  )
  def test_n_e_core_profile_setter_with_fGW(
      self,
      normalize_n_e_to_nbar: bool,
  ):
    """Tests setting the Greenwald fraction vs. not gives consistent results."""
    numerics = numerics_lib.NumericsConfig.from_dict({})
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_nbar_is_fGW': True,
        'nbar': 1.0,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
        'n_e_right_bc': 0.5e20,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    n_e_fGW = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )
    profile_conditions._update_fields(
        {
            'n_e_nbar_is_fGW': False,
            'n_e': {0: {0: 1.5e20, 1: 1e20}},
            'nbar': 1e20,
        },
    )
    # Need to reset n_e grid after private _update_fields. Otherwise, the grid
    # is None. Cannot use public ToraxConfig.update_fields since
    # profile_conditions is not a ToraxConfig.
    torax_pydantic.set_grid(
        profile_conditions, self.geo.torax_mesh, mode='relaxed'
    )

    n_e = getters.get_updated_electron_density(
        profile_conditions.build_runtime_params(1.0),
        self.geo,
    )

    ratio = n_e.value / n_e_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)


if __name__ == '__main__':
  absltest.main()
