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

"""Unit tests for internal boundary conditions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions
from torax._src.internal_boundary_conditions import no_ibc
from torax._src.internal_boundary_conditions import prescribed
from torax._src.internal_boundary_conditions import pydantic_model
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class PrescribedIBCTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('initial_time', 0.0, np.array([1.0, 0.0, 0.0, 2.0])),
      ('intermediate_time', 0.5, np.array([2.0, 0.0, 0.0, 3.0])),
      ('final_time', 1.0, np.array([3.0, 0.0, 0.0, 4.0])),
      ('after_final_time', 1.5, np.array([3.0, 0.0, 0.0, 4.0])),
  )
  def test_prescribed_ibc_config_build_runtime_params(
      self, t, expected_T_i
  ):
    ibc_config = pydantic_model.PrescribedIBC(
        T_i={
            0.0: {0: 1.0, 1: 2.0},
            1.0: {0: 3.0, 1: 4.0},
        }
    )
    face_centers = interpolated_param_2d.get_face_centers(4)
    grid = interpolated_param_2d.Grid1D(face_centers=face_centers)
    interpolated_param_2d.set_grid(ibc_config, grid=grid)

    runtime_params = ibc_config.build_runtime_params(t)

    np.testing.assert_allclose(runtime_params.T_i, expected_T_i)
    np.testing.assert_allclose(
        runtime_params.T_e,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        runtime_params.n_e,
        np.array([0.0, 0.0, 0.0, 0.0]),
    )

  def test_from_config(self):
    config = {
        'model_name': 'prescribed',
        'T_e': {
            0.0: {
                0.2: 10.0,
                (0.5, 0.9): 20.0,
            }
        },
    }
    ibc = pydantic_model.PrescribedIBC.model_validate(config)

    face_centers = interpolated_param_2d.get_face_centers(4)
    grid = interpolated_param_2d.Grid1D(face_centers=face_centers)
    interpolated_param_2d.set_grid(ibc, grid=grid)

    runtime_params = ibc.build_runtime_params(0.0)

    # Cell centers: [0.125, 0.375, 0.625, 0.875]
    # 0.2 is closest to 0.125 (cell 0). Value is 10.0.
    # (0.5, 0.9) covers cell 2 (0.625) and cell 3 (0.875).
    expected_T_e = np.array([10.0, 0.0, 20.0, 20.0])
    np.testing.assert_allclose(runtime_params.T_e, expected_T_e)

  def test_is_active_returns_false_when_empty(self):
    ibc_empty = pydantic_model.PrescribedIBC()
    self.assertFalse(ibc_empty.is_active())

  def test_is_active_returns_true_when_configured(self):
    ibc_configured = pydantic_model.PrescribedIBC(
        T_e={0.0: {(0.8, 1.0): 10.0}}
    )
    self.assertTrue(ibc_configured.is_active())

  def test_prescribed_ibc_build_runtime_params(self):
    geo = circular_geometry.CircularConfig(n_rho=5).build_geometry()
    ibc = pydantic_model.PrescribedIBC()
    torax_pydantic.set_grid(ibc, geo.torax_mesh)
    runtime_params = ibc.build_runtime_params(0.0)
    self.assertIsInstance(runtime_params, prescribed.RuntimeParams)

  def test_prescribed_ibc_model(self):
    geo = circular_geometry.CircularConfig(n_rho=5).build_geometry()
    ibc_params = prescribed.RuntimeParams(
        T_i=jnp.array([0.0, 5.0, 0.0, 0.0, 0.0]),
        T_e=jnp.array([0.0, 0.0, 0.0, 10.0, 0.0]),
        n_e=jnp.zeros(5),
    )

    class _MockProfileConditions:
      internal_boundary_conditions = ibc_params

    class _MockRuntimeParams:
      profile_conditions = _MockProfileConditions()

    l_mode_state = (
        pedestal_transition_state.PedestalTransitionState.empty_L_mode()
    )
    model = prescribed.PrescribedIBCModel()
    res = model(
        runtime_params=_MockRuntimeParams(),  # pyrefly: ignore[bad-argument-type]
        geo=geo,
        core_profiles=None,  # pyrefly: ignore[bad-argument-type]
        pedestal_transition_state=l_mode_state,
    )
    self.assertIsInstance(
        res, internal_boundary_conditions.InternalBoundaryConditions
    )
    np.testing.assert_array_equal(res.T_i, ibc_params.T_i)
    np.testing.assert_array_equal(res.T_e, ibc_params.T_e)
    np.testing.assert_array_equal(res.n_e, ibc_params.n_e)


class NoIBCTest(absltest.TestCase):

  def test_no_ibc_is_active_is_false(self):
    ibc = pydantic_model.NoIBC()
    self.assertFalse(ibc.is_active())

  def test_no_ibc_build_runtime_params(self):
    ibc = pydantic_model.NoIBC()
    runtime_params = ibc.build_runtime_params(0.0)
    self.assertIsInstance(runtime_params, pydantic_model.NoIBCRuntimeParams)

  def test_no_ibc_model_returns_empty(self):
    geo = circular_geometry.CircularConfig(n_rho=5).build_geometry()
    model = no_ibc.NoIBCModel()
    l_mode_state = (
        pedestal_transition_state.PedestalTransitionState.empty_L_mode()
    )
    res = model(
        runtime_params=None,  # pyrefly: ignore[bad-argument-type]
        geo=geo,
        core_profiles=None,  # pyrefly: ignore[bad-argument-type]
        pedestal_transition_state=l_mode_state,
    )
    np.testing.assert_array_equal(res.T_i, jnp.zeros(5))
    np.testing.assert_array_equal(res.T_e, jnp.zeros(5))
    np.testing.assert_array_equal(res.n_e, jnp.zeros(5))


class InternalBoundaryConditionsTest(absltest.TestCase):

  def test_merge(self):
    ibc1 = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([1.0, 0.0, 0.0]),
        T_e=jnp.array([0.0, 2.0, 0.0]),
        n_e=jnp.array([0.0, 0.0, 3.0]),
    )
    ibc2 = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([1.1, 1.1, 0.0]),
        T_e=jnp.array([0.0, 0.0, 2.2]),
        n_e=jnp.array([3.3, 0.0, 0.0]),
    )

    updated_ibc = ibc1.merge(ibc2)

    np.testing.assert_allclose(updated_ibc.T_i, jnp.array([1.1, 1.1, 0.0]))
    np.testing.assert_allclose(updated_ibc.T_e, jnp.array([0.0, 2.0, 2.2]))
    np.testing.assert_allclose(updated_ibc.n_e, jnp.array([3.3, 0.0, 3.0]))

  def test_get_two_point_face_mask(self):
    geo = circular_geometry.CircularConfig(n_rho=5).build_geometry()
    ibc = internal_boundary_conditions.InternalBoundaryConditions(
        T_i=jnp.array([0.0, 5.0, 0.0, 0.0, 0.0]),
        T_e=jnp.array([0.0, 0.0, 0.0, 10.0, 0.0]),
        n_e=jnp.zeros(5),
    )
    mask = ibc.get_two_point_face_mask(geo)
    # Pinned cells are cell 1 (T_i=5) and cell 3 (T_e=10).
    expected = np.array([False, True, False, True, False, False])
    np.testing.assert_array_equal(mask, expected)

  def test_from_config_defaults_to_prescribed(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['internal_boundary_conditions'] = {
        'T_e': {0.0: {0.8: 1.0}},
    }
    torax_config = model_config.ToraxConfig.model_validate(config_dict)
    self.assertEqual(
        torax_config.profile_conditions.internal_boundary_conditions.model_name,
        'prescribed',
    )


if __name__ == '__main__':
  absltest.main()
