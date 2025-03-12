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

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import interpolated_param
from torax.config import config_args
from torax.config import profile_conditions
from torax.geometry import pydantic_model as geometry_pydantic_model
import xarray as xr


# pylint: disable=invalid-name
class ProfileConditionsTest(parameterized.TestCase):

  def test_profile_conditions_make_provider(self):
    pc = profile_conditions.ProfileConditions()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

  @parameterized.named_parameters(
      ('no boundary condition', None, 2.0, 200.0),
      ('boundary condition provided', 3.0, 3.0, 3.0),
  )
  def test_profile_conditions_sets_Te_bound_right_correctly(
      self, Te_bound_right, expected_initial_value, expected_second_value
  ):
    """Tests that Te_bound_right is set correctly."""
    pc = profile_conditions.ProfileConditions(
        Te={0: {0: 1.0, 1: 2.0}, 1.5: {0: 100.0, 1: 200.0}},
        Te_bound_right=Te_bound_right,
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    dcs = provider.build_dynamic_params(t=0.0)
    self.assertEqual(dcs.Te_bound_right, expected_initial_value)
    dcs = provider.build_dynamic_params(t=1.5)
    self.assertEqual(dcs.Te_bound_right, expected_second_value)

  @parameterized.named_parameters(
      ('no boundary condition', None, 2.0, 200.0),
      ('boundary condition provided', 3, 3, 3),
  )
  def test_profile_conditions_sets_Ti_bound_right_correctly(
      self, Ti_bound_right, expected_initial_value, expected_second_value
  ):
    """Tests that Ti_bound_right is set correctly."""
    pc = profile_conditions.ProfileConditions(
        Ti={0: {0: 1.0, 1: 2.0}, 1.5: {0: 100.0, 1: 200.0}},
        Ti_bound_right=Ti_bound_right,
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    dcs = provider.build_dynamic_params(t=0.0)
    self.assertEqual(dcs.Ti_bound_right, expected_initial_value)
    dcs = provider.build_dynamic_params(t=1.5)
    self.assertEqual(dcs.Ti_bound_right, expected_second_value)

  @parameterized.named_parameters(
      ('no boundary condition', None, 2.0, 200.0),
      ('boundary condition provided', 3.0, 3.0, 3.0),
  )
  def test_profile_conditions_sets_ne_bound_right_correctly(
      self, ne_bound_right, expected_initial_value, expected_second_value
  ):
    """Tests that ne_bound_right is set correctly."""
    pc = profile_conditions.ProfileConditions(
        ne={0: {0: 1.0, 1: 2.0}, 1.5: {0: 100.0, 1: 200.0}},
        ne_bound_right=ne_bound_right,
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    dcs = provider.build_dynamic_params(t=0.0)
    self.assertEqual(dcs.ne_bound_right, expected_initial_value)
    if ne_bound_right is None:
      self.assertEqual(dcs.ne_bound_right_is_fGW, dcs.ne_is_fGW)
      self.assertFalse(dcs.ne_bound_right_is_absolute)
    else:
      self.assertTrue(dcs.ne_bound_right_is_absolute)
    dcs = provider.build_dynamic_params(t=1.5)
    self.assertEqual(dcs.ne_bound_right, expected_second_value)
    if ne_bound_right is None:
      self.assertEqual(dcs.ne_bound_right_is_fGW, dcs.ne_is_fGW)
      self.assertFalse(dcs.ne_bound_right_is_absolute)
    else:
      self.assertTrue(dcs.ne_bound_right_is_absolute)

  @parameterized.named_parameters(
      ('no psi provided', None, None, None),
      (
          'constant psi provided',
          3.0,
          np.array([3.0, 3.0, 3.0, 3.0]),
          np.array([3.0, 3.0, 3.0, 3.0]),
      ),
      (
          'rho dependent, time independent psi provided',
          {0: 1.0, 1: 2.0},
          np.array([1.125, 1.375, 1.625, 1.875]),
          np.array([1.125, 1.375, 1.625, 1.875]),
      ),
      (
          'rho dependent, time dependent psi provided',
          {0: {0: 1.0, 1: 2.0}, 1.5: {0: 100.0, 1: 200.0}},
          np.array([1.125, 1.375, 1.625, 1.875]),
          np.array([112.5, 137.5, 162.5, 187.5]),
      ),
  )
  def test_profile_conditions_sets_psi_correctly(
      self, psi, expected_initial_value, expected_second_value
  ):
    """Tests that psi is set correctly."""
    geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()
    pc = profile_conditions.ProfileConditions(
        psi=psi,
    )
    provider = pc.make_provider(geo.torax_mesh)
    dcs = provider.build_dynamic_params(t=0.0)
    if psi is None:
      self.assertIsNone(dcs.psi)
    else:
      np.testing.assert_allclose(dcs.psi, expected_initial_value)
    dcs = provider.build_dynamic_params(t=1.5)
    if psi is None:
      self.assertIsNone(dcs.psi)
    else:
      np.testing.assert_allclose(dcs.psi, expected_second_value)

  def test_interpolated_vars_are_only_constructed_once(
      self,
  ):
    """Tests that interpolated vars are only constructed once."""
    pc = profile_conditions.ProfileConditions()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = pc.make_provider(geo.torax_mesh)
    interpolated_params = {}
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        interpolated_params[field] = value

    # Check we don't make any additional calls to construct interpolated vars.
    provider.build_dynamic_params(t=1.0)
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        self.assertIs(value, interpolated_params[field])

  @parameterized.named_parameters(
      dict(testcase_name='float', values=1.0, raises=True),
      dict(testcase_name='int', values=1, raises=True),
      dict(
          testcase_name='invalid dict shortcut',
          values={0.0: 0.0, 0.9: 1.0},
          raises=True,
      ),
      dict(
          testcase_name='valid dict shortcut',
          values={0.0: 0.0, 1.0: 1.0},
          raises=False,
      ),
      dict(
          testcase_name='invalid dict',
          values={0.0: {0.0: 0.0, 0.9: 1.0}},
          raises=True,
      ),
      dict(
          testcase_name='valid dict',
          values={0.0: {0.0: 0.0, 1.0: 1.0}},
          raises=False,
      ),
      dict(
          testcase_name='invalid numpy shortcut',
          values=(np.array([0.0, 0.9]), np.array([0.0, 1.0])),
          raises=True,
      ),
      dict(
          testcase_name='valid numpy shortcut',
          values=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
          raises=False,
      ),
      dict(
          testcase_name='invalid numpy',
          values=(
              np.array([0.0]),
              np.array([0.0, 0.9]),
              np.array([[0.0, 1.0]]),
          ),
          raises=True,
      ),
      dict(
          testcase_name='valid numpy',
          values=(
              np.array([0.0]),
              np.array([0.0, 1.0]),
              np.array([[0.0, 1.0]]),
          ),
          raises=False,
      ),
      dict(
          testcase_name='invalid xarray',
          values=xr.DataArray(
              data=np.array([[0.0, 1.0]]),
              dims=['times', config_args.RHO_NORM],
              coords={config_args.RHO_NORM: [0.0, 0.9], 'times': [0.0]},
          ),
          raises=True,
      ),
      dict(
          testcase_name='valid xarray',
          values=xr.DataArray(
              data=np.array([[0.0, 1.0]]),
              dims=['times', config_args.RHO_NORM],
              coords={config_args.RHO_NORM: [0.0, 1.0], 'times': [0.0]},
          ),
          raises=False,
      ),
  )
  def test_profile_conditions_raises_error_if_boundary_condition_not_defined(
      self,
      values,
      raises,
  ):
    """Tests that an error is raised if the boundary condition is not defined."""
    if raises:
      with self.assertRaises(ValueError):
        profile_conditions.ProfileConditions(
            Ti=values,
            Ti_bound_right=None,
        )
    else:
      profile_conditions.ProfileConditions(
          Ti=values,
          Ti_bound_right=None,
      )


if __name__ == '__main__':
  absltest.main()
