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

"""Unit tests for torax.config.runtime_params."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class RuntimeParamsTest(parameterized.TestCase):

  def test_runtime_params_raises_for_invalid_temp_boundary_conditions(
      self,
  ):
    """Tests that runtime params validate boundary conditions."""
    with self.assertRaises(ValueError):
      general_runtime_params.GeneralRuntimeParams(
          profile_conditions=profile_conditions_lib.ProfileConditions(
              Ti={0.0: {0.0: 12.0, 0.95: 2.0}}
          )
      )

  @parameterized.parameters(
      (
          {0.0: {0.0: 12.0, 1.0: 2.0}},
          None,
      ),  # Ti includes 1.0.
      (
          {0.0: {0.0: 12.0, 1.0: 2.0}},
          1.0,
      ),  # Both provided.
      (
          {0.0: {0.0: 12.0, 0.95: 2.0}},
          1.0,
      ),  # Ti_bound_right provided.
  )
  def test_runtime_params_constructs_with_valid_profile_conditions(
      self,
      Ti,
      Ti_bound_right,
  ):
    """Tests that runtime params validate boundary conditions."""
    general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti=Ti,
            Ti_bound_right=Ti_bound_right,
        )
    )

  def test_runtime_params_build_dynamic_params(self):
    """Test that runtime params can build dynamic params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions()
    )
    torax_mesh = (
        geometry_pydantic_model.CircularConfig().build_geometry().torax_mesh
    )
    torax_pydantic.set_grid(runtime_params, torax_mesh)
    runtime_params.build_dynamic_params(0.0)

  def test_general_runtime_params_with_time_dependent_args(self):
    """Tests that we can build all types of attributes in the runtime params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams.model_validate({
        'plasma_composition': {
            'main_ion': 'D',
            'Zeff': {
                0: {0: 0.1, 1: 0.1},
                1: {0: 0.2, 1: 0.2},
                2: {0: 0.3, 1: 0.3},
            },  # time-dependent with constant radial profile.
        },
        'profile_conditions': {
            'ne_is_fGW': False,  # scalar fields.
            'Ip_tot': {0: 0.2, 1: 0.4, 2: 0.6},  # time-dependent.
        },
        'numerics': {
            # Designate the interpolation mode, as well, setting to "step".
            'resistivity_mult': ({0: 0.3, 1: 0.6, 2: 0.9}, 'step'),
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(
        list(runtime_params.plasma_composition.main_ion.keys()), ['D']
    )
    self.assertEqual(runtime_params.profile_conditions.ne_is_fGW, False)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(runtime_params, geo.torax_mesh)

    t = 1.5
    np.testing.assert_allclose(
        runtime_params.plasma_composition.Zeff.get_value(t), 0.25
    )
    np.testing.assert_allclose(
        runtime_params.profile_conditions.Ip_tot.get_value(t), 0.5
    )
    np.testing.assert_allclose(
        runtime_params.numerics.resistivity_mult.get_value(t), 0.6
    )


if __name__ == '__main__':
  absltest.main()
