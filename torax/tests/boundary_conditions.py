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

"""Tests for module torax.boundary_conditions."""


from absl.testing import absltest
import numpy as np
from torax import boundary_conditions
from torax import config as config_lib
from torax import config_slice
from torax import constants
from torax import geometry
from torax import initial_states


class BoundaryConditionsTest(absltest.TestCase):
  """Unit tests for the `torax.boundary_conditions` module."""

  def test_setting_boundary_conditions(self):
    """Tests that setting boundary conditions works."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # state, we want to grab the boundary condition params at time 0.
    config = config_lib.Config(
        Ti_bound_right=27.7,
        Te_bound_right={0.0: 42.0, 1.0: 0.0},
        ne_bound_right=config_lib.InterpolationParam(
            {0.0: 0.1, 0.1: 2.0},
            interpolation_mode=config_lib.InterpolationMode.STEP,
        ),
        Ip={0.0: 5, 1.0: 7},
    )

    geo = geometry.build_circular_geometry(config)
    core_profiles = initial_states.initial_core_profiles(config, geo)
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config, 0.5)

    bc = boundary_conditions.compute_boundary_conditions(
        dynamic_config_slice,
        geo,
    )

    updated = config_lib.recursive_replace(core_profiles, **bc)

    psi_constraint = 6e6 * constants.CONSTANTS.mu0 / geo.G2_face[-1] * geo.rmax
    np.testing.assert_allclose(updated.temp_ion.right_face_constraint, 27.7)
    np.testing.assert_allclose(updated.temp_el.right_face_constraint, 21.0)
    np.testing.assert_allclose(updated.ne.right_face_constraint, 2.0)
    np.testing.assert_allclose(
        updated.psi.right_face_grad_constraint, psi_constraint
    )


if __name__ == '__main__':
  absltest.main()
