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
from absl.testing import parameterized
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax import geometry
from torax import interpolated_param
from torax.config import config_args
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib


class BoundaryConditionsTest(parameterized.TestCase):
  """Unit tests for the `torax.boundary_conditions` module."""

  @parameterized.parameters(
      dict(
          ne={0.0: {0.0: 1.5, 1.0: 1.0}},
          ne_bound_right=None,
          expected_ne_bound_right=0.327923,  # Value from profile.
      ),
      dict(
          ne={0.0: {0.0: 1.5, 1.0: 1.0}},
          ne_bound_right=interpolated_param.InterpolatedVarSingleAxis(
              {0.0: 0.1, 0.1: 2.0},
              interpolation_mode=interpolated_param.InterpolationMode.STEP,
          ),
          expected_ne_bound_right=2.0,  # Value from boundary condition.
      ),
  )
  def test_setting_boundary_conditions(
      self,
      ne_bound_right,
      ne,
      expected_ne_bound_right,
  ):
    """Tests that setting boundary conditions works."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # state, we want to grab the boundary condition params at time 0.
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            Ti={0.0: {0.0: 27.7, 1.0: 1.0}},
            Te={0.0: {0.0: 42.0, 1.0: 0.0}, 1.0: 0},
            Ti_bound_right=27.7,
            Te_bound_right={0.0: 42.0, 1.0: 0.0},
            ne_bound_right=ne_bound_right,
            ne=ne,
            Ip={0.0: 5, 1.0: 7},
        ),
    )

    geo = geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    initial_dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models_builder.runtime_params,
            geo=geo,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        initial_dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models_builder.runtime_params,
            t=0.5,
            geo=geo,
        )
    )

    bc = core_profile_setters.compute_boundary_conditions(
        dynamic_runtime_params_slice,
        geo,
    )

    updated = config_args.recursive_replace(core_profiles, **bc)

    psi_constraint = (
        6e6
        * (16 * np.pi**4 * constants.CONSTANTS.mu0 * geo.B0 * geo.rmax)
        / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
        * geo.rmax
    )
    np.testing.assert_allclose(updated.temp_ion.right_face_constraint, 27.7)
    np.testing.assert_allclose(updated.temp_el.right_face_constraint, 21.0)
    np.testing.assert_allclose(
        updated.ne.right_face_constraint,
        expected_ne_bound_right,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        updated.psi.right_face_grad_constraint, psi_constraint
    )


if __name__ == '__main__':
  absltest.main()
