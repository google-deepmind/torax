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
from torax.config import config_args
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.sources import source_models as source_models_lib


class BoundaryConditionsTest(parameterized.TestCase):
  """Unit tests for the `torax.boundary_conditions` module."""

  @parameterized.parameters(
      dict(
          ne={0.0: {0.0: 1.5, 1.0: 1.0}},
          ne_bound_right=None,
          expected_ne_bound_right=1.0,  # Value from profile.
      ),
      dict(
          ne={0.0: {0.0: 1.5, 1.0: 1.0}},
          ne_bound_right=(
              (np.array([0.0, 0.1]), np.array([0.1, 2.0])),
              'step',
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
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={0.0: {0.0: 27.7, 1.0: 1.0}},
            Te={0.0: {0.0: 42.0, 1.0: 0.0}, 1.0: 0},
            Ti_bound_right=27.7,
            Te_bound_right={0.0: 42.0, 1.0: 0.0},
            ne_bound_right=ne_bound_right,
            ne=ne,
            ne_is_fGW=False,
            Ip_tot={0.0: 5, 1.0: 7},
            normalize_to_nbar=False,
        ),
    )

    geo = circular_geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    initial_dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_slice,
        initial_dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=0.5,
        )
    )

    bc = core_profile_setters.compute_boundary_conditions_for_t_plus_dt(
        dt=runtime_params.numerics.fixed_dt,
        dynamic_runtime_params_slice_t=None,  # Not used by this test
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        core_profiles_t=core_profiles,
        static_runtime_params_slice=static_slice,
        geo_t_plus_dt=geo,
    )
    # Remove Zi_edge and Zimp_edge which are not used in core_profiles
    bc.pop('Zi_edge')
    bc.pop('Zimp_edge')

    updated = config_args.recursive_replace(core_profiles, **bc)

    psi_constraint = (
        6e6
        * (16 * np.pi**3 * constants.CONSTANTS.mu0 * geo.Phib)
        / (geo.g2g3_over_rhon_face[-1] * geo.F_face[-1])
    )
    # pylint: disable=invalid-name
    Zi_face = core_profiles.Zi_face
    Zeff_face = dynamic_runtime_params_slice.plasma_composition.Zeff_face
    Zimp_face = core_profiles.Zimp_face
    # pylint: enable=invalid-name
    dilution_factor_face = (Zimp_face - Zeff_face) / (
        Zi_face * (Zimp_face - Zi_face)
    )
    expected_ni_bound_right = expected_ne_bound_right * dilution_factor_face[-1]
    expected_nimp_bound_right = (
        expected_ne_bound_right - expected_ni_bound_right * Zi_face[-1]
    ) / Zimp_face[-1]
    np.testing.assert_allclose(updated.temp_ion.right_face_constraint, 27.7)
    np.testing.assert_allclose(updated.temp_el.right_face_constraint, 21.0)
    np.testing.assert_allclose(
        updated.ne.right_face_constraint,
        expected_ne_bound_right,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        updated.ni.right_face_constraint, expected_ni_bound_right
    )
    np.testing.assert_allclose(
        updated.nimp.right_face_constraint, expected_nimp_bound_right
    )
    np.testing.assert_allclose(updated.temp_el.right_face_constraint, 21.0)
    np.testing.assert_allclose(
        updated.psi.right_face_grad_constraint, psi_constraint
    )


if __name__ == '__main__':
  absltest.main()
