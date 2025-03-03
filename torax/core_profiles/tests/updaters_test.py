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
from jax import numpy as jnp
import numpy as np
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.core_profiles import formulas
from torax.core_profiles import updaters
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.stepper import runtime_params as stepper_params_lib
from torax.transport_model import runtime_params as transport_params_lib

SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

  def test_get_ion_density_and_charge_states(self):
    expected_value = np.array([1.4375, 1.3125, 1.1875, 1.0625])
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            ne_bound_right_is_fGW=False,
            nbar=1,
            normalize_to_nbar=False,
        )
    )
    source_models_builder = source_models_lib.SourceModelsBuilder()
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    static_slice = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0)

    temp_el = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho_norm)
        * 100.0,  # ensure full ionization
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(100.0),
        dr=self.geo.drho_norm,
    )
    ne = formulas.get_ne(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        self.geo,
    )
    ni, nimp, Zi, _, Zimp, _ = updaters.get_ion_density_and_charge_states(
        static_slice,
        dynamic_runtime_params_slice,
        self.geo,
        ne,
        temp_el,
    )

    Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff

    dilution_factor = physics.get_main_ion_dilution_factor(Zi, Zimp, Zeff)
    np.testing.assert_allclose(
        ni.value,
        expected_value * dilution_factor,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        nimp.value,
        (expected_value - ni.value * Zi) / Zimp,
        atol=1e-6,
        rtol=1e-6,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='Set from ne',
          ne_bound_right=None,
          normalize_to_nbar=False,
          ne_is_fGW=False,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=1.0,
      ),
      dict(
          testcase_name='Set and normalize from ne',
          ne_bound_right=None,
          normalize_to_nbar=True,
          ne_is_fGW=False,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.8050314,
      ),
      dict(
          testcase_name='Set and normalize from ne in fGW',
          ne_bound_right=None,
          normalize_to_nbar=True,
          ne_is_fGW=True,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.8050314,
      ),
      dict(
          testcase_name='Set from ne_bound_right',
          ne_bound_right=0.5,
          normalize_to_nbar=False,
          ne_is_fGW=False,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.5,
      ),
      dict(
          testcase_name='Set from ne_bound_right absolute, ignore normalize',
          ne_bound_right=0.5,
          normalize_to_nbar=True,
          ne_is_fGW=False,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.5,
      ),
      dict(
          testcase_name='Set from ne in fGW',
          ne_bound_right=None,
          normalize_to_nbar=False,
          ne_is_fGW=True,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=1,  # This will be scaled by fGW in test.
      ),
      dict(
          testcase_name='Set from ne, ignore ne_bound_right_is_fGW',
          ne_bound_right=None,
          normalize_to_nbar=False,
          ne_is_fGW=False,
          ne_bound_right_is_fGW=True,
          expected_ne_bound_right=1.0,
      ),
      dict(
          testcase_name='Set from ne_bound_right, ignore ne_is_fGW',
          ne_bound_right=0.5,
          normalize_to_nbar=False,
          ne_is_fGW=True,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.5,
      ),
      dict(
          testcase_name=(
              'Set from ne_bound_right, ignore ne_is_fGW, ignore normalize'
          ),
          ne_bound_right=0.5,
          normalize_to_nbar=True,
          ne_is_fGW=True,
          ne_bound_right_is_fGW=False,
          expected_ne_bound_right=0.5,
      ),
  )
  def test_compute_boundary_conditions_ne(
      self,
      ne_bound_right,
      normalize_to_nbar,
      ne_is_fGW,
      ne_bound_right_is_fGW,
      expected_ne_bound_right,
  ):
    """Tests that compute_boundary_conditions_t_plus_dt works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=ne_is_fGW,
            nbar=1,
            normalize_to_nbar=normalize_to_nbar,
            ne_bound_right=ne_bound_right,
        ),
    )
    source_models_builder = source_models_lib.SourceModelsBuilder()
    static_slice = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self.geo.torax_mesh,
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )
    mock_core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
    )

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=runtime_params.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        # Unused
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=mock_core_profiles,  # Unused
    )

    if (ne_is_fGW and ne_bound_right is None) or (
        ne_bound_right_is_fGW and ne_bound_right is not None
    ):
      # Then we expect the boundary condition to be in fGW.
      # pylint: disable=invalid-name
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.Ip_tot
          / (np.pi * self.geo.Rmin**2)
          * 1e20
          / dynamic_runtime_params_slice.numerics.nref
      )
      np.testing.assert_allclose(
          boundary_conditions['ne']['right_face_constraint'],
          expected_ne_bound_right * nGW,
      )
    else:
      np.testing.assert_allclose(
          boundary_conditions['ne']['right_face_constraint'],
          expected_ne_bound_right,
      )

  @parameterized.named_parameters(
      ('Set from Te', None, 1.0), ('Set from Te_bound_right', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_Te(
      self,
      Te_bound_right,
      expected_Te_bound_right,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for Te."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Te={0: {0: 1.5, 1: 1}},
            Te_bound_right=Te_bound_right,
        ),
    )
    source_models_builder = source_models_lib.SourceModelsBuilder()
    static_slice = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self.geo.torax_mesh,
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )

    mock_core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
    )

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=runtime_params.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        # Unused
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=mock_core_profiles,  # Unused
    )

    self.assertEqual(
        boundary_conditions['temp_el']['right_face_constraint'],
        expected_Te_bound_right,
    )

  @parameterized.named_parameters(
      ('Set from Ti', None, 1.0), ('Set from Ti_bound_right', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_Ti(
      self,
      Ti_bound_right,
      expected_Ti_bound_right,
  ):
    """Tests that compute_boundary_conditions_for_t_plus_dt works for Ti."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={0: {0: 1.5, 1: 1}},
            Ti_bound_right=Ti_bound_right,
        ),
    )
    source_models_builder = source_models_lib.SourceModelsBuilder()
    static_slice = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self.geo.torax_mesh,
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )

    mock_core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
    )

    boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=runtime_params.numerics.fixed_dt,
        static_runtime_params_slice=static_slice,
        # Unused
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
        geo_t_plus_dt=self.geo,
        core_profiles_t=mock_core_profiles,  # Unused
    )

    self.assertEqual(
        boundary_conditions['temp_ion']['right_face_constraint'],
        expected_Ti_bound_right,
    )


if __name__ == '__main__':
  absltest.main()
