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
from torax import core_profile_setters
from torax import geometry
from torax import physics
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.sources import source_models as source_models_lib
from torax.stepper import runtime_params as stepper_params_lib
from torax.transport_model import runtime_params as transport_params_lib


SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class CoreProfileSettersTest(parameterized.TestCase):
  """Unit tests for setting the core profiles."""

  def setUp(self):
    super().setUp()
    self.geo = geometry.build_circular_geometry(n_rho=4)

  @parameterized.parameters(
      (0.0, np.array([10.5, 7.5, 4.5, 1.5])),
      (80.0, np.array([1.0, 1.0, 1.0, 1.0])),
      (
          40.0,
          np.array([
              (1.0 + 10.5) / 2,
              (1.0 + 7.5) / 2,
              (1.0 + 4.5) / 2,
              (1.0 + 1.5) / 2,
          ]),
      ),
  )
  def test_temperature_rho_and_time_interpolation(
      self,
      t: float,
      expected_temperature: np.ndarray,
  ):
    """Tests that the temperature rho and time interpolation works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 80.0: {0.0: 1.0}},
            Ti_bound_right=SMALL_VALUE,
            Te={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 80.0: {0.0: 1.0}},
            Te_bound_right=SMALL_VALUE,
        ),
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    dynamic_slice = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        torax_mesh=geo.torax_mesh,
    )(t=t)
    Ti = core_profile_setters.updated_ion_temperature(dynamic_slice, geo)
    Te = core_profile_setters.updated_electron_temperature(dynamic_slice, geo)
    np.testing.assert_allclose(
        Ti.value,
        expected_temperature,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        Te.value,
        expected_temperature,
        rtol=1e-6,
        atol=1e-6,
    )

  @parameterized.parameters(
      (None, None, 2.0, 2.0),
      (1.0, None, 1.0, 2.0),
      (None, 1.0, 2.0, 1.0),
      (None, None, 2.0, 2.0),
  )
  def test_temperature_boundary_condition_override(
      self,
      Ti_bound_right: float | None,
      Te_bound_right: float | None,
      expected_Ti_bound_right: float,
      expected_Te_bound_right: float,
  ):
    """Tests that the temperature boundary condition override works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={
                0.0: {0.0: 12.0, 1.0: 2.0},
            },
            Te={
                0.0: {0.0: 12.0, 1.0: 2.0},
            },
            Ti_bound_right=Ti_bound_right,
            Te_bound_right=Te_bound_right,
        ),
    )
    t = 0.0
    geo = geometry.build_circular_geometry(n_rho=4)
    dynamic_slice = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        torax_mesh=geo.torax_mesh,
    )(
        t=t,
    )
    Ti_bound_right = core_profile_setters.updated_ion_temperature(
        dynamic_slice, geo
    ).right_face_constraint
    Te_bound_right = core_profile_setters.updated_electron_temperature(
        dynamic_slice, geo
    ).right_face_constraint
    self.assertEqual(
        Ti_bound_right,
        expected_Ti_bound_right,
    )
    self.assertEqual(
        Te_bound_right,
        expected_Te_bound_right,
    )

  def test_time_dependent_provider_with_temperature_is_time_dependent(self):
    """Tests that the runtime_params slice provider is time dependent for T."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 3.0: {0.0: SMALL_VALUE}},
            Ti_bound_right=SMALL_VALUE,
            Te={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 3.0: {0.0: SMALL_VALUE}},
            Te_bound_right=SMALL_VALUE,
        ),
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=geo.torax_mesh,
    )

    dynamic_runtime_params_slice = provider(t=1.0,)
    Ti = core_profile_setters.updated_ion_temperature(
        dynamic_runtime_params_slice, geo
    )
    Te = core_profile_setters.updated_electron_temperature(
        dynamic_runtime_params_slice, geo
    )

    np.testing.assert_allclose(
        Ti.value,
        np.array([7.0, 5.0, 3.0, 1.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        Te.value,
        np.array([7.0, 5.0, 3.0, 1.0]),
        atol=1e-6,
        rtol=1e-6,
    )

    dynamic_runtime_params_slice = provider(t=2.0,)
    Ti = core_profile_setters.updated_ion_temperature(
        dynamic_runtime_params_slice, geo
    )
    Te = core_profile_setters.updated_electron_temperature(
        dynamic_runtime_params_slice, geo
    )
    np.testing.assert_allclose(
        Ti.value,
        np.array([3.5, 2.5, 1.5, 0.5]),
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        Te.value,
        np.array([3.5, 2.5, 1.5, 0.5]),
        atol=1e-6,
        rtol=1e-6,
    )

  def test_ne_core_profile_setter(self):
    """Tests that setting ne works."""
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

    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)

    ne, ni = core_profile_setters.updated_density(
        dynamic_runtime_params_slice,
        self.geo,
    )
    dilution_factor = physics.get_main_ion_dilution_factor(
        dynamic_runtime_params_slice.plasma_composition.Zimp,
        dynamic_runtime_params_slice.plasma_composition.Zeff,
    )
    np.testing.assert_allclose(
        ne.value,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ni.value, expected_value * dilution_factor, atol=1e-6, rtol=1e-6,
    )

  @parameterized.parameters(
      # When normalize_to_nbar=False, take ne_bound_right from ne[0.0][1.0]
      (None, False, 1.0),
      # Take ne_bound_right from provided value.
      (0.85, False, 0.85),
      # normalize_to_nbar=True, ne_bound_right from ne[0.0][1.0] and normalize
      (None, True, 0.8050314),
      # Even when normalize_to_nbar, boundary condition is absolute.
      (0.5, True, 0.5),
  )
  def test_density_boundary_condition_override(
      self,
      ne_bound_right: float | None,
      normalize_to_nbar: bool,
      expected_value: float,
  ):
    """Tests that setting ne right boundary works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            ne_bound_right_is_fGW=False,
            nbar=1,
            ne_bound_right=ne_bound_right,
            normalize_to_nbar=normalize_to_nbar,
        )
    )

    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)

    ne, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice,
        self.geo,
    )
    np.testing.assert_allclose(
        ne.right_face_constraint,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_ne_core_profile_setter_with_normalization(self,):
    """Tests that normalizing vs. not by nbar gives consistent results."""
    nbar = 1.0
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            nbar=nbar,
            normalize_to_nbar=True,
            ne_bound_right=0.5,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_normalized = provider(t=1.0,)

    ne_normalized, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice_normalized,
        self.geo,
    )
    np.testing.assert_allclose(np.mean(ne_normalized.value), nbar, rtol=1e-1)

    runtime_params.profile_conditions.normalize_to_nbar = False
    dynamic_runtime_params_slice_unnormalized = provider(t=1.0,)
    ne_unnormalized, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice_unnormalized,
        self.geo,
    )

    ratio = ne_unnormalized.value / ne_normalized.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  @parameterized.parameters(True, False,)
  def test_ne_core_profile_setter_with_fGW(
      self, normalize_to_nbar: bool,
  ):
    """Tests setting the Greenwald fraction vs. not gives consistent results."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=True,
            nbar=1,
            normalize_to_nbar=normalize_to_nbar,
            ne_bound_right=0.5,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_fGW = provider(t=1.0,)

    ne_fGW, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice_fGW,
        self.geo,
    )

    runtime_params.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice = provider(t=1.0,)
    ne, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice,
        self.geo,
    )

    ratio = ne.value / ne_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

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
    """Tests that compute_boundary_conditions works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=ne_is_fGW,
            nbar=1,
            normalize_to_nbar=normalize_to_nbar,
            ne_bound_right=ne_bound_right,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)

    boundary_conditions = core_profile_setters.compute_boundary_conditions(
        dynamic_runtime_params_slice,
        self.geo,
    )

    if (ne_is_fGW and ne_bound_right is None) or (
        ne_bound_right_is_fGW and ne_bound_right is not None
    ):
      # Then we expect the boundary condition to be in fGW.
      # pylint: disable=invalid-name
      nGW = (
          dynamic_runtime_params_slice.profile_conditions.Ip
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

  @parameterized.parameters(
      (
          {0.0: {0.0: 0.0, 1.0: 1.0}},
          np.array([0.125, 0.375, 0.625, 0.875]),
      ),
  )
  def test_initial_psi(
      self,
      psi,
      expected_psi,
  ):
    """Tests that runtime params validate boundary conditions."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            psi=psi,
        )
    )
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources=source_models_builder.runtime_params,
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice,
        self.geo,
        source_models,
    )

    np.testing.assert_allclose(
        core_profiles.psi.value, expected_psi, atol=1e-6, rtol=1e-6
    )

  @parameterized.named_parameters(
      ('Set from Te', None, 1.0), ('Set from Te_bound_right', 0.5, 0.5)
  )
  def test_compute_boundary_conditions_Te(
      self,
      Te_bound_right,
      expected_Te_bound_right,
  ):
    """Tests that compute_boundary_conditions works for Te."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Te={0: {0: 1.5, 1: 1}},
            Te_bound_right=Te_bound_right,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)

    boundary_conditions = core_profile_setters.compute_boundary_conditions(
        dynamic_runtime_params_slice,
        self.geo,
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
    """Tests that compute_boundary_conditions works for Ti."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti={0: {0: 1.5, 1: 1}},
            Ti_bound_right=Ti_bound_right,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0,)

    boundary_conditions = core_profile_setters.compute_boundary_conditions(
        dynamic_runtime_params_slice,
        self.geo,
    )

    self.assertEqual(
        boundary_conditions['temp_ion']['right_face_constraint'],
        expected_Ti_bound_right,
    )


if __name__ == '__main__':
  absltest.main()
