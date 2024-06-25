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
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.stepper import runtime_params as stepper_params_lib
from torax.transport_model import runtime_params as transport_params_lib


SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class CoreProfileSettersTest(parameterized.TestCase):
  """Unit tests for setting the core profiles."""

  def setUp(self):
    super().setUp()
    self.geo = geometry.build_circular_geometry(nr=4)

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
        profile_conditions=general_runtime_params.ProfileConditions(
            Ti={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 80.0: {0.0: 1.0}},
            Ti_bound_right=SMALL_VALUE,
            Te={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 80.0: {0.0: 1.0}},
            Te_bound_right=SMALL_VALUE,
        ),
    )
    geo = geometry.build_circular_geometry(nr=4)
    dynamic_slice = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params,
        t=t,
        geo=self.geo,
    )
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
        profile_conditions=general_runtime_params.ProfileConditions(
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
    geo = geometry.build_circular_geometry(nr=4)
    dynamic_slice = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params,
        t=t,
        geo=self.geo,
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
        profile_conditions=general_runtime_params.ProfileConditions(
            Ti={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 3.0: {0.0: SMALL_VALUE}},
            Ti_bound_right=SMALL_VALUE,
            Te={0.0: {0.0: 12.0, 1.0: SMALL_VALUE}, 3.0: {0.0: SMALL_VALUE}},
            Te_bound_right=SMALL_VALUE,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
    )
    geo = geometry.build_circular_geometry(nr=4)

    dynamic_runtime_params_slice = provider(t=1.0, geo=geo)
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

    dynamic_runtime_params_slice = provider(t=2.0, geo=geo)
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
    expected_value = np.array([1.15, 1.05, 0.95, 0.85])
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            ne_bound_right_is_fGW=False,
            nbar=1,
        )
    )

    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
    )
    dynamic_runtime_params_slice = provider(t=1.0, geo=self.geo)

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
      (None, 0.85,),
      (1.0, 1.0,),
  )
  def test_density_boundary_condition_override(
      self,
      ne_bound_right: float | None,
      expected_value: float,
  ):
    """Tests that setting ne works."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            ne_bound_right_is_fGW=False,
            nbar=1,
            ne_bound_right=ne_bound_right,
        )
    )

    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
    )
    dynamic_runtime_params_slice = provider(t=1.0, geo=self.geo)

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

  def test_ne_core_profile_setter_with_normalization(
      self,
  ):
    """Tests that normalizing vs. not by nbar gives consistent results."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=False,
            nbar=1,
            normalize_to_nbar=True,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
    )
    dynamic_runtime_params_slice_normalized = provider(t=1.0, geo=self.geo)

    ne_normalized, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice_normalized,
        self.geo,
    )

    runtime_params.profile_conditions.normalize_to_nbar = False
    dynamic_runtime_params_slice_unnormalized = provider(t=1.0, geo=self.geo)
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
        profile_conditions=general_runtime_params.ProfileConditions(
            ne={0: {0: 1.5, 1: 1}},
            ne_is_fGW=True,
            nbar=1,
            normalize_to_nbar=normalize_to_nbar,
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
    )
    dynamic_runtime_params_slice_fGW = provider(t=1.0, geo=self.geo)

    ne_fGW, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice_fGW,
        self.geo,
    )

    runtime_params.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice = provider(t=1.0, geo=self.geo)
    ne, _ = core_profile_setters.updated_density(
        dynamic_runtime_params_slice,
        self.geo,
    )

    ratio = ne.value / ne_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)


if __name__ == "__main__":
  absltest.main()
