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
import numpy as np
from torax import jax_utils
from torax.config import build_runtime_params
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import formulas
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.stepper import runtime_params as stepper_params_lib
from torax.transport_model import runtime_params as transport_params_lib

SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class UpdatersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

  def test_updated_ion_temperature(self):
    bound = np.array(42.0)
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.DynamicProfileConditions,
        instance=True,
        Ti_bound_right=bound,
        Ti=value,
    )
    result = formulas.get_updated_ion_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  @parameterized.parameters(0, -1)
  def test_updated_ion_temperature_negative_Ti_bound_right(
      self, Ti_bound_right: float
  ):
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.DynamicProfileConditions,
        instance=True,
        Ti_bound_right=np.array(Ti_bound_right),
        Ti=np.array([12.0, 10.0, 8.0, 6.0]),
    )
    with self.assertRaisesRegex(RuntimeError, 'Ti_bound_right'):
      formulas.get_updated_ion_temperature(
          profile_conditions,
          self.geo,
      )

  def test_updated_electron_temperature(self):
    bound = np.array(42.0)
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.DynamicProfileConditions,
        instance=True,
        Te_bound_right=bound,
        Te=value,
    )
    result = formulas.get_updated_electron_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  @parameterized.parameters(0, -1)
  def test_updated_electron_temperature_negative_Te_bound_right(
      self, Te_bound_right: float
  ):
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.DynamicProfileConditions,
        instance=True,
        Te_bound_right=np.array(Te_bound_right),
        Te=np.array([12.0, 10.0, 8.0, 6.0]),
    )
    with self.assertRaisesRegex(RuntimeError, 'Te_bound_right'):
      formulas.get_updated_electron_temperature(
          profile_conditions,
          self.geo,
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
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    ne = formulas.get_ne(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(
        ne.value,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
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

    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )
    ne = formulas.get_ne(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
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
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_normalized = provider(
        t=1.0,
    )

    ne_normalized = formulas.get_ne(
        dynamic_runtime_params_slice_normalized.numerics,
        dynamic_runtime_params_slice_normalized.profile_conditions,
        self.geo,
    )

    np.testing.assert_allclose(np.mean(ne_normalized.value), nbar, rtol=1e-1)

    runtime_params.profile_conditions.normalize_to_nbar = False
    dynamic_runtime_params_slice_unnormalized = provider(
        t=1.0,
    )

    ne_unnormalized = formulas.get_ne(
        dynamic_runtime_params_slice_unnormalized.numerics,
        dynamic_runtime_params_slice_unnormalized.profile_conditions,
        self.geo,
    )

    ratio = ne_unnormalized.value / ne_normalized.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  @parameterized.parameters(
      True,
      False,
  )
  def test_ne_core_profile_setter_with_fGW(
      self,
      normalize_to_nbar: bool,
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
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_fGW = provider(
        t=1.0,
    )
    ne_fGW = formulas.get_ne(
        dynamic_runtime_params_slice_fGW.numerics,
        dynamic_runtime_params_slice_fGW.profile_conditions,
        self.geo,
    )

    runtime_params.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )

    ne = formulas.get_ne(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        self.geo,
    )

    ratio = ne.value / ne_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Zi=1.0, Zimp=10.0, Zeff=1.0, expected=1.0),
      dict(Zi=1.0, Zimp=5.0, Zeff=1.0, expected=1.0),
      dict(Zi=2.0, Zimp=10.0, Zeff=2.0, expected=0.5),
      dict(Zi=2.0, Zimp=5.0, Zeff=2.0, expected=0.5),
      dict(Zi=1.0, Zimp=10.0, Zeff=1.9, expected=0.9),
      dict(Zi=2.0, Zimp=10.0, Zeff=3.6, expected=0.4),
  ])
  def test_get_main_ion_dilution_factor(self, Zi, Zimp, Zeff, expected):
    """Unit test of `get_main_ion_dilution_factor`."""
    np.testing.assert_allclose(
        formulas.get_main_ion_dilution_factor(Zi, Zimp, Zeff),
        expected,
    )

  # pylint: enable=invalid-name


if __name__ == '__main__':
  absltest.main()
