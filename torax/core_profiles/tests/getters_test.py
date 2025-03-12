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
from torax.config import build_runtime_params
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import getters
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.physics import formulas
from torax.sources import pydantic_model as sources_pydantic_model
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.transport_model import runtime_params as transport_params_lib

SMALL_VALUE = 1e-6


# pylint: disable=invalid-name
class GettersTest(parameterized.TestCase):

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
    result = getters.get_updated_ion_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  def test_updated_electron_temperature(self):
    bound = np.array(42.0)
    value = np.array([12.0, 10.0, 8.0, 6.0])
    profile_conditions = mock.create_autospec(
        profile_conditions_lib.DynamicProfileConditions,
        instance=True,
        Te_bound_right=bound,
        Te=value,
    )
    result = getters.get_updated_electron_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

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
        sources=sources_pydantic_model.Sources.from_dict({}),
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    ne = getters.get_updated_electron_density(
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
        sources=sources_pydantic_model.Sources.from_dict({}),
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )
    ne = getters.get_updated_electron_density(
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
        sources=sources_pydantic_model.Sources.from_dict({}),
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_normalized = provider(
        t=1.0,
    )

    ne_normalized = getters.get_updated_electron_density(
        dynamic_runtime_params_slice_normalized.numerics,
        dynamic_runtime_params_slice_normalized.profile_conditions,
        self.geo,
    )

    np.testing.assert_allclose(np.mean(ne_normalized.value), nbar, rtol=1e-1)

    runtime_params.profile_conditions.normalize_to_nbar = False
    dynamic_runtime_params_slice_unnormalized = provider(
        t=1.0,
    )

    ne_unnormalized = getters.get_updated_electron_density(
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
        sources=sources_pydantic_model.Sources.from_dict({}),
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice_fGW = provider(
        t=1.0,
    )
    ne_fGW = getters.get_updated_electron_density(
        dynamic_runtime_params_slice_fGW.numerics,
        dynamic_runtime_params_slice_fGW.profile_conditions,
        self.geo,
    )

    runtime_params.profile_conditions.ne_is_fGW = False
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )

    ne = getters.get_updated_electron_density(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        self.geo,
    )

    ratio = ne.value / ne_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

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
    sources = sources_pydantic_model.Sources.from_dict({})
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources=sources,
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
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
    ne = getters.get_updated_electron_density(
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        self.geo,
    )
    ni, nimp, Zi, _, Zimp, _ = getters.get_ion_density_and_charge_states(
        static_slice,
        dynamic_runtime_params_slice,
        self.geo,
        ne,
        temp_el,
    )

    Zeff = dynamic_runtime_params_slice.plasma_composition.Zeff

    dilution_factor = formulas.calculate_main_ion_dilution_factor(
        Zi, Zimp, Zeff
    )
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


if __name__ == '__main__':
  absltest.main()
