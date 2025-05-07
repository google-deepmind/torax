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
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params_slice
from torax.core_profiles import getters
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.physics import formulas
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config
from torax.torax_pydantic import torax_pydantic

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
        T_i_right_bc=bound,
        T_i=value,
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
        T_e_right_bc=bound,
        T_e=value,
    )
    result = getters.get_updated_electron_temperature(
        profile_conditions,
        self.geo,
    )
    np.testing.assert_allclose(result.value, value)
    np.testing.assert_equal(result.right_face_constraint, bound)

  def test_n_e_core_profile_setter(self):
    """Tests that setting n_e works."""
    expected_value = np.array([1.4375, 1.3125, 1.1875, 1.0625])
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict(
        {
            'n_e': {0: {0: 1.5, 1: 1}},
            'n_e_nbar_is_fGW': False,
            'n_e_right_bc_is_fGW': False,
            'nbar': 1,
            'normalize_n_e_to_nbar': False,
        },
    )
    numerics = numerics_lib.Numerics.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )
    np.testing.assert_allclose(
        n_e.value,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
    )

  @parameterized.parameters(
      # When normalize_n_e_to_nbar=False, take n_e_right_bc from n_e[0.0][1.0]
      (None, False, 1.0),
      # Take n_e_right_bc from provided value.
      (0.85, False, 0.85),
      # normalize_n_e_to_nbar=True, n_e_right_bc from n_e[0.0][1.0] and
      # normalize.
      (None, True, 0.8050314),
      # Even when normalize_n_e_to_nbar, boundary condition is absolute.
      (0.5, True, 0.5),
  )
  def test_density_boundary_condition_override(
      self,
      n_e_right_bc: float | None,
      normalize_n_e_to_nbar: bool,
      expected_value: float,
  ):
    """Tests that setting n_e right boundary works."""
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_nbar_is_fGW': False,
        'n_e_right_bc_is_fGW': False,
        'nbar': 1,
        'n_e_right_bc': n_e_right_bc,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
    })
    numerics = numerics_lib.Numerics.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )
    np.testing.assert_allclose(
        n_e.right_face_constraint,
        expected_value,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_n_e_core_profile_setter_with_normalization(
      self,
  ):
    """Tests that normalizing vs. not by nbar gives consistent results."""
    nbar = 1.0
    numerics = numerics_lib.Numerics.from_dict({})
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_nbar_is_fGW': False,
        'nbar': nbar,
        'normalize_n_e_to_nbar': True,
        'n_e_right_bc': 0.5,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_normalized = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    np.testing.assert_allclose(np.mean(n_e_normalized.value), nbar, rtol=1e-1)

    profile_conditions._update_fields({'normalize_n_e_to_nbar': False})
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_unnormalized = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    ratio = n_e_unnormalized.value / n_e_normalized.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  @parameterized.parameters(
      True,
      False,
  )
  def test_n_e_core_profile_setter_with_fGW(
      self,
      normalize_n_e_to_nbar: bool,
  ):
    """Tests setting the Greenwald fraction vs. not gives consistent results."""
    numerics = numerics_lib.Numerics.from_dict({})
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_nbar_is_fGW': True,
        'nbar': 1,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
        'n_e_right_bc': 0.5,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_fGW = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )
    profile_conditions._update_fields({'n_e_nbar_is_fGW': False})

    n_e = getters.get_updated_electron_density(
        static_slice,
        numerics.build_dynamic_params(1.0),
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    ratio = n_e.value / n_e_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  def test_get_ion_density_and_charge_states(self):
    expected_value = np.array([1.4375, 1.3125, 1.1875, 1.0625])
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'n_e': {0: {0: 1.5, 1: 1}},
        'n_e_nbar_is_fGW': False,
        'n_e_right_bc_is_fGW': False,
        'nbar': 1,
        'normalize_n_e_to_nbar': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    dynamic_runtime_params_slice = provider(t=1.0)
    geo = torax_config.geometry.build_provider(t=1.0)

    temp_el = cell_variable.CellVariable(
        value=jnp.ones_like(geo.rho_norm) * 100.0,  # ensure full ionization
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(100.0, dtype=jax_utils.get_dtype()),
        dr=geo.drho_norm,
    )
    n_e = getters.get_updated_electron_density(
        static_slice,
        dynamic_runtime_params_slice.numerics,
        dynamic_runtime_params_slice.profile_conditions,
        geo,
    )
    ni, nimp, Zi, _, Zimp, _ = getters.get_ion_density_and_charge_states(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        n_e,
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


def _create_static_slice_mock(
    profile_conditions: profile_conditions_lib.ProfileConditions,
) -> runtime_params_slice.StaticRuntimeParamsSlice:
  return mock.create_autospec(
      runtime_params_slice.StaticRuntimeParamsSlice,
      instance=True,
      profile_conditions=mock.create_autospec(
          profile_conditions_lib.StaticRuntimeParams,
          instance=True,
          normalize_n_e_to_nbar=profile_conditions.normalize_n_e_to_nbar,
          n_e_right_bc_is_absolute=False
          if profile_conditions.n_e_right_bc is None
          else True,
      ),
  )


if __name__ == '__main__':
  absltest.main()
