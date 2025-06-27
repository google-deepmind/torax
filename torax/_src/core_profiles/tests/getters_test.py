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
from torax._src import jax_utils
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import profile_conditions as profile_conditions_lib
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.physics import formulas
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic

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
    expected_value = np.array([1.4375e20, 1.3125e20, 1.1875e20, 1.0625e20])
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict(
        {
            'n_e': {0: {0: 1.5e20, 1: 1e20}},
            'n_e_nbar_is_fGW': False,
            'n_e_right_bc_is_fGW': False,
            'nbar': 1e20,
            'normalize_n_e_to_nbar': False,
        },
    )
    numerics = numerics_lib.Numerics.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e = getters.get_updated_electron_density(
        static_slice,
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
      # When normalize_n_e_to_nbar=False, take n_e_right_bc from n_e
      (None, False, 1.0e20),
      # Take n_e_right_bc from provided value.
      (0.85e20, False, 0.85e20),
      # normalize_n_e_to_nbar=True, n_e_right_bc from n_e and normalize.
      (None, True, 0.8050314e20),
      # Even when normalize_n_e_to_nbar, boundary condition is absolute.
      (0.5e20, True, 0.5e20),
  )
  def test_density_boundary_condition_override(
      self,
      n_e_right_bc: float | None,
      normalize_n_e_to_nbar: bool,
      expected_value: float,
  ):
    """Tests that setting n_e right boundary works."""
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'n_e': {0: {0: 1.5e20, 1: 1e20}},
        'n_e_nbar_is_fGW': False,
        'n_e_right_bc_is_fGW': False,
        'nbar': 1e20,
        'n_e_right_bc': n_e_right_bc,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
    })
    numerics = numerics_lib.Numerics.from_dict({})
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e = getters.get_updated_electron_density(
        static_slice,
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
    nbar = 1e20
    numerics = numerics_lib.Numerics.from_dict({})
    profile_conditions = profile_conditions_lib.ProfileConditions.from_dict({
        'n_e': {0: {0: 1.5e20, 1: 1e20}},
        'n_e_nbar_is_fGW': False,
        'nbar': nbar,
        'normalize_n_e_to_nbar': True,
        'n_e_right_bc': 0.5e20,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_normalized = getters.get_updated_electron_density(
        static_slice,
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    np.testing.assert_allclose(np.mean(n_e_normalized.value), nbar, rtol=1e-1)

    profile_conditions._update_fields({'normalize_n_e_to_nbar': False})
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_unnormalized = getters.get_updated_electron_density(
        static_slice,
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
        'nbar': 1.0,
        'normalize_n_e_to_nbar': normalize_n_e_to_nbar,
        'n_e_right_bc': 0.5e20,
    })
    torax_pydantic.set_grid(profile_conditions, self.geo.torax_mesh)
    torax_pydantic.set_grid(numerics, self.geo.torax_mesh)
    static_slice = _create_static_slice_mock(profile_conditions)
    n_e_fGW = getters.get_updated_electron_density(
        static_slice,
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )
    profile_conditions._update_fields(
        {
            'n_e_nbar_is_fGW': False,
            'n_e': {0: {0: 1.5e20, 1: 1e20}},
            'nbar': 1e20,
        },
    )
    # Need to reset n_e grid after private _update_fields. Otherwise, the grid
    # is None. Cannot use public ToraxConfig.update_fields since
    # profile_conditions is not a ToraxConfig.
    torax_pydantic.set_grid(
        profile_conditions, self.geo.torax_mesh, mode='relaxed'
    )

    n_e = getters.get_updated_electron_density(
        static_slice,
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    ratio = n_e.value / n_e_fGW.value
    np.all(np.isclose(ratio, ratio[0]))
    self.assertNotEqual(ratio[0], 1.0)

  def test_get_updated_ion_data(self):
    expected_value = np.array([1.4375e20, 1.3125e20, 1.1875e20, 1.0625e20])
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'n_e': {0: {0: 1.5e20, 1: 1e20}},
        'n_e_nbar_is_fGW': False,
        'n_e_right_bc_is_fGW': False,
        'nbar': 1e20,
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

    T_e = cell_variable.CellVariable(
        value=jnp.ones_like(geo.rho_norm) * 100.0,  # ensure full ionization
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(100.0, dtype=jax_utils.get_dtype()),
        dr=geo.drho_norm,
    )
    n_e = getters.get_updated_electron_density(
        static_slice,
        dynamic_runtime_params_slice.profile_conditions,
        geo,
    )
    ions = getters.get_updated_ions(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        n_e,
        T_e,
    )

    Z_eff = dynamic_runtime_params_slice.plasma_composition.Z_eff

    dilution_factor = formulas.calculate_main_ion_dilution_factor(
        ions.Z_i, ions.Z_impurity, Z_eff
    )
    np.testing.assert_allclose(
        ions.n_i.value,
        expected_value * dilution_factor,
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ions.n_impurity.value,
        (expected_value - ions.n_i.value * ions.Z_i) / ions.Z_impurity,
        atol=1e-6,
        rtol=1e-6,
    )

  def test_Z_eff_calculation(self):
    config = default_configs.get_default_config_dict()
    config['plasma_composition']['Z_eff'] = {0.0: 1.0, 1.0: 2.0}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    dynamic_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    # dynamic_runtime_params_slice.plasma_composition.Z_eff_face is not
    # expected to match core_profiles.Z_eff_face, since the main ion dilution
    # does not scale linearly with Z_eff, and thus Z_eff calculated from the
    # face values of the core profiles will not match the interpolated
    # Z_eff_face from plasma_composition. Only the cell grid Z_eff and
    # edge Z_eff should match exactly, since those were actually used to
    # calculate n_i and n_impurity.
    expected_Z_eff = dynamic_runtime_params_slice.plasma_composition.Z_eff
    expected_Z_eff_edge = (
        dynamic_runtime_params_slice.plasma_composition.Z_eff_face[-1]
    )

    calculated_Z_eff = getters._calculate_Z_eff(
        core_profiles.Z_i,
        core_profiles.Z_impurity,
        core_profiles.n_i.value,
        core_profiles.n_impurity.value,
        core_profiles.n_e.value,
    )

    calculated_Z_eff_face = getters._calculate_Z_eff(
        core_profiles.Z_i_face,
        core_profiles.Z_impurity_face,
        core_profiles.n_i.face_value(),
        core_profiles.n_impurity.face_value(),
        core_profiles.n_e.face_value(),
    )

    np.testing.assert_allclose(
        core_profiles.Z_eff,
        expected_Z_eff,
        err_msg=(
            'Calculated Z_eff does not match expectation from config.\n'
            f'Calculated Z_eff: {calculated_Z_eff}\n'
            f'Expected Z_eff: {expected_Z_eff}'
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        core_profiles.Z_eff_face[-1],
        expected_Z_eff_edge,
        err_msg=(
            'Calculated Z_eff edge does not match expectation from config.\n'
            f'Calculated Z_eff edge: {calculated_Z_eff_face[-1]}\n'
            f'Expected Z_eff edge: {expected_Z_eff_edge}'
        ),
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
