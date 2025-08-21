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
import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import numerics as numerics_lib
from torax._src.config import plasma_composition as plasma_composition_lib
from torax._src.config import profile_conditions as profile_conditions_lib
from torax._src.core_profiles import getters
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.orchestration import run_simulation
from torax._src.physics import charge_states
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
    n_e = getters.get_updated_electron_density(
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
    n_e = getters.get_updated_electron_density(
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
    n_e_normalized = getters.get_updated_electron_density(
        profile_conditions.build_dynamic_params(1.0),
        self.geo,
    )

    np.testing.assert_allclose(np.mean(n_e_normalized.value), nbar, rtol=1e-1)

    profile_conditions._update_fields({'normalize_n_e_to_nbar': False})
    n_e_unnormalized = getters.get_updated_electron_density(
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
    n_e_fGW = getters.get_updated_electron_density(
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
    config['plasma_composition']['Z_eff'] = 2.0
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
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
        dynamic_runtime_params_slice.profile_conditions,
        geo,
    )
    ions = getters.get_updated_ions(
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
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        ions.n_impurity.value,
        (expected_value - ions.n_i.value * ions.Z_i) / ions.Z_impurity,
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
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
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

  def test_get_updated_ions_impurity_mixture(self):
    """Tests that ion densities are correctly calculated for an impurity mixture."""

    # 1. Define a "ground truth" plasma with individual impurity species.
    # Assumes a single deuterium main ion
    T_e = 50.0  # keV
    n_e = 1e20
    n_e_ratio_w = 1e-4  # n_W / n_e
    n_e_ratio_he3 = 0.03  # n_He3 / n_e
    # Calculate ion charges for the ground truth plasma.
    z_d = constants.ION_PROPERTIES_DICT['D'].Z
    z_he3 = constants.ION_PROPERTIES_DICT['He3'].Z
    z_w = charge_states.calculate_average_charge_state_single_species(
        jnp.array(T_e), 'W'
    )
    # Calculate the dilution factor for the ground truth plasma.
    n_d = n_e * (1 - z_w * n_e_ratio_w - z_he3 * n_e_ratio_he3) / z_d
    zeff = float((
        (n_d / n_e) * z_d**2 + n_e_ratio_w * z_w**2 + n_e_ratio_he3 * z_he3**2
    ))
    dilution = n_d / n_e

    # 2. Set up TORAX to use an effective impurity mixture.
    impurity_fraction_w = n_e_ratio_w / (n_e_ratio_w + n_e_ratio_he3)
    impurity_fraction_he3 = 1.0 - impurity_fraction_w
    config_dict = {
        'profile_conditions': {
            'n_e': n_e,
            'T_e': T_e,
            'T_e_right_bc': T_e,
            'n_e_right_bc': n_e,
        },
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'W': impurity_fraction_w,
                'He3': impurity_fraction_he3,
            },
            'Z_eff': zeff,
        },
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    # 3. Call the function under test.
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    T_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, T_e),
        dr=geo.drho_norm,
        right_face_constraint=T_e,
        right_face_grad_constraint=None,
    )
    n_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, n_e),
        dr=geo.drho_norm,
        right_face_constraint=n_e,
        right_face_grad_constraint=None,
    )
    ions = getters.get_updated_ions(
        dynamic_runtime_params_slice,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )

    # 4. Assertions
    # Check that the effective impurity charge is calculated as <Z^2>/<Z>.
    expected_Z_avg = impurity_fraction_w * z_w + impurity_fraction_he3 * z_he3
    expected_Z2_avg = (
        impurity_fraction_w * z_w**2 + impurity_fraction_he3 * z_he3**2
    )
    expected_Z_impurity = expected_Z2_avg / expected_Z_avg
    np.testing.assert_allclose(ions.Z_impurity, expected_Z_impurity, rtol=1e-6)

    # Check that the calculated dilution factor matches the ground truth.
    calculated_dilution = ions.n_i.value / n_e_cell_variable.value
    np.testing.assert_allclose(calculated_dilution, dilution, rtol=1e-6)

    # Check that Z_eff can be reconstructed correctly.
    reconstructed_zeff = (
        ions.n_i.value / n_e_cell_variable.value
    ) * ions.Z_i**2 + (
        ions.n_impurity.value / n_e_cell_variable.value
    ) * ions.Z_impurity**2
    np.testing.assert_allclose(reconstructed_zeff, zeff, rtol=1e-6)

  def test_get_updated_ions_with_n_e_ratios(self):
    """Tests get_updated_ions for n_e_ratios vs fractions mode."""
    # 1. Define ground truth plasma parameters
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ne': 0.005, 'Ar': 0.001}
    impurity_symbols = tuple(n_e_ratios.keys())

    # 2. Calculate equivalent fractions and Z_eff for a fractions-based config
    # Calculate charge states
    z_main = constants.ION_PROPERTIES_DICT['D'].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            jnp.array(t_e_keV), symbol
        )
        for symbol in impurity_symbols
    }

    # Calculate Z_eff
    zeff = (
        1 - sum(r * z_impurities[s] for s, r in n_e_ratios.items())
    ) * z_main + sum(r * z_impurities[s] ** 2 for s, r in n_e_ratios.items())

    # Calculate impurity fractions
    total_impurity_ratio = sum(n_e_ratios.values())
    impurity_fractions = {
        symbol: ratio / total_impurity_ratio
        for symbol, ratio in n_e_ratios.items()
    }

    # 3. Create the two configurations
    base_config_dict = {
        'profile_conditions': {
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},  # to be filled
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    # Config 1: n_e_ratios
    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib.IMPURITY_MODE_NE_RATIOS,
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = model_config.ToraxConfig.from_dict(
        config_dict_ne_ratios
    )

    # Config 2: fractions + Z_eff
    config_dict_fractions = base_config_dict.copy()
    config_dict_fractions['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib.IMPURITY_MODE_FRACTIONS,
            'species': impurity_fractions,
        },
        'Z_eff': float(zeff),
    }
    torax_config_fractions = model_config.ToraxConfig.from_dict(
        config_dict_fractions
    )

    # 4. Run get_updated_ions for both and compare
    def _run_get_updated_ions(torax_config):
      provider = (
          build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
              torax_config
          )
      )
      dynamic_runtime_params_slice = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          dr=geo.drho_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          dr=geo.drho_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return getters.get_updated_ions(
          dynamic_runtime_params_slice,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios = _run_get_updated_ions(torax_config_ne_ratios)
    ions_fractions = _run_get_updated_ions(torax_config_fractions)

    # 5. Assertions
    chex.assert_trees_all_close(ions_ne_ratios, ions_fractions, rtol=1e-5)

  def test_get_updated_ions_with_n_e_ratios_Z_eff(self):
    """Tests get_updated_ions for n_e_ratios_Z_eff vs fractions mode."""
    # 1. Define plasma parameters
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ne': 0.005, 'Ar': 0.001}  # for n_e_ratios mode

    # 2. Create the two configurations
    base_config_dict = {
        'profile_conditions': {
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},  # to be filled
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    # Config 1: n_e_ratios (ground truth)
    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = model_config.ToraxConfig.from_dict(
        config_dict_ne_ratios
    )

    # 3. Run get_updated_ions for n_e_ratios to get the ground truth ions and
    # Z_eff
    def _run_get_updated_ions(torax_config):
      provider = (
          build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
              torax_config
          )
      )
      dynamic_runtime_params_slice = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          dr=geo.drho_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          dr=geo.drho_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return getters.get_updated_ions(
          dynamic_runtime_params_slice,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios = _run_get_updated_ions(torax_config_ne_ratios)
    ground_truth_zeff = ions_ne_ratios.Z_eff

    # 4. Config 2: n_e_ratios_Z_eff (what we are testing)
    config_dict_ne_ratios_zeff = base_config_dict.copy()
    config_dict_ne_ratios_zeff['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': (
                plasma_composition_lib.IMPURITY_MODE_NE_RATIOS_ZEFF
            ),
            'species': {'C': 0.01, 'Ne': None, 'Ar': 0.001},
        },
        'Z_eff': float(
            ground_truth_zeff[0]
        ),  # Use the calculated Z_eff as input
    }
    torax_config_ne_ratios_zeff = model_config.ToraxConfig.from_dict(
        config_dict_ne_ratios_zeff
    )

    # 5. Run get_updated_ions for n_e_ratios_Z_eff
    ions_ne_ratios_zeff = _run_get_updated_ions(torax_config_ne_ratios_zeff)

    # 6. Assertions

    # Reshape the impurity_fractions from the n_e_ratios mode to match the
    # shape from the n_e_ratios_Z_eff mode (n_species, n_grid).
    fractions_2d = np.broadcast_to(
        ions_ne_ratios.impurity_fractions[:, np.newaxis],
        ions_ne_ratios_zeff.impurity_fractions.shape,
    )
    ions_ne_ratios = dataclasses.replace(
        ions_ne_ratios,
        impurity_fractions=fractions_2d,
    )

    chex.assert_trees_all_close(ions_ne_ratios, ions_ne_ratios_zeff, rtol=1e-5)

  @parameterized.parameters(
      (
          plasma_composition_lib.IMPURITY_MODE_FRACTIONS,
          {'Ne': 0.5, 'W': 0.5},
          1.0,
      ),
      (
          plasma_composition_lib.IMPURITY_MODE_NE_RATIOS,
          {'Ne': 0.0, 'W': 0.0},
          None,
      ),
      (
          plasma_composition_lib.IMPURITY_MODE_NE_RATIOS_ZEFF,
          {'Ne': 0.0, 'W': None},
          1.0,
      ),
  )
  def test_get_updated_ions_with_zero_impurities(
      self, impurity_mode, species, Z_eff
  ):
    config_dict = {
        'profile_conditions': {},
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'impurity_mode': impurity_mode,
                'species': species,
            },
        },
        'numerics': {'t_final': 0.1},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    if Z_eff is not None:
      config_dict['plasma_composition']['Z_eff'] = Z_eff
    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    initial_core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    ions = getters.get_updated_ions(
        dynamic_runtime_params_slice,
        geo,
        initial_core_profiles.n_e,
        initial_core_profiles.T_e,
    )

    np.testing.assert_allclose(ions.n_impurity.value, 0.0)
    np.testing.assert_allclose(ions.Z_eff, 1.0)
    np.testing.assert_allclose(ions.n_i.value, initial_core_profiles.n_e.value)

    _, state_history = run_simulation.run_simulation(torax_config)
    np.testing.assert_equal(
        state_history.sim_error,
        state.SimError.NO_ERROR,
        err_msg='Simulation resulted in an unexpected error.',
    )


if __name__ == '__main__':
  absltest.main()
