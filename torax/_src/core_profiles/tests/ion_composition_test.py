# Copyright 2026 DeepMind Technologies Limited
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

"""Unit tests for IonComposition and its from_profiles constructor."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import numpy as jnp
import numpy as np
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import getters
from torax._src.core_profiles import ion_composition as ion_composition_lib
from torax._src.core_profiles.plasma_composition import plasma_composition as plasma_composition_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.geometry import config as geometry_config
from torax._src.orchestration import run_simulation
from torax._src.physics import charge_states
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class IonCompositionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)

  def test_from_profiles_data(self):
    expected_value = np.array([1.4375e20, 1.3125e20, 1.1875e20, 1.0625e20])
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions'] = {
        'I_p': 15e6,
        'I_p_source': 'profile_conditions',
        'initial_psi_source': 'j',
        'n_e': {0: {0: 1.5e20, 1: 1e20}},
        'n_e_nbar_is_fGW': False,
        'n_e_right_bc_is_fGW': False,
        'nbar': 1e20,
        'normalize_n_e_to_nbar': False,
    }
    config_dict['plasma_composition']['Z_eff'] = 2.0
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=1.0)
    geo = torax_config.geometry.build_provider(t=1.0)

    T_e = cell_variable.CellVariable(
        value=jnp.ones_like(geo.rho_norm) * 100.0,
        left_face_grad_constraint=jnp.zeros(()),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(100.0, dtype=jax_utils.get_dtype()),
        face_centers=geo.rho_face_norm,
    )
    n_e = getters.get_updated_electron_density(
        runtime_params.profile_conditions,
        geo,
    )
    ions, n_i, n_impurity = ion_composition_lib.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        n_e,
        T_e,
    )

    Z_eff = runtime_params.plasma_composition.Z_eff

    dilution_factor = ion_composition_lib._calculate_main_ion_dilution_factor(
        ions.Z_i, ions.Z_impurity, Z_eff
    )
    np.testing.assert_allclose(
        n_i.value,
        expected_value * dilution_factor,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        n_impurity.value,
        (expected_value - n_i.value * ions.Z_i) / ions.Z_impurity,
        rtol=1e-6,
    )

  def test_Z_eff_calculation(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['plasma_composition']['Z_eff'] = {0.0: 1.0, 1.0: 2.0}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        core_profiles,
        runtime_params,
        _,
    ) = core_profile_helpers.make_initial_inputs(torax_config)

    expected_Z_eff = runtime_params.plasma_composition.Z_eff
    expected_Z_eff_edge = runtime_params.plasma_composition.Z_eff_face[-1]

    calculated_Z_eff = ion_composition_lib._calculate_Z_eff(
        core_profiles.ions.Z_i,
        core_profiles.ions.Z_impurity,
        core_profiles.kinetic.n_i.value,
        core_profiles.kinetic.n_impurity.value,
        core_profiles.kinetic.n_e.value,
    )

    calculated_Z_eff_face = ion_composition_lib._calculate_Z_eff(
        core_profiles.ions.Z_i_face,
        core_profiles.ions.Z_impurity_face,
        core_profiles.kinetic.n_i.face_value(),
        core_profiles.kinetic.n_impurity.face_value(),
        core_profiles.kinetic.n_e.face_value(),
    )

    np.testing.assert_allclose(
        core_profiles.ions.Z_eff,
        expected_Z_eff,
        err_msg=(
            'Calculated Z_eff does not match expectation from config.\n'
            f'Calculated Z_eff: {calculated_Z_eff}\n'
            f'Expected Z_eff: {expected_Z_eff}'
        ),
        rtol=1e-6,
    )

    np.testing.assert_allclose(
        core_profiles.ions.Z_eff_face[-1],
        expected_Z_eff_edge,
        err_msg=(
            'Calculated Z_eff edge does not match expectation from config.\n'
            f'Calculated Z_eff edge: {calculated_Z_eff_face[-1]}\n'
            f'Expected Z_eff edge: {expected_Z_eff_edge}'
        ),
        rtol=1e-6,
    )

  def test_from_profiles_impurity_mixture(self):
    T_e = 50.0
    n_e = 1e20
    n_e_ratio_w = 1e-4
    n_e_ratio_he3 = 0.03
    z_d = constants.ION_PROPERTIES_DICT['D'].Z
    z_he3 = constants.ION_PROPERTIES_DICT['He3'].Z
    z_w = charge_states.calculate_average_charge_state_single_species(
        jnp.array([T_e]), 'W'
    )
    n_d = n_e * (1 - z_w[0] * n_e_ratio_w - z_he3 * n_e_ratio_he3) / z_d
    zeff = float(
        (n_d / n_e) * z_d**2
        + n_e_ratio_w * z_w[0] ** 2
        + n_e_ratio_he3 * z_he3**2
    )
    dilution = n_d / n_e

    impurity_fraction_w = n_e_ratio_w / (n_e_ratio_w + n_e_ratio_he3)
    impurity_fraction_he3 = 1.0 - impurity_fraction_w
    config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e,
            'T_e': T_e,
            'T_e_right_bc': T_e,
            'n_e_right_bc': n_e,
        },
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'impurity_mode': 'fractions',
                'species': {
                    'W': impurity_fraction_w,
                    'He3': impurity_fraction_he3,
                },
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
    torax_config = config.ToraxConfig.from_dict(config_dict)

    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    T_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, T_e),
        face_centers=geo.rho_face_norm,
        right_face_constraint=T_e,
        right_face_grad_constraint=None,
    )
    n_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, n_e),
        face_centers=geo.rho_face_norm,
        right_face_constraint=n_e,
        right_face_grad_constraint=None,
    )
    ions, n_i, n_impurity = ion_composition_lib.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )

    expected_Z_avg = (
        impurity_fraction_w * z_w[0] + impurity_fraction_he3 * z_he3
    )
    expected_Z2_avg = (
        impurity_fraction_w * z_w[0] ** 2 + impurity_fraction_he3 * z_he3**2
    )
    expected_Z_impurity = expected_Z2_avg / expected_Z_avg
    np.testing.assert_allclose(ions.Z_impurity, expected_Z_impurity, rtol=1e-6)

    calculated_dilution = n_i.value / n_e_cell_variable.value
    np.testing.assert_allclose(calculated_dilution, dilution, rtol=1e-6)

    reconstructed_zeff = (n_i.value / n_e_cell_variable.value) * ions.Z_i**2 + (
        n_impurity.value / n_e_cell_variable.value
    ) * ions.Z_impurity**2
    np.testing.assert_allclose(reconstructed_zeff, zeff, rtol=1e-6)

  def test_from_profiles_impurity_mixture_radially_dependent(self):
    T_e = 50.0
    n_e = 1e20
    n_e_ratio_w = np.array([1e-4, 5e-5, 5e-5, 5e-5])
    n_e_ratio_he3 = np.array([0.03, 0.015, 0.015, 0.015])
    z_d = constants.ION_PROPERTIES_DICT['D'].Z
    z_he3 = constants.ION_PROPERTIES_DICT['He3'].Z
    z_w = charge_states.calculate_average_charge_state_single_species(
        jnp.array([T_e]), 'W'
    )
    n_d = n_e * (1 - z_w[0] * n_e_ratio_w - z_he3 * n_e_ratio_he3) / z_d
    zeff = (
        (n_d / n_e) * z_d**2
        + n_e_ratio_w * z_w[0] ** 2
        + n_e_ratio_he3 * z_he3**2
    )
    geo = circular_geometry.CircularConfig(n_rho=4).build_geometry()
    dilution = n_d / n_e

    impurity_fraction_w = n_e_ratio_w / (n_e_ratio_w + n_e_ratio_he3)
    impurity_fraction_he3 = 1.0 - impurity_fraction_w
    config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e,
            'T_e': T_e,
            'T_e_right_bc': T_e,
            'n_e_right_bc': n_e,
        },
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'species': {
                    'W': (geo.rho_norm, impurity_fraction_w),
                    'He3': (geo.rho_norm, impurity_fraction_he3),
                },
                'impurity_mode': (
                    plasma_composition_lib._IMPURITY_MODE_FRACTIONS
                ),
            },
            'Z_eff': (geo.rho_norm, zeff),
        },
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)

    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    T_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, T_e),
        face_centers=geo.rho_face_norm,
        right_face_constraint=T_e,
        right_face_grad_constraint=None,
    )
    n_e_cell_variable = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, n_e),
        face_centers=geo.rho_face_norm,
        right_face_constraint=n_e,
        right_face_grad_constraint=None,
    )
    ions, n_i, n_impurity = ion_composition_lib.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        n_e_cell_variable,
        T_e_cell_variable,
    )

    expected_Z_avg = (
        impurity_fraction_w * z_w[0] + impurity_fraction_he3 * z_he3
    )
    expected_Z2_avg = (
        impurity_fraction_w * z_w[0] ** 2 + impurity_fraction_he3 * z_he3**2
    )
    expected_Z_impurity = expected_Z2_avg / expected_Z_avg
    np.testing.assert_allclose(ions.Z_impurity, expected_Z_impurity, rtol=1e-6)

    calculated_dilution = n_i.value / n_e_cell_variable.value
    np.testing.assert_allclose(calculated_dilution, dilution, rtol=1e-6)

    reconstructed_zeff = (n_i.value / n_e_cell_variable.value) * ions.Z_i**2 + (
        n_impurity.value / n_e_cell_variable.value
    ) * ions.Z_impurity**2
    np.testing.assert_allclose(reconstructed_zeff, zeff, rtol=1e-6)

  def test_from_profiles_with_n_e_ratios(self):
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ne': 0.005, 'Ar': 0.001}
    impurity_symbols = tuple(n_e_ratios.keys())

    z_main = constants.ION_PROPERTIES_DICT['D'].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            jnp.array([t_e_keV]), symbol
        )
        for symbol in impurity_symbols
    }

    zeff = (
        1 - sum(r * z_impurities[s][0] for s, r in n_e_ratios.items())
    ) * z_main + sum(r * z_impurities[s][0] ** 2 for s, r in n_e_ratios.items())

    total_impurity_ratio = sum(n_e_ratios.values())
    impurity_fractions = {
        symbol: ratio / total_impurity_ratio
        for symbol, ratio in n_e_ratios.items()
    }

    base_config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib._IMPURITY_MODE_NE_RATIOS,
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = config.ToraxConfig.from_dict(config_dict_ne_ratios)

    config_dict_fractions = base_config_dict.copy()
    config_dict_fractions['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib._IMPURITY_MODE_FRACTIONS,
            'species': impurity_fractions,
        },
        'Z_eff': float(zeff),
    }
    torax_config_fractions = config.ToraxConfig.from_dict(config_dict_fractions)

    def _run_from_profiles(torax_config):
      provider = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )
      runtime_params = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          face_centers=geo.rho_face_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          face_centers=geo.rho_face_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return ion_composition_lib.IonComposition.from_profiles(
          runtime_params.plasma_composition,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios = _run_from_profiles(torax_config_ne_ratios)
    ions_fractions = _run_from_profiles(torax_config_fractions)

    chex.assert_trees_all_close(ions_ne_ratios, ions_fractions, rtol=1e-5)

  def test_from_profiles_with_n_e_ratios_He_main_ion(self):
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ne': 0.005}

    config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {
            'main_ion': 'He4',
            'impurity': {
                'impurity_mode': (
                    plasma_composition_lib._IMPURITY_MODE_NE_RATIOS
                ),
                'species': n_e_ratios,
            },
        },
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)

    T_e_cv = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, t_e_keV),
        face_centers=geo.rho_face_norm,
        right_face_constraint=t_e_keV,
        right_face_grad_constraint=None,
    )
    n_e_cv = cell_variable.CellVariable(
        value=jnp.full_like(geo.rho_norm, n_e_val),
        face_centers=geo.rho_face_norm,
        right_face_constraint=n_e_val,
        right_face_grad_constraint=None,
    )
    ions, n_i, _ = ion_composition_lib.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        n_e_cv,
        T_e_cv,
    )

    z_main = constants.ION_PROPERTIES_DICT['He4'].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            jnp.array([t_e_keV]), symbol
        )
        for symbol in n_e_ratios
    }
    total_impurity_charge = sum(
        r * z_impurities[s][0] for s, r in n_e_ratios.items()
    )
    expected_dilution = (1.0 - total_impurity_charge) / z_main

    np.testing.assert_allclose(
        n_i.value / n_e_cv.value,
        expected_dilution,
        rtol=1e-6,
    )

    expected_zeff = expected_dilution * z_main**2 + sum(
        r * z_impurities[s][0] ** 2 for s, r in n_e_ratios.items()
    )
    np.testing.assert_allclose(ions.Z_eff, expected_zeff, rtol=1e-6)

  def test_from_profiles_with_n_e_ratios_radially_dependent(self):
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {
        'C': {0.0: 0.01, 1.0: 0.005},
        'Ne': {0.0: 0.005, 1.0: 0.0025},
        'Ar': {0.0: 0.001, 1.0: 0.0005},
    }
    total_rho_zero = 0.01 + 0.005 + 0.001
    total_rho_one = 0.005 + 0.0025 + 0.0005
    impurity_fractions = {
        'C': {0.0: 0.01 / total_rho_zero, 1.0: 0.005 / total_rho_one},
        'Ne': {0.0: 0.005 / total_rho_zero, 1.0: 0.0025 / total_rho_one},
        'Ar': {0.0: 0.001 / total_rho_zero, 1.0: 0.0005 / total_rho_one},
    }
    impurity_symbols = tuple(n_e_ratios.keys())

    z_main = constants.ION_PROPERTIES_DICT['D'].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            jnp.array([t_e_keV]), symbol
        )
        for symbol in impurity_symbols
    }

    base_config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    geo = geometry_config.GeometryConfig.from_dict(
        base_config_dict['geometry']
    ).build_provider(t=0.0)

    total = np.zeros((5,))
    total_sq = np.zeros((5,))
    for s, r in n_e_ratios.items():
      impurity_value_tva = torax_pydantic.TimeVaryingArray.model_validate(r)
      torax_pydantic.set_grid(impurity_value_tva, geo.torax_mesh)
      impurity_value = impurity_value_tva.get_value(t=0.0, grid_type='face')
      z_impurity = z_impurities[s][0]
      total += impurity_value * z_impurity
      total_sq += impurity_value * z_impurity**2

    zeff = (1.0 - total) * z_main + total_sq

    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib._IMPURITY_MODE_NE_RATIOS,
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = config.ToraxConfig.from_dict(config_dict_ne_ratios)

    config_dict_fractions = base_config_dict.copy()
    config_dict_fractions['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': plasma_composition_lib._IMPURITY_MODE_FRACTIONS,
            'species': impurity_fractions,
        },
        'Z_eff': (geo.rho_face_norm, zeff),
    }
    torax_config_fractions = config.ToraxConfig.from_dict(config_dict_fractions)

    def _run_from_profiles(torax_config):
      provider = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )
      runtime_params = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          face_centers=geo.rho_face_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          face_centers=geo.rho_face_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return ion_composition_lib.IonComposition.from_profiles(
          runtime_params.plasma_composition,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios = _run_from_profiles(torax_config_ne_ratios)
    ions_fractions = _run_from_profiles(torax_config_fractions)

    chex.assert_trees_all_close(ions_ne_ratios, ions_fractions, rtol=1e-5)

  def test_from_profiles_with_n_e_ratios_Z_eff(self):
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ne': 0.005, 'Ar': 0.001}

    base_config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = config.ToraxConfig.from_dict(config_dict_ne_ratios)

    def _run_from_profiles(torax_config):
      provider = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )
      runtime_params = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          face_centers=geo.rho_face_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          face_centers=geo.rho_face_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return ion_composition_lib.IonComposition.from_profiles(
          runtime_params.plasma_composition,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios, _, _ = _run_from_profiles(torax_config_ne_ratios)
    ground_truth_zeff = ions_ne_ratios.Z_eff

    config_dict_ne_ratios_zeff = base_config_dict.copy()
    config_dict_ne_ratios_zeff['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': (
                plasma_composition_lib._IMPURITY_MODE_NE_RATIOS_ZEFF
            ),
            'species': {'C': 0.01, 'Ne': None, 'Ar': 0.001},
        },
        'Z_eff': float(ground_truth_zeff[0]),
    }
    torax_config_ne_ratios_zeff = config.ToraxConfig.from_dict(
        config_dict_ne_ratios_zeff
    )

    ions_ne_ratios_zeff, _, _ = _run_from_profiles(torax_config_ne_ratios_zeff)

    chex.assert_trees_all_close(ions_ne_ratios, ions_ne_ratios_zeff, rtol=1e-5)

  def test_from_profiles_with_n_e_ratios_Z_eff_radially_dependent(self):
    t_e_keV = 10.0
    n_e_val = 1e20
    n_e_ratios = {
        'C': {0.0: 0.01, 1.0: 0.005},
        'Ne': {0.0: 0.005, 1.0: 0.0025},
        'Ar': {0.0: 0.001, 1.0: 0.0005},
    }
    z_main = constants.ION_PROPERTIES_DICT['D'].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            jnp.array([t_e_keV]), symbol
        )
        for symbol in n_e_ratios.keys()
    }

    base_config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_e_right_bc': t_e_keV,
            'n_e_right_bc': n_e_val,
        },
        'plasma_composition': {},
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = config.ToraxConfig.from_dict(config_dict_ne_ratios)

    def _run_from_profiles(torax_config):
      provider = build_runtime_params.RuntimeParamsProvider.from_config(
          torax_config
      )
      runtime_params = provider(t=0.0)
      geo = torax_config.geometry.build_provider(t=0.0)

      t_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, t_e_keV),
          face_centers=geo.rho_face_norm,
          right_face_constraint=t_e_keV,
          right_face_grad_constraint=None,
      )
      n_e_cell_variable = cell_variable.CellVariable(
          value=jnp.full_like(geo.rho_norm, n_e_val),
          face_centers=geo.rho_face_norm,
          right_face_constraint=n_e_val,
          right_face_grad_constraint=None,
      )
      return ion_composition_lib.IonComposition.from_profiles(
          runtime_params.plasma_composition,
          geo,
          n_e_cell_variable,
          t_e_cell_variable,
      )

    ions_ne_ratios = _run_from_profiles(torax_config_ne_ratios)

    config_dict_ne_ratios_zeff = base_config_dict.copy()
    geo = geometry_config.GeometryConfig.from_dict(
        config_dict_ne_ratios_zeff['geometry']
    ).build_provider(t=0.0)

    total = np.zeros((5,))
    total_sq = np.zeros((5,))
    for s, r in n_e_ratios.items():
      impurity_value_tva = torax_pydantic.TimeVaryingArray.model_validate(r)
      torax_pydantic.set_grid(impurity_value_tva, geo.torax_mesh)
      impurity_value = impurity_value_tva.get_value(t=0.0, grid_type='face')
      z_impurity = z_impurities[s][0]
      total += impurity_value * z_impurity
      total_sq += impurity_value * z_impurity**2

    zeff = (1.0 - total) * z_main + total_sq
    config_dict_ne_ratios_zeff['plasma_composition'] = {
        'main_ion': 'D',
        'impurity': {
            'impurity_mode': (
                plasma_composition_lib._IMPURITY_MODE_NE_RATIOS_ZEFF
            ),
            'species': {
                'C': {0.0: 0.01, 1.0: 0.005},
                'Ne': None,
                'Ar': {0.0: 0.001, 1.0: 0.0005},
            },
        },
        'Z_eff': (geo.rho_face_norm, zeff),
    }
    torax_config_ne_ratios_zeff = config.ToraxConfig.from_dict(
        config_dict_ne_ratios_zeff
    )

    ions_ne_ratios_zeff = _run_from_profiles(torax_config_ne_ratios_zeff)

    chex.assert_trees_all_close(ions_ne_ratios, ions_ne_ratios_zeff, rtol=1e-5)

  @parameterized.parameters(
      (
          plasma_composition_lib._IMPURITY_MODE_FRACTIONS,
          {'Ne': 0.5, 'W': 0.5},
          1.0,
      ),
      (
          plasma_composition_lib._IMPURITY_MODE_NE_RATIOS,
          {'Ne': 0.0, 'W': 0.0},
          None,
      ),
      (
          plasma_composition_lib._IMPURITY_MODE_NE_RATIOS_ZEFF,
          {'Ne': 0.0, 'W': None},
          1.0,
      ),
  )
  def test_from_profiles_with_zero_impurities(
      self, impurity_mode, species, Z_eff
  ):
    config_dict = {
        'profile_conditions': {
            'I_p': 15e6,
            'I_p_source': 'profile_conditions',
            'initial_psi_source': 'j',
        },
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'impurity_mode': impurity_mode,
                'species': species,
            },
        },
        'numerics': {},
        'time_step_calculator': {'t_final': 0.1},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }
    if Z_eff is not None:
      config_dict['plasma_composition']['Z_eff'] = Z_eff
    torax_config = config.ToraxConfig.from_dict(config_dict)

    initial_core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    ions, n_i, n_impurity = ion_composition_lib.IonComposition.from_profiles(
        runtime_params.plasma_composition,
        geo,
        initial_core_profiles.kinetic.n_e,
        initial_core_profiles.kinetic.T_e,
    )

    np.testing.assert_allclose(n_impurity.value, 0.0)
    np.testing.assert_allclose(ions.Z_eff, 1.0)
    np.testing.assert_allclose(
        n_i.value, initial_core_profiles.kinetic.n_e.value
    )

    state_history = run_simulation.run_simulation(torax_config, max_steps=100)
    np.testing.assert_equal(
        state_history.sim_error,
        state.SimError.NO_ERROR,
        err_msg='Simulation resulted in an unexpected error.',
    )

  def test_from_profiles_without_right_face_constraint_raises(self):
    config_dict = default_configs.get_default_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    initial_core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    n_e_no_bc = dataclasses.replace(
        initial_core_profiles.kinetic.n_e,
        right_face_constraint=None,
        right_face_grad_constraint=jnp.array(0.0),
    )
    with self.assertRaisesRegex(ValueError, 'right_face_constraint'):
      ion_composition_lib.IonComposition.from_profiles(
          runtime_params.plasma_composition,
          geo,
          n_e_no_bc,
          initial_core_profiles.kinetic.T_e,
      )


if __name__ == '__main__':
  absltest.main()
