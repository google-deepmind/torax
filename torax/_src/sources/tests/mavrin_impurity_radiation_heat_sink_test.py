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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import constants
from torax._src.config import build_runtime_params
from torax._src.config import plasma_composition
from torax._src.core_profiles import initialization
from torax._src.physics import charge_states
from torax._src.sources import pydantic_model as sources_pydantic_model
from torax._src.sources import source as source_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_heat_sink as impurity_radiation_heat_sink_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax._src.sources.tests import test_lib
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


class MarvinImpurityRadiationHeatSinkTest(test_lib.SingleProfileSourceTestCase):

  def setUp(self):
    super().setUp(
        source_config_class=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
        source_name=impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME,
        model_name=impurity_radiation_mavrin_fit.DEFAULT_MODEL_FUNCTION_NAME,
    )

  def _run_source_model(self, torax_config: model_config.ToraxConfig):
    """Helper to run the impurity radiation model for a given config."""
    provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    dynamic_runtime_params_slice = provider(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
        neoclassical_models,
    )
    return impurity_radiation_mavrin_fit.impurity_radiation_mavrin_fit(
        unused_static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        unused_geo=geo,
        source_name=self._source_name,
        core_profiles=core_profiles,
        unused_calculated_source_profiles=None,
        unused_conductivity=None,
    )

  def test_correct_dynamic_params_built(self):
    # Source models
    sources = sources_pydantic_model.Sources.from_dict({
        impurity_radiation_heat_sink_lib.ImpurityRadiationHeatSink.SOURCE_NAME: {},
    })
    # Set the grid to allows the dynamic params to be built without making the
    # full config.
    torax_pydantic.set_grid(sources, torax_pydantic.Grid1D(nx=4,))
    runtime_params = getattr(sources, self._source_name).build_dynamic_params(
        t=0.0
    )

    # Extract the source we're testing and check that it's been built correctly
    source_config = sources.impurity_radiation
    self.assertIsNotNone(source_config)
    impurity_radiation_sink = source_config.build_source()
    self.assertIsInstance(impurity_radiation_sink, source_lib.Source)

    assert isinstance(
        runtime_params,
        impurity_radiation_mavrin_fit.DynamicRuntimeParams,
    )

  # pylint: disable=invalid-name
  @parameterized.product(
      ion_symbol=[
          ('C',),
          ('N',),
          ('O',),
          ('Ne',),
          ('Ar',),
          ('Kr',),
          ('Xe',),
          ('W',),
      ],
      temperature=[0.1, 1.0, [10.0, 20.0], 90.0],
  )
  def test_calculate_total_impurity_radiation_sanity(
      self, ion_symbol, temperature
  ):
    """Test with valid ions and within temperature range."""
    T_e = np.array(temperature)
    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            np.array([1.0]),
            T_e,
        )
    )
    np.testing.assert_equal(
        LZ_calculated.shape,
        T_e.shape,
        err_msg=(
            f'LZ and T shapes unequal for {ion_symbol} at temperature {T_e}. LZ'
            f' = {LZ_calculated}, LZ.shape = {LZ_calculated.shape}, T_e.shape ='
            f' {T_e.shape}.'
        ),
    )
    # Physical sanity checking
    np.testing.assert_array_less(
        0.0,
        LZ_calculated,
        err_msg=(
            f'Unphysical negative LZ for {ion_symbol} at temperature {T_e}. '
            f'LZ = {LZ_calculated}.'
        ),
    )

  @parameterized.named_parameters(
      ('T_e_low_input', 0.05, 0.1),
      ('T_e_high_input', 150.0, 100.0),
  )
  def test_temperature_clipping(self, T_e_input, T_e_clipped):
    """Test with valid ions and within temperature range."""
    ion_symbol = ('W',)
    impurity_fractions = np.array([1.0])

    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            impurity_fractions,
            T_e_input,
        )
    )
    LZ_expected = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbol,
            impurity_fractions,
            T_e_clipped,
        )
    )

    np.testing.assert_allclose(
        LZ_calculated,
        LZ_expected,
        err_msg=(
            f'T_e clipping not working as expected for T_e_input={T_e_input},'
            f' LZ_calculated = {LZ_calculated}, Z_expected={LZ_expected}'
        ),
    )

  @parameterized.named_parameters(
      (
          'Helium-3',
          {'He3': 1.0},
          [0.1, 2, 10],
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Helium-4',
          {'He4': 1.0},
          [0.1, 2, 10],
          [2.26267051e-36, 3.55291080e-36, 6.25952387e-36],
      ),
      (
          'Lithium',
          {'Li': 1.0},
          [0.1, 2, 10],
          [1.37075024e-35, 9.16765402e-36, 1.60346076e-35],
      ),
      (
          'Beryllium',
          {'Be': 1.0},
          [0.1, 2, 10],
          [6.86895406e-35, 1.88578938e-35, 3.04614535e-35],
      ),
      (
          'Carbon',
          {'C': 1.0},
          [0.1, 2, 10],
          [6.74683566e-34, 5.89332177e-35, 7.94786067e-35],
      ),
      (
          'Nitrogen',
          {'N': 1.0},
          [0.1, 2, 10],
          [6.97912189e-34, 9.68950644e-35, 1.15250226e-34],
      ),
      (
          'Oxygen',
          {'O': 1.0},
          [0.1, 2, 10],
          [4.10676301e-34, 1.57469152e-34, 1.58599054e-34],
      ),
      (
          'Neon',
          {'Ne': 1.0},
          [0.1, 2, 10],
          [1.19151664e-33, 3.27468464e-34, 2.82416557e-34],
      ),
      (
          'Argon',
          {'Ar': 1.0},
          [0.1, 2, 10],
          [1.92265224e-32, 4.02388371e-33, 1.53295491e-33],
      ),
      (
          'Krypton',
          {'Kr': 1.0},
          [0.1, 2, 10],
          [6.57654706e-32, 3.23512795e-32, 7.53285680e-33],
      ),
      (
          'Xenon',
          {'Xe': 1.0},
          [0.1, 2, 10],
          [2.89734288e-31, 8.96916315e-32, 2.87740863e-32],
      ),
      (
          'Tungsten',
          {'W': 1.0},
          [0.1, 2, 10],
          [1.66636258e-31, 4.46651033e-31, 1.31222935e-31],
      ),
      (
          'Mixture',
          {
              'He4': 0.2,
              'Li': 0.1,
              'Be': 0.197,
              'C': 0.1,
              'N': 0.1,
              'O': 0.1,
              'Ne': 0.1,
              'Ar': 0.1,
              'Kr': 0.001,
              'Xe': 0.001,
              'W': 0.001,
          },
          [0.1, 2, 10],
          [2.75762225e-33, 1.04050126e-33, 3.93256085e-34],
      ),
  )
  def test_calculate_total_impurity_radiation(
      self,
      species,
      T_e,
      expected_LZ,
  ):
    """Test calculate_total_impurity_radiation.

    Args:
      species: A dictionary of ion symbols and their fractions.
      T_e: The temperature in KeV.
      expected_LZ: The expected effective cooling curve value.

    expected_LZ references were verified against plots in the Mavrin 2018 paper.
    """
    T_e = np.array(T_e)
    expected_LZ = np.array(expected_LZ)
    ion_symbols = tuple(species.keys())
    impurity_fractions = np.array(tuple(species.values()))
    LZ_calculated = (
        impurity_radiation_mavrin_fit.calculate_total_impurity_radiation(
            ion_symbols,
            impurity_fractions,
            T_e,
        )
    )

    np.testing.assert_allclose(LZ_calculated, expected_LZ, rtol=1e-5)

  @parameterized.named_parameters(
      (
          'Low_Te',
          {'Ne': 0.02, 'Ar': 0.01},  # n_imp/n_e ratios
          1.5,  # T_e in keV
      ),
      (
          'High_Te',
          {'Ne': 0.01, 'W': 0.0001},  # n_imp/n_e ratios
          25.0,  # T_e in keV
      ),
      (
          'Mid_Te_3_impurities',
          {'C': 0.01, 'Ar': 0.005, 'Kr': 0.001},  # n_imp/n_e ratios
          8.0,  # T_e in keV
      ),
  )
  def test_mixture_radiation_matches_sum_of_individual_radiations(
      self,
      impurity_ratios: dict[str, float],
      t_e_keV: float,
  ):
    """Verifies that radiation from a mixture equals the sum from individual species."""

    # --- 1. Calculate ground truth from individual impurity contributions ---
    n_e_true = 1e20  # m^-3
    # Calculate Z and LZ for each individual impurity species
    impurities_z = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            np.array(t_e_keV), symbol
        )
        for symbol in impurity_ratios
    }
    impurities_lz = {
        symbol: impurity_radiation_mavrin_fit._calculate_impurity_radiation_single_species(
            np.array(t_e_keV), symbol
        )
        for symbol in impurity_ratios
    }
    # Calculate true densities, Z_eff, and total radiation
    n_impurities_true = {
        symbol: ratio * n_e_true for symbol, ratio in impurity_ratios.items()
    }
    # Explictly assumes that Z_i = 1.0
    ni_true = n_e_true - sum(
        n_imp * z_imp
        for n_imp, z_imp in zip(
            n_impurities_true.values(), impurities_z.values()
        )
    )
    zeff_true = (ni_true / n_e_true) + sum(
        (n_imp / n_e_true) * (z_imp**2)
        for n_imp, z_imp in zip(
            n_impurities_true.values(), impurities_z.values()
        )
    )
    # This is the reference radiation we expect TORAX to calculate.
    radiation_ref = -n_e_true * sum(
        n_imp * lz
        for n_imp, lz in zip(n_impurities_true.values(), impurities_lz.values())
    )

    # --- 2. Set up TORAX with an effective impurity mixture ---
    n_imp_total_true = sum(n_impurities_true.values())
    impurity_mixture_fractions = {
        symbol: n_imp / n_imp_total_true
        for symbol, n_imp in n_impurities_true.items()
    }
    config_dict = {
        'profile_conditions': {
            'n_e': n_e_true,
            'T_e': t_e_keV,
            'T_i': t_e_keV,
            'n_e_right_bc': n_e_true,
            'T_e_right_bc': t_e_keV,
            'T_i_right_bc': t_e_keV,
        },
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': impurity_mixture_fractions,
            'Z_eff': zeff_true.item(),
        },
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {self._source_name: {'model_name': self._model_name}},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    # --- 3. Call the function under test ---
    calculated_radiation = self._run_source_model(torax_config)

    # --- 4. Assertions ---
    np.testing.assert_allclose(
        calculated_radiation,
        radiation_ref,
        rtol=1e-6,
        err_msg=(
            'Radiation from mixture does not match sum of individual'
            ' radiations.'
        ),
    )

  def test_radiation_from_ne_ratios_vs_fractions(self):
    """Test that n_e_ratios and fractions impurity modes give same radiation."""
    # 1. Define plasma parameters
    t_e_keV = 15.0
    n_e_val = 1e20
    n_e_ratios = {'C': 0.01, 'Ar': 0.001, 'W': 0.0001}
    impurity_symbols = tuple(n_e_ratios.keys())
    main_ion_symbol = 'D'

    # 2. Calculate equivalent fractions and Z_eff for the fractions-based config
    # Calculate charge states
    z_main = constants.ION_PROPERTIES_DICT[main_ion_symbol].Z
    z_impurities = {
        symbol: charge_states.calculate_average_charge_state_single_species(
            np.array(t_e_keV), symbol
        )
        for symbol in impurity_symbols
    }

    # Calculate Z_eff
    zeff = (1 - sum(
        r * z_impurities[s] for s, r in n_e_ratios.items()
    )) * z_main + sum(r * z_impurities[s] ** 2 for s, r in n_e_ratios.items())

    # Calculate impurity fractions
    total_impurity_ne_ratio = sum(n_e_ratios.values())
    impurity_fractions = {
        symbol: ratio / total_impurity_ne_ratio
        for symbol, ratio in n_e_ratios.items()
    }

    # 3. Create the two configurations
    base_config_dict = {
        'profile_conditions': {
            'n_e': n_e_val,
            'T_e': t_e_keV,
            'T_i': t_e_keV,
            'n_e_right_bc': n_e_val,
            'T_e_right_bc': t_e_keV,
            'T_i_right_bc': t_e_keV,
        },
        'plasma_composition': {},  # to be filled
        'numerics': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': {self._source_name: {'model_name': self._model_name}},
        'solver': {},
        'transport': {},
        'pedestal': {},
    }

    # Config 1: n_e_ratios
    config_dict_ne_ratios = base_config_dict.copy()
    config_dict_ne_ratios['plasma_composition'] = {
        'main_ion': main_ion_symbol,
        'impurity': {
            'impurity_mode': (
                plasma_composition.IMPURITY_MODE_NE_RATIOS
            ),
            'species': n_e_ratios,
        },
    }
    torax_config_ne_ratios = model_config.ToraxConfig.from_dict(
        config_dict_ne_ratios
    )

    # Config 2: fractions + Z_eff
    config_dict_fractions = base_config_dict.copy()
    config_dict_fractions['plasma_composition'] = {
        'main_ion': main_ion_symbol,
        'impurity': {
            'impurity_mode': (
                plasma_composition.IMPURITY_MODE_FRACTIONS
            ),
            'species': impurity_fractions,
        },
        'Z_eff': float(zeff),
    }
    torax_config_fractions = model_config.ToraxConfig.from_dict(
        config_dict_fractions
    )

    # 4. Run the impurity radiation model for both and compare
    radiation_ne_ratios = self._run_source_model(torax_config_ne_ratios)
    radiation_fractions = self._run_source_model(torax_config_fractions)

    # 5. Assertions
    np.testing.assert_allclose(
        radiation_ne_ratios,
        radiation_fractions,
        rtol=1e-6,
        err_msg=(
            'Radiation from n_e_ratios mode does not match radiation from'
            ' fractions mode.'
        ),
    )


if __name__ == '__main__':
  absltest.main()
