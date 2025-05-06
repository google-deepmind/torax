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
import scipy
from torax import post_processing
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.orchestration import run_simulation
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import default_configs
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import sim_test_case
from torax.torax_pydantic import model_config


class PostProcessingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['sources'] = default_sources.get_default_source_config()
    torax_config = model_config.ToraxConfig.from_dict(config)
    self.dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(t=0.0)
    )
    self.geo = torax_config.geometry.build_provider(t=0.0)
    # Make some dummy source profiles.
    ones = np.ones_like(self.geo.rho)
    self.source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        temp_ion={
            'fusion': ones,
            'generic_heat': 2 * ones,
        },
        temp_el={
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
            'fusion': ones,
            'generic_heat': 3 * ones,
            'ecrh': 7 * ones,
        },
        psi={
            'generic_current': 2 * ones,
            'ecrh': 2 * ones,
        },
        n_e={},
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    self.core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=self.dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=self.geo,
        source_models=source_models,
    )

  def test_calculate_integrated_sources(self):
    """Checks integrated quantities match expectations."""
    # pylint: disable=protected-access
    integrated_sources = post_processing._calculate_integrated_sources(
        self.geo,
        self.core_profiles,
        self.source_profiles,
        self.dynamic_runtime_params_slice,
    )
    # pylint: enable=protected-access

    expected_keys = {
        'P_ei_exchange_i',
        'P_ei_exchange_e',
        'P_aux_generic_i',
        'P_aux_generic_e',
        'P_aux_generic_total',
        'P_alpha_i',
        'P_alpha_e',
        'P_alpha_total',
        'P_ohmic_e',
        'P_bremsstrahlung_e',
        'P_ecrh_e',
        'P_icrh_i',
        'P_icrh_e',
        'P_icrh_total',
        'P_radiation_e',
        'P_cyclotron_e',
        'P_SOL_i',
        'P_SOL_e',
        'P_SOL_total',
        'P_external_ion',
        'P_external_el',
        'P_external_tot',
        'P_external_injected',
        'I_ecrh',
        'I_aux_generic',
    }

    self.assertSameElements(expected_keys, integrated_sources.keys())

    # Volume is calculated in terms of a cell integration (see math_utils.py)
    volume = np.sum(self.geo.vpr * self.geo.drho_norm)

    # Check sums of electron and ion heating.
    np.testing.assert_allclose(
        integrated_sources['P_aux_generic_i']
        + integrated_sources['P_alpha_i']
        + integrated_sources['P_ei_exchange_i'],
        integrated_sources['P_SOL_i'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_aux_generic_e']
        + integrated_sources['P_ohmic_e']
        + integrated_sources['P_bremsstrahlung_e']
        + integrated_sources['P_ecrh_e']
        + integrated_sources['P_alpha_e']
        + integrated_sources['P_ei_exchange_e'],
        integrated_sources['P_SOL_e'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_SOL_e'] + integrated_sources['P_SOL_i'],
        integrated_sources['P_SOL_total'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_external_el']
        + integrated_sources['P_external_ion'],
        integrated_sources['P_external_tot'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_external_tot']
        + integrated_sources['P_bremsstrahlung_e'],
        +integrated_sources['P_alpha_total'],
        integrated_sources['P_SOL_total'],
    )

    # Check expected values.
    np.testing.assert_allclose(
        integrated_sources['P_aux_generic_i'], 2 * volume
    )
    np.testing.assert_allclose(
        integrated_sources['P_aux_generic_e'], 3 * volume
    )
    np.testing.assert_allclose(
        integrated_sources['P_aux_generic_total'], 5 * volume
    )
    np.testing.assert_allclose(
        integrated_sources['P_alpha_i'],
        integrated_sources['P_alpha_e'],
    )
    np.testing.assert_allclose(
        integrated_sources['P_ohmic_e'],
        5 * volume,
    )
    np.testing.assert_allclose(
        integrated_sources['P_bremsstrahlung_e'],
        -volume,
    )
    np.testing.assert_allclose(
        integrated_sources['P_ei_exchange_i'],
        -integrated_sources['P_ei_exchange_e'],
    )


class PostProcessingSimTest(sim_test_case.SimTestCase):
  """Tests for the cumulative outputs."""

  def test_cumulative_energies_match_power_integrals(self):
    """Tests E_fusion and E_external are calculated correctly."""

    # Use a test config with both external and fusion sources.
    config_name = 'test_all_transport_fusion_qlknn.py'
    torax_config = self._get_torax_config(config_name)

    state_history = run_simulation.run_simulation(torax_config)
    p_fusion = state_history.post_processed_outputs.P_alpha_total
    p_external = state_history.post_processed_outputs.P_external_tot
    e_fusion = state_history.post_processed_outputs.E_fusion
    e_external = state_history.post_processed_outputs.E_aux
    t = state_history.times

    # Calculate the cumulative energies from the powers.
    e_fusion_expected = scipy.integrate.cumulative_trapezoid(
        p_fusion * 5, t, initial=0.0
    )

    e_external_expected = scipy.integrate.cumulative_trapezoid(
        p_external, t, initial=0.0
    )

    np.testing.assert_allclose(e_fusion, e_fusion_expected)
    np.testing.assert_allclose(e_external, e_external_expected)


if __name__ == '__main__':
  absltest.main()
