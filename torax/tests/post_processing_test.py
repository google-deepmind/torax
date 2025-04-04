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
from torax.config import runtime_params as runtime_params_lib
from torax.core_profiles import initialization
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.orchestration import run_simulation
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import sim_test_case
from torax.tests.test_lib import torax_refs


class PostProcessingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    geo_provider = geometry_provider.ConstantGeometryProvider(
        geometry_pydantic_model.CircularConfig().build_geometry()
    )
    sources = default_sources.get_default_sources()
    self.dynamic_runtime_params_slice, self.geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=sources,
        )
    )
    # Make some dummy source profiles.
    ones = np.ones_like(self.geo.rho)
    self.source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        temp_ion={
            'fusion_heat_source': ones,
            'generic_ion_el_heat_source': 2 * ones,
        },
        temp_el={
            'bremsstrahlung_heat_sink': -ones,
            'ohmic_heat_source': ones * 5,
            'fusion_heat_source': ones,
            'generic_ion_el_heat_source': 3 * ones,
            'electron_cyclotron_source': 7 * ones,
        },
        psi={
            'generic_current_source': 2 * ones,
            'electron_cyclotron_source': 2 * ones,
        },
        ne={},
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        profile_conditions=runtime_params.profile_conditions,
        numerics=runtime_params.numerics,
        plasma_composition=runtime_params.plasma_composition,
        sources=sources,
        torax_mesh=self.geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
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
        'P_ei_exchange_ion',
        'P_ei_exchange_el',
        'P_generic_ion',
        'P_generic_el',
        'P_generic_tot',
        'P_alpha_ion',
        'P_alpha_el',
        'P_alpha_tot',
        'P_ohmic',
        'P_brems',
        'P_ecrh',
        'P_icrh_ion',
        'P_icrh_el',
        'P_icrh_tot',
        'P_rad',
        'P_cycl',
        'P_sol_ion',
        'P_sol_el',
        'P_sol_tot',
        'P_external_ion',
        'P_external_el',
        'P_external_tot',
        'P_external_injected',
        'I_ecrh',
        'I_generic',
    }

    self.assertSameElements(expected_keys, integrated_sources.keys())

    # Volume is calculated in terms of a cell integration (see math_utils.py)
    volume = np.sum(self.geo.vpr * self.geo.drho_norm)

    # Check sums of electron and ion heating.
    np.testing.assert_allclose(
        integrated_sources['P_generic_ion']
        + integrated_sources['P_alpha_ion']
        + integrated_sources['P_ei_exchange_ion'],
        integrated_sources['P_sol_ion'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_generic_el']
        + integrated_sources['P_ohmic']
        + integrated_sources['P_brems']
        + integrated_sources['P_ecrh']
        + integrated_sources['P_alpha_el']
        + integrated_sources['P_ei_exchange_el'],
        integrated_sources['P_sol_el'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_sol_el'] + integrated_sources['P_sol_ion'],
        integrated_sources['P_sol_tot'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_external_el']
        + integrated_sources['P_external_ion'],
        integrated_sources['P_external_tot'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_external_tot'] + integrated_sources['P_brems'],
        +integrated_sources['P_alpha_tot'],
        integrated_sources['P_sol_tot'],
    )

    # Check expected values.
    np.testing.assert_allclose(integrated_sources['P_generic_ion'], 2 * volume)

    np.testing.assert_allclose(integrated_sources['P_generic_el'], 3 * volume)

    np.testing.assert_allclose(integrated_sources['P_generic_tot'], 5 * volume)

    np.testing.assert_allclose(
        integrated_sources['P_alpha_ion'],
        integrated_sources['P_alpha_el'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_ohmic'],
        5 * volume,
    )
    np.testing.assert_allclose(
        integrated_sources['P_brems'],
        -volume,
    )

    np.testing.assert_allclose(
        integrated_sources['P_ei_exchange_ion'],
        -integrated_sources['P_ei_exchange_el'],
    )


class PostProcessingSimTest(sim_test_case.SimTestCase):
  """Tests for the cumulative outputs."""

  def test_cumulative_energies_match_power_integrals(self):
    """Tests E_fusion and E_external are calculated correctly."""

    # Use a test config with both external and fusion sources.
    config_name = 'test_all_transport_fusion_qlknn.py'
    torax_config = self._get_torax_config(config_name)

    state_history = run_simulation.run_simulation(torax_config)
    p_alpha = state_history.post_processed_outputs.P_alpha_tot
    p_external = state_history.post_processed_outputs.P_external_tot
    e_fusion = state_history.post_processed_outputs.E_cumulative_fusion
    e_external = state_history.post_processed_outputs.E_cumulative_external
    t = state_history.times

    # Calculate the cumulative energies from the powers.
    e_fusion_expected = scipy.integrate.cumulative_trapezoid(
        p_alpha * 5, t, initial=0.0
    )

    e_external_expected = scipy.integrate.cumulative_trapezoid(
        p_external, t, initial=0.0
    )

    np.testing.assert_allclose(e_fusion, e_fusion_expected)
    np.testing.assert_allclose(e_external, e_external_expected)


if __name__ == '__main__':
  absltest.main()
