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
from jax import numpy as jnp
import numpy as np
import scipy
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import run_simulation
from torax._src.orchestration import sim_state
from torax._src.output_tools import post_processing
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import default_configs
from torax._src.test_utils import default_sources
from torax._src.test_utils import sim_test_case
from torax._src.torax_pydantic import model_config


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
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        T_i={
            'fusion': ones,
            'generic_heat': 2 * ones,
        },
        T_e={
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
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
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
        'P_aux_i',
        'P_aux_e',
        'P_aux_total',
        'P_external_injected',
        'I_ecrh',
        'I_aux_generic',
        'S_gas_puff',
        'S_pellet',
        'S_generic_particle',
        'S_total',
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
        integrated_sources['P_aux_e'] + integrated_sources['P_aux_i'],
        integrated_sources['P_aux_total'],
    )

    np.testing.assert_allclose(
        integrated_sources['P_aux_total']
        + integrated_sources['P_bremsstrahlung_e']
        - integrated_sources['P_ohmic_e'],
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

  def test_zero_sources_do_not_make_nans(self):
    source_profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
    )
    input_state = sim_state.ToraxSimState(
        t=jnp.array(0.0),
        dt=jnp.array(1e-3),
        core_profiles=self.core_profiles,
        core_transport=state.CoreTransport.zeros(self.geo),
        core_sources=source_profiles,
        geometry=self.geo,
        solver_numeric_outputs=state.SolverNumericOutputs(),
    )
    post_processed_outputs = post_processing.make_post_processed_outputs(
        sim_state=input_state,
        dynamic_runtime_params_slice=self.dynamic_runtime_params_slice,
    )
    self.assertEqual(
        post_processed_outputs.check_for_errors(), state.SimError.NO_ERROR
    )


class PostProcessingSimTest(sim_test_case.SimTestCase):
  """Tests for the cumulative outputs."""

  def test_cumulative_energies_match_power_integrals(self):
    """Tests E_fusion and E_external are calculated correctly."""

    # Use a test config with both external and fusion sources.
    config_name = 'test_all_transport_fusion_qlknn.py'
    torax_config = self._get_torax_config(config_name)

    _, state_history = run_simulation.run_simulation(torax_config)
    p_fusion = state_history._stacked_post_processed_outputs.P_alpha_total
    p_external = state_history._stacked_post_processed_outputs.P_aux_total
    e_fusion = state_history._stacked_post_processed_outputs.E_fusion
    e_external = state_history._stacked_post_processed_outputs.E_aux
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
