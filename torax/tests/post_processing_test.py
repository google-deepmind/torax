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
import jax
import numpy as np
import scipy
from torax import output
from torax import post_processing
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params as runtime_params_lib
from torax.core_profiles import initialization
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import sim_test_case
from torax.tests.test_lib import torax_refs


class PostProcessingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    self.geo = geometry_pydantic_model.CircularConfig().build_geometry()
    geo_provider = geometry_provider.ConstantGeometryProvider(self.geo)
    sources = default_sources.get_default_sources()
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=sources,
        )
    )
    # Make some dummy source profiles.
    ones = np.ones_like(geo.rho)
    self.source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
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
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    self.core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

  def test_make_outputs(self):
    """Test that post-processing outputs are added to the state."""
    sim_state = state.ToraxSimState(
        core_profiles=self.core_profiles,
        core_transport=state.CoreTransport.zeros(self.geo),
        core_sources=self.source_profiles,
        t=jax.numpy.array(0.0),
        dt=jax.numpy.array(0.1),
        time_step_calculator_state=None,
        post_processed_outputs=state.PostProcessedOutputs.zeros(self.geo),
        stepper_numeric_outputs=state.StepperNumericOutputs(
            outer_stepper_iterations=1,
            stepper_error_state=1,
            inner_solver_iterations=1,
        ),
        geometry=self.geo,
    )

    updated_sim_state = post_processing.make_outputs(
        sim_state, 
        self.geo, 
        dynamic_runtime_params_slice=sim_state.dynamic_runtime_params_slice
    )

    # Check that the outputs were updated.
    for field in state.PostProcessedOutputs.__dataclass_fields__:
      with self.subTest(field=field):
        try:
          np.testing.assert_array_equal(
              getattr(updated_sim_state.post_processed_outputs, field),
              getattr(sim_state.post_processed_outputs, field),
          )
        except AssertionError:
          # At least one field is different, so the test passes.
          return
    # If no assertion error was raised, then all fields are the same
    # so raise an error.
    raise AssertionError('PostProcessedOutputs did not change.')

  def test_calculate_integrated_sources(self):
    """Checks integrated quantities match expectations."""
    # pylint: disable=protected-access
    integrated_sources = post_processing._calculate_integrated_sources(
        self.geo,
        self.core_profiles,
        self.source_profiles,
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
        'P_sol_ion',
        'P_sol_el',
        'P_sol_tot',
        'P_external_ion',
        'P_external_el',
        'P_external_tot',
        'I_ecrh',
        'I_generic',
    }

    self.assertSameElements(integrated_sources.keys(), expected_keys)

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
    config_name = 'test_all_transport_fusion_qlknn'

    # Load the config and run the simulation.
    sim = self._get_sim(config_name + '.py')
    sim_outputs = sim.run()

    # Get the power and energy histories.
    state_history = output.StateHistory(sim_outputs, sim.source_models)
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
