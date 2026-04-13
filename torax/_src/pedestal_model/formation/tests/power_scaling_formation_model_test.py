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

import dataclasses
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function_processing
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.physics import scaling_laws
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class PowerScalingFormationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    # Switch to use the PowerScaling formation model.
    config['pedestal'] = {
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
        # Individual tests will override the model name.
        'formation_model': {'model_name': 'martin_scaling'},
    }
    # Add a source so that P_SOL is non-zero.
    config['sources'] = {
        'generic_heat': {
            'gaussian_location': 0.15,
            'gaussian_width': 0.1,
            'P_total': 20.0e6,
            'electron_heat_fraction': 0.8,
        }
    }
    self.torax_config = model_config.ToraxConfig.from_dict(config)
    self.step_fn = run_simulation.make_step_fn(self.torax_config)
    self.initial_state, self.initial_post_processed_outputs = (
        initial_state.get_initial_state_and_post_processed_outputs(self.step_fn)
    )
    self.runtime_params = self.step_fn.runtime_params_provider(t=0.0)

  def test_calculate_P_SOL_total(self):
    P_SOL_total = power_scaling_formation_model.calculate_P_SOL_total(
        self.initial_state.core_profiles.internal_plasma_energy,
        self.initial_state.core_sources,
        self.initial_state.geometry,
    )

    np.testing.assert_allclose(
        P_SOL_total, self.initial_post_processed_outputs.P_SOL_total
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='martin_above_threshold',
          scaling_law='martin',
          power=1e6,
          expected_multiplier=1e-6,
      ),
      dict(
          testcase_name='martin_below_threshold',
          scaling_law='martin',
          power=-1e6,
          expected_multiplier=1.0,
      ),
      dict(
          testcase_name='delabie_above_threshold',
          scaling_law='delabie',
          power=1e6,
          expected_multiplier=1e-6,
      ),
      dict(
          testcase_name='delabie_below_threshold',
          scaling_law='delabie',
          power=-1e6,
          expected_multiplier=1.0,
      ),
  )
  def test_power_scaling_formation_model_suppression(
      self, scaling_law, power, expected_multiplier
  ):
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_law,
        divertor_configuration=scaling_laws.DivertorConfiguration.VT,
    )

    aux_power_profile = power * jnp.ones_like(self.initial_state.geometry.rho)
    high_power_profiles = dataclasses.replace(
        self.initial_state.core_sources,
        T_e={'aux': aux_power_profile},
        T_i={'aux': aux_power_profile},
    )

    transport_multipliers = formation_model(
        self.runtime_params,
        self.initial_state.geometry,
        self.initial_state.core_profiles,
        high_power_profiles,
    )
    for k, multiplier in dataclasses.asdict(transport_multipliers).items():
      np.testing.assert_allclose(
          multiplier,
          expected_multiplier,
          atol=1e-3,
          err_msg=(
              f'{k}={multiplier} is not close to the expected value of'
              f' {expected_multiplier} for scaling law {scaling_law}.'
          ),
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='stays_in_h_mode',
          p_sol_factor=0.9,
          expected_mode=pedestal_transition_state_lib.ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='transitions_to_l_mode',
          p_sol_factor=0.7,
          expected_mode=pedestal_transition_state_lib.ConfinementMode.TRANSITIONING_TO_L_MODE,
      ),
  )
  def test_pedestal_transition_hysteresis(self, p_sol_factor, expected_mode):
    """Tests that H-L transition respects the hysteresis factor."""
    # Get P_LH
    P_LH, _ = scaling_laws.calculate_P_LH(
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
        divertor_configuration=scaling_laws.DivertorConfiguration.VT,
    )

    # Set hysteresis factor
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            P_LH_hysteresis_factor=0.8,
        ),
    )

    # Create a state in H-mode
    initial_transition_state = (
        pedestal_transition_state_lib.PedestalTransitionState(
            confinement_mode=jnp.array(
                pedestal_transition_state_lib.ConfinementMode.H_MODE
            ),
            transition_start_time=jnp.array(0.0),
            T_i_ped_L_mode=jnp.array(0.5),
            T_e_ped_L_mode=jnp.array(0.5),
            n_e_ped_L_mode=jnp.array(0.5e19),
        )
    )

    target_P_SOL = P_LH * p_sol_factor

    with mock.patch.object(
        step_function_processing.power_scaling_formation_model_lib,
        'calculate_P_SOL_total',
    ) as mock_calc:
      mock_calc.return_value = target_P_SOL

      new_state = step_function_processing._update_pedestal_transition_state(
          pedestal_transition_state=initial_transition_state,
          runtime_params=runtime_params,
          geo=self.initial_state.geometry,
          core_profiles=self.initial_state.core_profiles,
          core_sources=self.initial_state.core_sources,
          models=self.step_fn._solver.models,
      )

      self.assertEqual(new_state.confinement_mode, expected_mode)


if __name__ == '__main__':
  absltest.main()
