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

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.pedestal_model import (
    runtime_params as pedestal_runtime_params_lib,
)
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config

# pylint: disable=invalid-name
ConfinementMode = pedestal_runtime_params_lib.ConfinementMode


class StepFunctionProcessingTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='explicit_pedestal',
          explicit_pedestal=True,
      ),
      dict(
          testcase_name='implicit_pedestal',
          explicit_pedestal=False,
      ),
  )
  def test_pedestal_transition_updates_in_pre_step_at_t(
      self,
      explicit_pedestal: bool,
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict['time_step_calculator'] = {
        'calculator_type': 'fixed',
        't_initial': 0.0,
        't_final': 2.0,
        'dt': 1.0,
    }
    config_dict['pedestal'] = {
        'explicit_pedestal': explicit_pedestal,
        'structure_model': {
            'model_name': 'set_T_ped_n_ped',
            'T_i_ped': 4.5,
            'T_e_ped': 4.5,
            'n_e_ped': 0.62e20,
            'rho_norm_ped_top': 0.9,
        },
        'formation_model': {
            'model_name': 'prescribed',
            'pedestal_active': ({0.0: False, 0.5: True}, 'STEP'),
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    step_fn = run_simulation.make_step_fn(torax_config)
    sim_state, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs(step_fn)
    )
    self.assertEqual(
        sim_state.pedestal_output.confinement_mode,
        ConfinementMode.L_MODE,
    )

    # Step 1 (t=0.0 -> 1.0): pre_step evaluates transition conditions at t=0.0
    # (where pedestal_active is False), so the step remains in L_MODE.
    sim_state_1, post_processed_1 = step_fn(sim_state, post_processed)
    self.assertEqual(
        sim_state_1.pedestal_output.confinement_mode,
        ConfinementMode.L_MODE,
    )

    # Step 2 (t=1.0 -> 2.0): pre_step evaluates transition conditions at t=1.0
    # (where pedestal_active is True), transitioning to H_MODE.
    sim_state_2, _ = step_fn(sim_state_1, post_processed_1)
    self.assertEqual(
        sim_state_2.pedestal_output.confinement_mode,
        ConfinementMode.H_MODE,
    )


if __name__ == '__main__':
  absltest.main()
