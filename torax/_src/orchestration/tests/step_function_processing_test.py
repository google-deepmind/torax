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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.orchestration import step_function_processing
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class StepFunctionProcessingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'explicit_pedestal': True,
        'mode': 'INTERNAL_BOUNDARY_CONDITION',
        'use_formation_model_with_internal_boundary_condition': True,
        'formation_model': {'model_name': 'martin_scaling'},
        'P_LH_hysteresis_factor': 0.8,
        'transition_time_width': 2.0,
        'T_i_ped': 4.5,
        'T_e_ped': 4.5,
        'n_e_ped': 0.62e20,
        'rho_norm_ped_top': 0.9,
    }
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
    self.initial_state, _ = (
        initial_state.get_initial_state_and_post_processed_outputs(self.step_fn)
    )
    self.models = self.step_fn._solver.models

  @parameterized.parameters((True, 1), (False, 0))
  def test_pre_step_pedestal_evaluation(
      self, explicit_pedestal: bool, expected_pedestal_calls: int
  ):
    """pre_step always updates transition state and evaluates pedestal iff explicit."""
    self.torax_config.pedestal._update_fields(
        {'explicit_pedestal': explicit_pedestal}
    )
    step_fn = run_simulation.make_step_fn(self.torax_config)
    init_state, _ = initial_state.get_initial_state_and_post_processed_outputs(
        step_fn
    )
    models = step_fn._solver.models
    pedestal_cls = type(models.pedestal_model)
    with (
        mock.patch.object(
            pedestal_cls,
            'update_transition_state',
            autospec=True,
            side_effect=pedestal_cls.update_transition_state,
        ) as mock_update,
        mock.patch.object(
            pedestal_cls,
            '__call__',
            autospec=True,
            side_effect=pedestal_cls.__call__,
        ) as mock_call,
    ):
      step_function_processing.pre_step(
          input_state=init_state,
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          models=models,
      )
      self.assertEqual(mock_update.call_count, 1)
      self.assertEqual(mock_call.call_count, expected_pedestal_calls)


if __name__ == '__main__':
  absltest.main()
