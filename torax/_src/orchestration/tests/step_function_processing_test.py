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
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name
ConfinementMode = pedestal_transition_state_lib.ConfinementMode


class StepFunctionProcessingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'explicit_pedestal': True,
        'mode': 'INTERNAL_BOUNDARY_CONDITION',
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
    """Discrete mode transitions are evaluated once per step in pre_step at t."""
    config = default_configs.get_default_config_dict()
    config['numerics'] = {'fixed_dt': 1.0, 't_initial': 0.0, 't_final': 2.0}
    config['time_step_calculator'] = {'calculator_type': 'fixed'}
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'explicit_pedestal': explicit_pedestal,
        'formation_model': {
            'model_name': 'prescribed',
            'pedestal_active': ({0.0: False, 0.5: True}, 'STEP'),
        },
        'T_i_ped': 4.5,
        'T_e_ped': 4.5,
        'n_e_ped': 0.62e20,
        'rho_norm_ped_top': 0.9,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    step_fn = run_simulation.make_step_fn(torax_config)
    sim_state, post_processed = (
        initial_state.get_initial_state_and_post_processed_outputs(step_fn)
    )
    self.assertEqual(
        sim_state.pedestal_transition_state.confinement_mode,
        ConfinementMode.L_MODE,
    )

    # Step 1 (t=0.0 -> 1.0): pre_step evaluates transition conditions at t=0.0
    # (where pedestal_active is False), so the step remains in L_MODE.
    sim_state_1, post_processed_1 = step_fn(sim_state, post_processed)
    self.assertEqual(
        sim_state_1.pedestal_transition_state.confinement_mode,
        ConfinementMode.L_MODE,
    )

    # Step 2 (t=1.0 -> 2.0): pre_step evaluates transition conditions at t=1.0
    # (where pedestal_active is True), transitioning to H_MODE.
    sim_state_2, _ = step_fn(sim_state_1, post_processed_1)
    self.assertEqual(
        sim_state_2.pedestal_transition_state.confinement_mode,
        ConfinementMode.H_MODE,
    )

  def test_prescribed_formation_adaptive_transport_L_to_H_and_H_to_L(self):
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'mode': 'ADAPTIVE_TRANSPORT',
        'formation_model': {
            'model_name': 'prescribed',
            'pedestal_active': {0.0: False, 1.0: True, 2.0: False},
            'base_multiplier': 1e-5,
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    step_fn = run_simulation.make_step_fn(torax_config)
    sim_state, _ = initial_state.get_initial_state_and_post_processed_outputs(
        step_fn
    )

    with self.subTest('initial_L_mode'):
      init_ts = sim_state.pedestal_transition_state
      self.assertEqual(init_ts.confinement_mode, ConfinementMode.L_MODE)
      np.testing.assert_allclose(
          init_ts.pedestal_model_output.transport_multipliers.chi_i_multiplier,
          1.0,
      )

    with self.subTest('L_to_H_transition'):
      _, _, _, _, state_h = step_function_processing.pre_step(
          input_state=dataclasses.replace(sim_state, t=jnp.array(1.5)),
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          models=step_fn._solver.models,
      )
      self.assertEqual(state_h.confinement_mode, ConfinementMode.H_MODE)
      np.testing.assert_allclose(
          state_h.pedestal_model_output.transport_multipliers.chi_i_multiplier,
          1e-5,
      )

    with self.subTest('H_to_L_transition'):
      _, _, _, _, state_l = step_function_processing.pre_step(
          input_state=dataclasses.replace(
              sim_state, t=jnp.array(2.5), pedestal_transition_state=state_h
          ),
          runtime_params_provider=step_fn.runtime_params_provider,
          geometry_provider=step_fn.geometry_provider,
          models=step_fn._solver.models,
      )
      self.assertEqual(state_l.confinement_mode, ConfinementMode.L_MODE)
      np.testing.assert_allclose(
          state_l.pedestal_model_output.transport_multipliers.chi_i_multiplier,
          1.0,
      )


if __name__ == '__main__':
  absltest.main()
