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
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
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
        'structure_model': {'model_name': 'set_T_ped_n_ped'},
        'mode': 'ADAPTIVE_TRANSPORT',
        # Individual tests will override the model name.
        'formation_model': {
            'model_name': 'martin_scaling',
            'include_dW_dt_in_P_SOL': True,
        },
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
    formation_state = self.initial_state.model_state.pedestal.formation
    self.assertIsInstance(
        formation_state,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    self.assertIsNotNone(formation_state.W_thermal_total)
    self.assertIsNotNone(formation_state.dW_thermal_dt_smoothed)
    self.assertIsNotNone(formation_state.t)
    P_SOL_total = power_scaling_formation_model.calculate_P_SOL_total(
        self.initial_state.core_sources,
        self.initial_state.geometry,
        dW_thermal_dt_smoothed=formation_state.dW_thermal_dt_smoothed,
    )

    np.testing.assert_allclose(
        P_SOL_total, self.initial_post_processed_outputs.P_SOL_total
    )

  def test_calculate_P_SOL_total_without_dW_dt(self):
    formation_state = self.initial_state.model_state.pedestal.formation
    self.assertIsInstance(
        formation_state,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    self.assertIsNotNone(formation_state.dW_thermal_dt_smoothed)
    P_SOL = power_scaling_formation_model.calculate_P_SOL_total(
        self.initial_state.core_sources,
        self.initial_state.geometry,
        dW_thermal_dt_smoothed=formation_state.dW_thermal_dt_smoothed,
    )
    P_heat = power_scaling_formation_model.calculate_P_SOL_total(
        self.initial_state.core_sources,
        self.initial_state.geometry,
        dW_thermal_dt_smoothed=None,
    )
    dW_dt = formation_state.dW_thermal_dt_smoothed
    np.testing.assert_allclose(P_heat, P_SOL + dW_dt)

  def test_power_scaling_formation_state_none_when_dW_dt_disabled(self):
    self.assertIsInstance(
        self.runtime_params.pedestal.formation,
        power_scaling_formation_model.PowerScalingFormationRuntimeParams,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            formation=dataclasses.replace(
                self.runtime_params.pedestal.formation,
                include_dW_dt_in_P_SOL=False,
            ),
        ),
    )
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
    )
    state = formation_model.initial_state(
        runtime_params=runtime_params,
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
    )
    self.assertIsNone(state.W_thermal_total)
    self.assertIsNone(state.dW_thermal_dt_smoothed)
    self.assertIsNone(state.t)

    with self.subTest('update_state'):
      updated_state = formation_model.update_state(
          runtime_params=runtime_params,
          geo=self.initial_state.geometry,
          core_profiles=self.initial_state.core_profiles,
          formation_state=state,
      )
      self.assertIsInstance(
          updated_state,
          power_scaling_formation_model.PowerScalingFormationState,
      )
      self.assertIsNone(updated_state.W_thermal_total)
      self.assertIsNone(updated_state.dW_thermal_dt_smoothed)
      self.assertIsNone(updated_state.t)

    with self.subTest('__call__'):
      multipliers = formation_model(
          runtime_params=runtime_params,
          geo=self.initial_state.geometry,
          core_profiles=self.initial_state.core_profiles,
          source_profiles=self.initial_state.core_sources,
          confinement_mode=(
              pedestal_runtime_params_lib.ConfinementMode.L_MODE
          ),
          formation_state=state,
      )
      self.assertIsNotNone(multipliers)

    with self.subTest('evaluate_transition_conditions'):
      trigger_l_to_h, trigger_h_to_l = (
          formation_model.evaluate_transition_conditions(
              runtime_params=runtime_params,
              geo=self.initial_state.geometry,
              core_profiles=self.initial_state.core_profiles,
              source_profiles=self.initial_state.core_sources,
              formation_state=state,
          )
      )
      self.assertIsInstance(bool(trigger_l_to_h), bool)
      self.assertIsInstance(bool(trigger_h_to_l), bool)

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
        pedestal_runtime_params_lib.ConfinementMode.L_MODE,
        self.initial_state.model_state.pedestal.formation,
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
          expected_mode=pedestal_runtime_params_lib.ConfinementMode.H_MODE,
      ),
      dict(
          testcase_name='transitions_to_l_mode',
          p_sol_factor=0.7,
          expected_mode=pedestal_runtime_params_lib.ConfinementMode.L_MODE,
      ),
  )
  def test_pedestal_transition_hysteresis(self, p_sol_factor, expected_mode):
    """Tests that H-L transition respects the hysteresis factor.

    With ADAPTIVE_TRANSPORT mode, the state machine uses simple L<->H
    transitions (no TRANSITIONING states). The hysteresis factor means:
    - Stays in H_MODE when P_SOL > P_LH * hysteresis_factor
    - Transitions to L_MODE when P_SOL < P_LH * hysteresis_factor

    Args:
      p_sol_factor: Factor to multiply P_LH to get target P_SOL.
      expected_mode: Expected confinement mode after update.
    """
    # Get P_LH
    P_LH, _ = scaling_laws.calculate_P_LH(
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
        divertor_configuration=scaling_laws.DivertorConfiguration.VT,
    )

    # Set hysteresis factor
    self.assertIsInstance(
        self.runtime_params.pedestal.formation,
        power_scaling_formation_model.PowerScalingFormationRuntimeParams,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            formation=dataclasses.replace(
                self.runtime_params.pedestal.formation,
                P_LH_hysteresis_factor=0.8,
            ),
        ),
    )

    # Create a state in H-mode
    initial_transition_state = dataclasses.replace(
        self.initial_state.pedestal_transition_state,
        confinement_mode=pedestal_runtime_params_lib.ConfinementMode.H_MODE,
        transition_start_time=jnp.array(0.0),
        T_i_ped_L_mode=jnp.array(0.5),
        T_e_ped_L_mode=jnp.array(0.5),
        n_e_ped_L_mode=jnp.array(0.5e19),
    )

    target_P_SOL = P_LH * p_sol_factor

    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
    ) as mock_calc:
      mock_calc.return_value = target_P_SOL

      new_state, _ = (
          self.step_fn._solver.models.pedestal_model.update_transition_state(
              runtime_params=runtime_params,
              geo=self.initial_state.geometry,
              core_profiles=self.initial_state.core_profiles,
              source_profiles=self.initial_state.core_sources,
              pedestal_transition_state=initial_transition_state,
              pedestal_state=self.initial_state.model_state.pedestal,
          )
      )

      self.assertEqual(new_state.confinement_mode, expected_mode)

  def test_formation_model_uses_hysteresis_threshold_in_h_mode(self):
    """Verifies that in H-mode the sigmoid center uses P_LH * hysteresis."""
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
        divertor_configuration=scaling_laws.DivertorConfiguration.VT,
    )

    # Set hysteresis factor
    self.assertIsInstance(
        self.runtime_params.pedestal.formation,
        power_scaling_formation_model.PowerScalingFormationRuntimeParams,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            formation=dataclasses.replace(
                self.runtime_params.pedestal.formation,
                P_LH_hysteresis_factor=0.5,
            ),
        ),
    )

    # Mock calculate_P_SOL_total and calculate_P_LH to avoid depending on
    # specific profile values and scaling laws in this test.
    target_P_LH = 100.0e6
    target_P_SOL = 75.0e6

    with mock.patch.object(
        power_scaling_formation_model, 'calculate_P_SOL_total'
    ) as mock_calc_p_sol:
      mock_calc_p_sol.return_value = jnp.array(target_P_SOL)

      with mock.patch.object(scaling_laws, 'calculate_P_LH') as mock_calc_p_lh:
        mock_calc_p_lh.return_value = (jnp.array(target_P_LH), None)

        multiplier_l_mode = formation_model(
            runtime_params,
            self.initial_state.geometry,
            self.initial_state.core_profiles,
            self.initial_state.core_sources,
            pedestal_runtime_params_lib.ConfinementMode.L_MODE,
            self.initial_state.model_state.pedestal.formation,
        )
        multiplier_h_mode = formation_model(
            runtime_params,
            self.initial_state.geometry,
            self.initial_state.core_profiles,
            self.initial_state.core_sources,
            pedestal_runtime_params_lib.ConfinementMode.H_MODE,
            self.initial_state.model_state.pedestal.formation,
        )

    # In H-mode, the effective P_LH is lower (P_LH * 0.5), so P_SOL is
    # more likely to be above threshold, meaning the multiplier should be
    # smaller (more suppression). In L-mode the full P_LH is used.
    self.assertLess(
        multiplier_h_mode.chi_e_multiplier,
        multiplier_l_mode.chi_e_multiplier,
    )

  def test_power_scaling_formation_model_updates_state(self):
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
        divertor_configuration=scaling_laws.DivertorConfiguration.VT,
    )
    initial_formation_state = self.initial_state.model_state.pedestal.formation
    self.assertIsInstance(
        initial_formation_state,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(0.1))
    modified_core_profiles = dataclasses.replace(
        self.initial_state.core_profiles,
        T_e=dataclasses.replace(
            self.initial_state.core_profiles.T_e,
            value=self.initial_state.core_profiles.T_e.value * 2.0,
        ),
    )
    # When dt > 0.0, update_state advances state.
    updated_state = formation_model.update_state(
        runtime_params=runtime_params,
        geo=self.initial_state.geometry,
        core_profiles=modified_core_profiles,
        formation_state=initial_formation_state,
    )
    self.assertIsInstance(
        updated_state,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    self.assertIsNotNone(updated_state.t)
    self.assertIsNotNone(updated_state.W_thermal_total)
    self.assertIsNotNone(initial_formation_state.W_thermal_total)
    self.assertIsNotNone(updated_state.dW_thermal_dt_smoothed)
    np.testing.assert_allclose(updated_state.t, 0.1)
    self.assertGreater(
        updated_state.W_thermal_total,
        initial_formation_state.W_thermal_total,
    )
    self.assertGreater(updated_state.dW_thermal_dt_smoothed, 0.0)

    # When dt == 0.0, update_state is a safe no-op.
    noop_state = formation_model.update_state(
        runtime_params=runtime_params,
        geo=self.initial_state.geometry,
        core_profiles=modified_core_profiles,
        formation_state=updated_state,
    )
    self.assertIsInstance(
        noop_state,
        power_scaling_formation_model.PowerScalingFormationState,
    )
    self.assertIsNotNone(noop_state.t)
    self.assertIsNotNone(noop_state.W_thermal_total)
    self.assertIsNotNone(noop_state.dW_thermal_dt_smoothed)
    assert noop_state.t is not None
    assert noop_state.W_thermal_total is not None
    assert noop_state.dW_thermal_dt_smoothed is not None
    np.testing.assert_allclose(noop_state.t, updated_state.t)
    np.testing.assert_allclose(
        noop_state.W_thermal_total, updated_state.W_thermal_total
    )
    np.testing.assert_allclose(
        noop_state.dW_thermal_dt_smoothed,
        updated_state.dW_thermal_dt_smoothed,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='above_P_LH',
          p_sol=120.0e6,
          expected_trigger_l_to_h=True,
          expected_trigger_h_to_l=False,
      ),
      dict(
          testcase_name='in_hysteresis_band',
          p_sol=90.0e6,
          expected_trigger_l_to_h=False,
          expected_trigger_h_to_l=False,
      ),
      dict(
          testcase_name='below_hysteresis_threshold',
          p_sol=70.0e6,
          expected_trigger_l_to_h=False,
          expected_trigger_h_to_l=True,
      ),
  )
  def test_evaluate_transition_conditions(
      self, p_sol, expected_trigger_l_to_h, expected_trigger_h_to_l
  ):
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
    )
    self.assertIsInstance(
        self.runtime_params.pedestal.formation,
        power_scaling_formation_model.PowerScalingFormationRuntimeParams,
    )
    runtime_params = dataclasses.replace(
        self.runtime_params,
        pedestal=dataclasses.replace(
            self.runtime_params.pedestal,
            formation=dataclasses.replace(
                self.runtime_params.pedestal.formation,
                P_LH_hysteresis_factor=0.8,
            ),
        ),
    )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(p_sol),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(100.0e6), None),
    ):
      trigger_l_to_h, trigger_h_to_l = (
          formation_model.evaluate_transition_conditions(
              runtime_params=runtime_params,
              geo=self.initial_state.geometry,
              core_profiles=self.initial_state.core_profiles,
              source_profiles=self.initial_state.core_sources,
              formation_state=self.initial_state.model_state.pedestal.formation,
          )
      )
    self.assertEqual(bool(trigger_l_to_h), expected_trigger_l_to_h)
    self.assertEqual(bool(trigger_h_to_l), expected_trigger_h_to_l)

  def test_call_uses_formation_state_dW_dt(self):
    formation_model = power_scaling_formation_model.PowerScalingFormationModel(
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
    )
    p_heat = power_scaling_formation_model.calculate_P_SOL_total(
        self.initial_state.core_sources,
        self.initial_state.geometry,
    )
    p_lh, _ = scaling_laws.calculate_P_LH(
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
        scaling_law=scaling_laws.PLHScalingLaw.MARTIN,
        divertor_configuration=formation_model.divertor_configuration,
    )
    low_dw_dt_state = power_scaling_formation_model.PowerScalingFormationState(
        W_thermal_total=jnp.array(1.0e6),
        dW_thermal_dt_smoothed=p_heat - 1.2 * p_lh,
        t=jnp.array(0.0),
    )
    high_dw_dt_state = dataclasses.replace(
        low_dw_dt_state,
        dW_thermal_dt_smoothed=p_heat - 0.8 * p_lh,
    )
    multipliers_low_dw_dt = formation_model(
        runtime_params=self.runtime_params,
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
        source_profiles=self.initial_state.core_sources,
        confinement_mode=pedestal_runtime_params_lib.ConfinementMode.L_MODE,
        formation_state=low_dw_dt_state,
    )
    multipliers_high_dw_dt = formation_model(
        runtime_params=self.runtime_params,
        geo=self.initial_state.geometry,
        core_profiles=self.initial_state.core_profiles,
        source_profiles=self.initial_state.core_sources,
        confinement_mode=pedestal_runtime_params_lib.ConfinementMode.L_MODE,
        formation_state=high_dw_dt_state,
    )
    # Higher dW/dt lowers P_SOL = P_heat - dW/dt, yielding less suppression.
    self.assertGreater(
        multipliers_high_dw_dt.chi_e_multiplier,
        multipliers_low_dw_dt.chi_e_multiplier,
    )


if __name__ == '__main__':
  absltest.main()
