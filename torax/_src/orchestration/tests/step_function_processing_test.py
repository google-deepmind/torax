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

# Constants for test readability.
_P_LH = 10.0  # MW, mocked L-H threshold power.
_HYSTERESIS = 0.8  # P_LH_hysteresis_factor.
_TRANSITION_WIDTH = 2.0  # seconds.


def _make_transition_state(
    mode: ConfinementMode,
    start_time: float = jnp.inf,
    T_i_ped_L: float = 0.5,
    T_e_ped_L: float = 0.4,
    n_e_ped_L: float = 0.5e19,
) -> pedestal_transition_state_lib.PedestalTransitionState:
  """Helper to create a PedestalTransitionState with given values."""
  return pedestal_transition_state_lib.PedestalTransitionState(
      confinement_mode=jnp.array(mode),
      transition_start_time=jnp.array(start_time),
      T_i_ped_L_mode=jnp.array(T_i_ped_L),
      T_e_ped_L_mode=jnp.array(T_e_ped_L),
      n_e_ped_L_mode=jnp.array(n_e_ped_L),
  )


class UpdatePedestalTransitionStateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'mode': 'ADAPTIVE_SOURCE',
        'use_formation_model_with_adaptive_source': True,
        'formation_model': {'model_name': 'martin_scaling'},
        'P_LH_hysteresis_factor': _HYSTERESIS,
        'transition_time_width': _TRANSITION_WIDTH,
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
    self.runtime_params = self.step_fn.runtime_params_provider(t=0.0)
    self.models = self.step_fn._solver.models

  def _call_update(
      self,
      transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      P_SOL: float,
      t: float = 5.0,
  ) -> pedestal_transition_state_lib.PedestalTransitionState:
    """Calls _update_pedestal_transition_state with mocked P_SOL and P_LH."""
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(t))
    with mock.patch.object(
        step_function_processing.power_scaling_formation_model_lib,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        step_function_processing.scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      return step_function_processing._update_pedestal_transition_state(
          pedestal_transition_state=transition_state,
          runtime_params=runtime_params,
          geo=self.initial_state.geometry,
          core_profiles=self.initial_state.core_profiles,
          core_sources=self.initial_state.core_sources,
          models=self.models,
      )

  # ===== Confinement mode transitions =====

  def test_L_mode_stays_L_mode_when_P_SOL_below_P_LH(self):
    state = _make_transition_state(ConfinementMode.L_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 0.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.L_MODE)

  def test_L_mode_to_transitioning_to_H_mode_when_P_SOL_above_P_LH(self):
    state = _make_transition_state(ConfinementMode.L_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_H_MODE
    )

  def test_transitioning_to_H_mode_stays_when_incomplete(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH * 0.5  # Only half elapsed.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_H_MODE
    )

  def test_transitioning_to_H_mode_completes_to_H_mode(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH - 0.1  # Transition complete.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_stays_H_mode_within_hysteresis_band(self):
    state = _make_transition_state(ConfinementMode.H_MODE)
    # P_SOL between h*P_LH and P_LH.
    P_SOL = _P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0)
    new_state = self._call_update(state, P_SOL=P_SOL)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_stays_H_mode_when_P_SOL_above_P_LH(self):
    state = _make_transition_state(ConfinementMode.H_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_to_transitioning_to_L_mode_below_hysteresis(self):
    state = _make_transition_state(ConfinementMode.H_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_L_MODE
    )

  def test_transitioning_to_L_mode_stays_when_incomplete(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH * 0.5
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_L_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_L_MODE
    )

  def test_transitioning_to_L_mode_completes_to_L_mode(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH - 0.1  # Transition complete.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_L_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.L_MODE)

  # ===== Dither transitions =====

  def test_dither_LH_to_HL(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH * 0.3  # Only partially transitioned.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_L_MODE
    )

  def test_dither_HL_to_LH(self):
    t = 5.0
    start_time = t - _TRANSITION_WIDTH * 0.3
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_L_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    self.assertEqual(
        new_state.confinement_mode, ConfinementMode.TRANSITIONING_TO_H_MODE
    )

  # ===== transition_start_time =====

  def test_standard_L_to_H_sets_start_time_to_current_time(self):
    t = 5.0
    state = _make_transition_state(ConfinementMode.L_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    np.testing.assert_allclose(new_state.transition_start_time, t)

  def test_standard_H_to_L_sets_start_time_to_current_time(self):
    t = 5.0
    state = _make_transition_state(ConfinementMode.H_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    np.testing.assert_allclose(new_state.transition_start_time, t)

  def test_ongoing_LH_transition_preserves_start_time(self):
    t = 5.0
    original_start_time = 3.5
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE,
        start_time=original_start_time,
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    np.testing.assert_allclose(
        new_state.transition_start_time, original_start_time
    )

  def test_ongoing_HL_transition_preserves_start_time(self):
    t = 5.0
    original_start_time = 4.0
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_L_MODE,
        start_time=original_start_time,
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    np.testing.assert_allclose(
        new_state.transition_start_time, original_start_time
    )

  def test_dither_sets_mirrored_start_time(self):
    """Dither should set start_time = 2t - t0 - w for symmetric reversal."""
    t = 5.0
    t0 = 4.0  # Original transition started 1s ago.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE, start_time=t0
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5, t=t)
    expected_start = 2.0 * t - t0 - _TRANSITION_WIDTH
    np.testing.assert_allclose(new_state.transition_start_time, expected_start)

  def test_completed_transition_preserves_start_time(self):
    t = 5.0
    start_time = 2.0  # Transition complete: elapsed = 3.0 > width = 2.0.
    state = _make_transition_state(
        ConfinementMode.TRANSITIONING_TO_H_MODE, start_time=start_time
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5, t=t)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)
    # start_time is preserved (not reset to inf), but this is fine because
    # H_MODE doesn't use it.
    np.testing.assert_allclose(new_state.transition_start_time, start_time)

  # ===== L-mode pedestal values =====

  def test_L_to_H_captures_L_mode_values(self):
    """LH transition should capture current pedestal-top profile values."""
    state = _make_transition_state(
        ConfinementMode.L_MODE, T_i_ped_L=0.0, T_e_ped_L=0.0, n_e_ped_L=0.0
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    # Values should be updated from core_profiles (non-zero).
    self.assertNotEqual(new_state.T_i_ped_L_mode, 0.0)
    self.assertNotEqual(new_state.T_e_ped_L_mode, 0.0)
    self.assertNotEqual(new_state.n_e_ped_L_mode, 0.0)

  def test_non_L_to_H_preserves_L_mode_values(self):
    """Non LH transitions should keep existing L-mode values."""
    original_T_i = 0.5
    original_T_e = 0.4
    original_n_e = 0.5e19
    state = _make_transition_state(
        ConfinementMode.H_MODE,
        T_i_ped_L=original_T_i,
        T_e_ped_L=original_T_e,
        n_e_ped_L=original_n_e,
    )
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5)
    np.testing.assert_allclose(new_state.T_i_ped_L_mode, original_T_i)
    np.testing.assert_allclose(new_state.T_e_ped_L_mode, original_T_e)
    np.testing.assert_allclose(new_state.n_e_ped_L_mode, original_n_e)


class AdaptiveTransportTransitionStateTest(parameterized.TestCase):
  """Tests for the simplified ADAPTIVE_TRANSPORT state machine."""

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
        'formation_model': {'model_name': 'martin_scaling'},
        'P_LH_hysteresis_factor': _HYSTERESIS,
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
    self.runtime_params = self.step_fn.runtime_params_provider(t=0.0)
    self.models = self.step_fn._solver.models

  def _call_update(
      self,
      transition_state: pedestal_transition_state_lib.PedestalTransitionState,
      P_SOL: float,
      t: float = 5.0,
  ) -> pedestal_transition_state_lib.PedestalTransitionState:
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(t))
    with mock.patch.object(
        step_function_processing.power_scaling_formation_model_lib,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        step_function_processing.scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      return step_function_processing._update_pedestal_transition_state(
          pedestal_transition_state=transition_state,
          runtime_params=runtime_params,
          geo=self.initial_state.geometry,
          core_profiles=self.initial_state.core_profiles,
          core_sources=self.initial_state.core_sources,
          models=self.models,
      )

  def test_L_mode_stays_L_mode_below_threshold(self):
    state = _make_transition_state(ConfinementMode.L_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 0.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.L_MODE)

  def test_L_mode_to_H_mode_directly(self):
    """ADAPTIVE_TRANSPORT goes directly L→H, no TRANSITIONING state."""
    state = _make_transition_state(ConfinementMode.L_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_stays_H_mode_above_threshold(self):
    state = _make_transition_state(ConfinementMode.H_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_stays_in_hysteresis_band(self):
    """P_SOL between h*P_LH and P_LH should stay in H_MODE."""
    state = _make_transition_state(ConfinementMode.H_MODE)
    P_SOL = _P_LH * (_HYSTERESIS + (1.0 - _HYSTERESIS) / 2.0)
    new_state = self._call_update(state, P_SOL=P_SOL)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.H_MODE)

  def test_H_mode_to_L_mode_directly(self):
    """ADAPTIVE_TRANSPORT goes directly H→L, no TRANSITIONING state."""
    state = _make_transition_state(ConfinementMode.H_MODE)
    new_state = self._call_update(state, P_SOL=_P_LH * _HYSTERESIS * 0.5)
    self.assertEqual(new_state.confinement_mode, ConfinementMode.L_MODE)

  def test_no_L_mode_value_capture(self):
    """ADAPTIVE_TRANSPORT should preserve L-mode values unchanged."""
    original_T_i = 0.5
    state = _make_transition_state(
        ConfinementMode.L_MODE, T_i_ped_L=original_T_i
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    # L-mode values should be preserved (not updated from core_profiles).
    np.testing.assert_allclose(new_state.T_i_ped_L_mode, original_T_i)

  def test_no_transition_timer_updates(self):
    """ADAPTIVE_TRANSPORT should not update transition_start_time."""
    original_start = jnp.inf
    state = _make_transition_state(
        ConfinementMode.L_MODE, start_time=float(original_start)
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    np.testing.assert_allclose(new_state.transition_start_time, original_start)


if __name__ == '__main__':
  absltest.main()
