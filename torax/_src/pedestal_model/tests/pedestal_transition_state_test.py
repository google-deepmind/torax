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

"""Unit tests for pedestal_transition_state and PedestalModel.update_transition_state."""

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import power_scaling_formation_model
from torax._src.physics import scaling_laws
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name,protected-access
ConfinementMode = pedestal_transition_state.ConfinementMode

# Constants for test readability.
_P_LH = 10.0  # MW, mocked L-H threshold power.
_HYSTERESIS = 0.8  # P_LH_hysteresis_factor.
_TRANSITION_WIDTH = 2.0  # seconds.


def _make_transition_state(
    mode: ConfinementMode,
    start_time: float = -jnp.inf,
    T_i_ped_L: float = 0.5,
    T_e_ped_L: float = 0.4,
    n_e_ped_L: float = 0.5e19,
) -> pedestal_transition_state.PedestalTransitionState:
  """Helper to create a PedestalTransitionState with given values."""
  return pedestal_transition_state.PedestalTransitionState(
      confinement_mode=jnp.array(mode),
      transition_start_time=jnp.array(start_time),
      T_i_ped_L_mode=jnp.array(T_i_ped_L),
      T_e_ped_L_mode=jnp.array(T_e_ped_L),
      n_e_ped_L_mode=jnp.array(n_e_ped_L),
      pedestal_model_output=pedestal_model_output.PedestalModelOutput(
          rho_norm_ped_top=jnp.array(0.9),
          T_i_ped=4.5,
          T_e_ped=4.5,
          n_e_ped=0.62e20,
      ),
      previous_pedestal_model_output=pedestal_model_output.PedestalModelOutput(
          rho_norm_ped_top=jnp.inf,
          T_i_ped=0.0,
          T_e_ped=0.0,
          n_e_ped=0.0,
      ),
  )


class PedestalTransitionStateTest(parameterized.TestCase):

  def test_compute_ramp_fraction(self):
    transition_state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(0.0),
        T_e_ped_L_mode=jnp.array(0.0),
        n_e_ped_L_mode=jnp.array(0.0),
        confinement_mode=(
            pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE
        ),
        pedestal_model_output=pedestal_model_output.PedestalModelOutput.no_pedestal(),
        previous_pedestal_model_output=(
            pedestal_model_output.PedestalModelOutput.no_pedestal()
        ),
    )
    # transition_time_width = 1.0. Start at 1.0. Clip at both ends.
    self.assertEqual(transition_state._compute_ramp_fraction(0.5, 1.0), 0.0)
    self.assertEqual(transition_state._compute_ramp_fraction(1.0, 1.0), 0.0)
    self.assertEqual(transition_state._compute_ramp_fraction(1.5, 1.0), 0.5)
    self.assertEqual(transition_state._compute_ramp_fraction(2.0, 1.0), 1.0)
    self.assertEqual(transition_state._compute_ramp_fraction(2.5, 1.0), 1.0)
    # transition_time_width = 0.0 returns 1.0 with finite gradients.
    self.assertEqual(transition_state._compute_ramp_fraction(1.0, 0.0), 1.0)
    grad_w = jax.grad(
        lambda w: transition_state._compute_ramp_fraction(jnp.array(1.5), w)
    )(jnp.array(0.0))
    self.assertTrue(bool(jnp.isfinite(grad_w)))

  def test_apply_transition_ramp_scaling_l_to_h(self):
    l_mode_baseline = 1.0
    h_mode_target = 3.0

    transition_state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(l_mode_baseline),
        T_e_ped_L_mode=jnp.array(l_mode_baseline),
        n_e_ped_L_mode=jnp.array(l_mode_baseline),
        confinement_mode=(
            pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE
        ),
        pedestal_model_output=pedestal_model_output.PedestalModelOutput(
            T_i_ped=h_mode_target,
            T_e_ped=h_mode_target,
            n_e_ped=h_mode_target,
            rho_norm_ped_top=0.5,
        ),
        previous_pedestal_model_output=(
            pedestal_model_output.PedestalModelOutput.no_pedestal()
        ),
    )

    scaled_pedestal_model_output = (
        transition_state._apply_transition_ramp_scaling(
            ramp_fraction=0.5,
        )
    )

    # Expected: 1.0 + 0.5 * (3.0 - 1.0) = 2.0
    self.assertTrue(jnp.allclose(scaled_pedestal_model_output.T_i_ped, 2.0))
    self.assertTrue(jnp.allclose(scaled_pedestal_model_output.T_e_ped, 2.0))
    self.assertTrue(jnp.allclose(scaled_pedestal_model_output.n_e_ped, 2.0))

  def test_get_scaled_pedestal_model_output(self):
    l_mode_baseline = 1.0
    h_mode_target = 5.0

    transition_state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(2.0),
        T_i_ped_L_mode=jnp.array(l_mode_baseline),
        T_e_ped_L_mode=jnp.array(l_mode_baseline),
        n_e_ped_L_mode=jnp.array(l_mode_baseline),
        confinement_mode=(
            pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE
        ),
        pedestal_model_output=pedestal_model_output.PedestalModelOutput(
            T_i_ped=h_mode_target,
            T_e_ped=h_mode_target,
            n_e_ped=h_mode_target,
            rho_norm_ped_top=0.8,
        ),
        previous_pedestal_model_output=(
            pedestal_model_output.PedestalModelOutput.no_pedestal()
        ),
    )

    # At t=2.5 with width=2.0, ramp fraction is (2.5 - 2.0) / 2.0 = 0.25
    # Expected: 1.0 + 0.25 * (5.0 - 1.0) = 2.0
    scaled = transition_state._get_scaled_pedestal_model_output(
        t=2.5, transition_time_width=2.0
    )
    self.assertTrue(jnp.allclose(scaled.T_i_ped, 2.0))

  @parameterized.parameters(
      (
          pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT,
          pedestal_transition_state.ConfinementMode.H_MODE,
          False,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          pedestal_transition_state.ConfinementMode.L_MODE,
          False,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          pedestal_transition_state.ConfinementMode.H_MODE,
          True,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE,
          True,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_L_MODE,
          True,
      ),
  )
  def test_is_ibc_active(
      self,
      mode,
      confinement_mode,
      expected,
  ):
    output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.9,
        T_i_ped=2.0,
        T_e_ped=2.0,
        n_e_ped=1e19,
    )
    transition_state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(0.0),
        T_i_ped_L_mode=jnp.array(0.0),
        T_e_ped_L_mode=jnp.array(0.0),
        n_e_ped_L_mode=jnp.array(0.0),
        confinement_mode=confinement_mode,
        pedestal_model_output=output,
        previous_pedestal_model_output=output,
    )
    pedestal_params = mock.create_autospec(
        pedestal_runtime_params_lib.RuntimeParams,
        instance=True,
        mode=mode,
    )
    self.assertEqual(
        bool(transition_state.is_ibc_active(pedestal_params)), expected
    )


class UpdatePedestalTransitionStateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'mode': 'INTERNAL_BOUNDARY_CONDITION',
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
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    self.geo = torax_config.geometry.build_provider(0.0)
    self.runtime_params = provider(t=0.0)
    self.pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        source_models,
        neoclassical_model,
    )
    self.source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_models=source_models,
        explicit=True,
    )

  def _call_update(
      self,
      transition_state: pedestal_transition_state.PedestalTransitionState,
      P_SOL: float,
      t: float = 5.0,
  ) -> pedestal_transition_state.PedestalTransitionState:
    """Calls update_transition_state with mocked P_SOL and P_LH."""
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(t))
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      return self.pedestal_model.update_transition_state(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_transition_state=transition_state,
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
    # start_time is preserved (not reset to -inf), but this is fine because
    # H_MODE doesn't use it.
    np.testing.assert_allclose(new_state.transition_start_time, start_time)

  # ===== L-mode pedestal values =====

  def test_L_to_H_captures_L_mode_values(self):
    """LH transition should capture current pedestal-top profile values."""
    state = _make_transition_state(
        ConfinementMode.L_MODE, T_i_ped_L=0.0, T_e_ped_L=0.0, n_e_ped_L=0.0
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    ped_top_idx = jnp.argmin(
        jnp.abs(
            self.geo.rho_norm - state.pedestal_model_output.rho_norm_ped_top
        )
    )
    np.testing.assert_allclose(
        new_state.T_i_ped_L_mode,
        self.core_profiles.T_i.value[ped_top_idx],
    )
    np.testing.assert_allclose(
        new_state.T_e_ped_L_mode,
        self.core_profiles.T_e.value[ped_top_idx],
    )
    np.testing.assert_allclose(
        new_state.n_e_ped_L_mode,
        self.core_profiles.n_e.value[ped_top_idx],
    )

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

  def test_zero_transition_time_width_steps_directly_and_captures_L_mode_values(
      self,
  ):
    """When transition_time_width == 0.0, L->H and H->L step directly."""
    zero_width_pedestal = dataclasses.replace(
        self.runtime_params.pedestal, transition_time_width=jnp.array(0.0)
    )
    runtime_params = dataclasses.replace(
        self.runtime_params, t=jnp.array(5.0), pedestal=zero_width_pedestal
    )
    state = _make_transition_state(
        ConfinementMode.L_MODE, T_i_ped_L=0.0, T_e_ped_L=0.0, n_e_ped_L=0.0
    )
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(_P_LH * 1.5),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      h_state = self.pedestal_model.update_transition_state(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_transition_state=state,
      )
    self.assertEqual(h_state.confinement_mode, ConfinementMode.H_MODE)
    ped_top_idx = jnp.argmin(
        jnp.abs(
            self.geo.rho_norm - state.pedestal_model_output.rho_norm_ped_top
        )
    )
    np.testing.assert_allclose(
        h_state.T_i_ped_L_mode,
        self.core_profiles.T_i.value[ped_top_idx],
    )

    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(_P_LH * _HYSTERESIS * 0.5),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      l_state = self.pedestal_model.update_transition_state(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_transition_state=h_state,
      )
    self.assertEqual(l_state.confinement_mode, ConfinementMode.L_MODE)

  def test_initialize_transition_state_two_pass_previous_output(self):
    """Verifies two-pass evaluation for models using previous width."""
    pedestal_cls = type(self.pedestal_model)

    def width_dependent_impl(
        model_self,
        runtime_params,
        geo,
        core_profiles,
        transition_state,
    ):
      del model_self, runtime_params, geo, core_profiles
      prev_rho = (
          transition_state.previous_pedestal_model_output.rho_norm_ped_top
      )
      # Simulate EPED-NN behavior: first call has prev_rho == inf (uses fallback
      # width 0.9 -> T_ped=4.0, computes new width 0.85); second call has
      # prev_rho == 0.85 -> T_ped=6.0.
      t_ped = jnp.where(jnp.isinf(prev_rho), 4.0, 6.0)
      return pedestal_model_output.PedestalModelOutput(
          rho_norm_ped_top=jnp.array(0.85),
          T_i_ped=t_ped,
          T_e_ped=t_ped,
          n_e_ped=jnp.array(0.7e20),
      )

    with mock.patch.object(
        pedestal_cls,
        '_call_implementation',
        autospec=True,
        side_effect=width_dependent_impl,
    ):
      init_ts = self.pedestal_model.initialize_transition_state(
          self.runtime_params,
          self.geo,
          self.core_profiles,
          self.source_profiles,
      )
    np.testing.assert_allclose(
        init_ts.pedestal_model_output.rho_norm_ped_top, 0.85
    )
    np.testing.assert_allclose(init_ts.pedestal_model_output.T_i_ped, 6.0)
    np.testing.assert_allclose(
        init_ts.previous_pedestal_model_output.rho_norm_ped_top, 0.85
    )
    np.testing.assert_allclose(
        init_ts.previous_pedestal_model_output.T_i_ped, 6.0
    )

  def test_initialize_transition_state_L_mode_baselines(self):
    """Verifies L-mode baselines when starting in L-mode vs H-mode."""
    with self.subTest('starting_in_L_mode'):
      with mock.patch.object(
          power_scaling_formation_model,
          'calculate_P_SOL_total',
          return_value=jnp.array(_P_LH * 0.5),
      ), mock.patch.object(
          scaling_laws,
          'calculate_P_LH',
          return_value=(jnp.array(_P_LH), None),
      ):
        l_init_ts = self.pedestal_model.initialize_transition_state(
            self.runtime_params,
            self.geo,
            self.core_profiles,
            self.source_profiles,
        )
      ped_top_idx = jnp.argmin(
          jnp.abs(
              self.geo.rho_norm
              - l_init_ts.pedestal_model_output.rho_norm_ped_top
          )
      )
      self.assertEqual(l_init_ts.confinement_mode, ConfinementMode.L_MODE)
      np.testing.assert_allclose(
          l_init_ts.T_i_ped_L_mode,
          self.core_profiles.T_i.value[ped_top_idx],
      )
      np.testing.assert_allclose(
          l_init_ts.T_e_ped_L_mode,
          self.core_profiles.T_e.value[ped_top_idx],
      )
      np.testing.assert_allclose(
          l_init_ts.n_e_ped_L_mode,
          self.core_profiles.n_e.value[ped_top_idx],
      )

    with self.subTest('starting_in_H_mode_uses_2x_lcfs_fallback'):
      with mock.patch.object(
          power_scaling_formation_model,
          'calculate_P_SOL_total',
          return_value=jnp.array(_P_LH * 1.5),
      ), mock.patch.object(
          scaling_laws,
          'calculate_P_LH',
          return_value=(jnp.array(_P_LH), None),
      ):
        h_init_ts = self.pedestal_model.initialize_transition_state(
            self.runtime_params,
            self.geo,
            self.core_profiles,
            self.source_profiles,
        )
      self.assertEqual(h_init_ts.confinement_mode, ConfinementMode.H_MODE)
      np.testing.assert_allclose(
          h_init_ts.T_i_ped_L_mode,
          2.0 * self.core_profiles.T_i.face_value()[-1],
      )
      np.testing.assert_allclose(
          h_init_ts.T_e_ped_L_mode,
          2.0 * self.core_profiles.T_e.face_value()[-1],
      )
      np.testing.assert_allclose(
          h_init_ts.n_e_ped_L_mode,
          2.0 * self.core_profiles.n_e.face_value()[-1],
      )


class AdaptiveTransportTransitionStateTest(parameterized.TestCase):
  """Tests for the simplified ADAPTIVE_TRANSPORT state machine."""

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
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
    torax_config = model_config.ToraxConfig.from_dict(config)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    source_models = torax_config.sources.build_models()
    neoclassical_model = torax_config.neoclassical.build_model()
    self.geo = torax_config.geometry.build_provider(0.0)
    self.runtime_params = provider(t=0.0)
    self.pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.core_profiles = initialization.initial_core_profiles(
        self.runtime_params,
        self.geo,
        source_models,
        neoclassical_model,
    )
    self.source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
        source_models=source_models,
        explicit=True,
    )

  def _call_update(
      self,
      transition_state: pedestal_transition_state.PedestalTransitionState,
      P_SOL: float,
      t: float = 5.0,
  ) -> pedestal_transition_state.PedestalTransitionState:
    runtime_params = dataclasses.replace(self.runtime_params, t=jnp.array(t))
    with mock.patch.object(
        power_scaling_formation_model,
        'calculate_P_SOL_total',
        return_value=jnp.array(P_SOL),
    ), mock.patch.object(
        scaling_laws,
        'calculate_P_LH',
        return_value=(jnp.array(_P_LH), None),
    ):
      return self.pedestal_model.update_transition_state(
          runtime_params=runtime_params,
          geo=self.geo,
          core_profiles=self.core_profiles,
          source_profiles=self.source_profiles,
          pedestal_transition_state=transition_state,
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
    original_start = -jnp.inf
    state = _make_transition_state(
        ConfinementMode.L_MODE, start_time=float(original_start)
    )
    new_state = self._call_update(state, P_SOL=_P_LH * 1.5)
    np.testing.assert_allclose(new_state.transition_start_time, original_start)


if __name__ == '__main__':
  absltest.main()
