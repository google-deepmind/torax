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

"""Unit tests for pedestal_transition_state."""

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name,protected-access


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
        pedestal_model_output=pedestal_model_output.PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
        previous_pedestal_model_output=(
            pedestal_model_output.PedestalModelOutput(
                rho_norm_ped_top=jnp.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
            )
        ),
    )
    # transition_time_width = 1.0. Start at 1.0. Clip at both ends.
    self.assertEqual(transition_state._compute_ramp_fraction(0.5, 1.0), 0.0)
    self.assertEqual(transition_state._compute_ramp_fraction(1.0, 1.0), 0.0)
    self.assertEqual(transition_state._compute_ramp_fraction(1.5, 1.0), 0.5)
    self.assertEqual(transition_state._compute_ramp_fraction(2.0, 1.0), 1.0)
    self.assertEqual(transition_state._compute_ramp_fraction(2.5, 1.0), 1.0)

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
            pedestal_model_output.PedestalModelOutput(
                rho_norm_ped_top=jnp.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
            )
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
            pedestal_model_output.PedestalModelOutput(
                rho_norm_ped_top=jnp.inf,
                T_i_ped=0.0,
                T_e_ped=0.0,
                n_e_ped=0.0,
            )
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
          True,
          True,
          pedestal_transition_state.ConfinementMode.H_MODE,
          False,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          False,
          True,
          pedestal_transition_state.ConfinementMode.H_MODE,
          False,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          True,
          True,
          pedestal_transition_state.ConfinementMode.L_MODE,
          False,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          True,
          True,
          pedestal_transition_state.ConfinementMode.H_MODE,
          True,
      ),
      (
          pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION,
          True,
          False,
          pedestal_transition_state.ConfinementMode.L_MODE,
          True,
      ),
  )
  def test_is_ibc_active(
      self,
      mode,
      set_pedestal,
      use_formation_model,
      confinement_mode,
      expected,
  ):
    output = pedestal_model_output.PedestalModelOutput(
        rho_norm_ped_top=0.9, T_i_ped=2.0, T_e_ped=2.0, n_e_ped=1e19
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
        set_pedestal=set_pedestal,
        use_formation_model_with_internal_boundary_condition=use_formation_model,
    )
    self.assertEqual(
        bool(transition_state.is_ibc_active(pedestal_params)), expected
    )


if __name__ == '__main__':
  absltest.main()
