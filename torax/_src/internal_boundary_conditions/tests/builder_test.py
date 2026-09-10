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

"""Unit tests for internal boundary conditions builder."""

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.internal_boundary_conditions import builder as ibc_builder
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.torax_pydantic import model_config


class TransitionCalculationsTest(parameterized.TestCase):

  def test_compute_ramp_fraction(self):
    state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(0.0),
        T_e_ped_L_mode=jnp.array(0.0),
        n_e_ped_L_mode=jnp.array(0.0),
        confinement_mode=pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE,
        pedestal_model_output=pedestal_model_output_lib.PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
        previous_pedestal_model_output=pedestal_model_output_lib.PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
    )
    # transition_time_width = 1.0. Start at 1.0. Clip at both ends.
    self.assertEqual(
        ibc_builder._compute_ramp_fraction(state, 1.0, 0.5),
        0.0,
    )  # t < start
    self.assertEqual(
        ibc_builder._compute_ramp_fraction(state, 1.0, 1.0),
        0.0,
    )  # t = start
    self.assertEqual(
        ibc_builder._compute_ramp_fraction(state, 1.0, 1.5),
        0.5,
    )  # t = start + 0.5
    self.assertEqual(
        ibc_builder._compute_ramp_fraction(state, 1.0, 2.0),
        1.0,
    )  # t = start + 1.0
    self.assertEqual(
        ibc_builder._compute_ramp_fraction(state, 1.0, 2.5),
        1.0,
    )  # t = start + 1.5

  def test_apply_transition_ramp_scaling_l_to_h(self):
    l_mode_baseline = 1.0
    h_mode_target = 3.0

    state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(l_mode_baseline),
        T_e_ped_L_mode=jnp.array(l_mode_baseline),
        n_e_ped_L_mode=jnp.array(l_mode_baseline),
        confinement_mode=pedestal_transition_state.ConfinementMode.TRANSITIONING_TO_H_MODE,
        pedestal_model_output=pedestal_model_output_lib.PedestalModelOutput(
            T_i_ped=h_mode_target,
            T_e_ped=h_mode_target,
            n_e_ped=h_mode_target,
            rho_norm_ped_top=0.5,
        ),
        previous_pedestal_model_output=pedestal_model_output_lib.PedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        ),
    )

    scaled_pedestal_model_output = ibc_builder._apply_transition_ramp_scaling(  # pyrefly: ignore[bad-argument-type]
        pedestal_transition_state=state,
        ramp_fraction=0.5,
    )

    # Expected: 1.0 + 0.5 * (3.0 - 1.0) = 2.0
    self.assertTrue(jnp.allclose(scaled_pedestal_model_output.T_i_ped, 2.0))


class BuilderTest(parameterized.TestCase):

  def test_build_internal_boundary_conditions_mutual_exclusion(self):
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(),
            plasma_composition=dict(),
            profile_conditions=dict(),
            internal_boundary_conditions=dict(
                T_e={0.0: {(0.0, 0.3): 10.0}},
            ),
            geometry=dict(geometry_type='circular', n_rho=20),
            pedestal=dict(
                set_pedestal=True,
                explicit_pedestal=False,
                model_name='set_T_ped_n_ped',
                rho_norm_ped_top=0.9,
                T_e_ped=2.0,
                mode='INTERNAL_BOUNDARY_CONDITION',
                use_formation_model_with_internal_boundary_condition=True,
            ),
            sources=dict(),
            solver=dict(use_predictor_corrector=False),
            transport=dict(),
            time_step_calculator=dict(),
        )
    )
    models = torax_config.build_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=models.source_models,
        neoclassical_models=models.neoclassical_models,
    )

    inboard_mask = geo.rho_norm <= 0.3
    ped_cell = jnp.argmin(jnp.abs(geo.rho_norm - 0.9))

    with self.subTest('l_mode_config_ibc_active_pedestal_disabled'):
      l_mode_state = (
          pedestal_transition_state.PedestalTransitionState.empty_L_mode()
      )
      built_ibc_l = ibc_builder.build_internal_boundary_conditions(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          pedestal_transition_state=l_mode_state,
      )
      # Configured IBC pins cell values in rho_norm <= 0.3 to 10.0 keV
      self.assertTrue(jnp.any(built_ibc_l.T_e[inboard_mask] == 10.0))
      # Pedestal location (0.9) is unpinned (0.0) in L-mode
      self.assertEqual(built_ibc_l.T_e[ped_cell], 0.0)

    with self.subTest('h_mode_pedestal_active_config_ibc_disabled'):
      h_mode_state = dataclasses.replace(
          l_mode_state,
          confinement_mode=pedestal_transition_state.ConfinementMode.H_MODE,
          pedestal_model_output=pedestal_model_output_lib.PedestalModelOutput(
              T_i_ped=2.0,
              T_e_ped=2.0,
              n_e_ped=1e19,
              rho_norm_ped_top=0.9,
          ),
      )
      built_ibc_h = ibc_builder.build_internal_boundary_conditions(
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          pedestal_transition_state=h_mode_state,
      )
      # Configured IBC region is completely turned off (0.0) in H-mode
      self.assertTrue(jnp.all(built_ibc_h.T_e[inboard_mask] == 0.0))
      # Pedestal top cell is pinned to pedestal height (2.0)
      self.assertEqual(built_ibc_h.T_e[ped_cell], 2.0)


if __name__ == '__main__':
  absltest.main()
