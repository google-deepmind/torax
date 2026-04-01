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

import copy
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.fvm import calc_coeffs
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config


class CalcCoeffsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=4, theta_implicit=0, set_pedestal=False),
      dict(num_cells=4, theta_implicit=0, set_pedestal=True),
      dict(num_cells=4, theta_implicit=0.5, set_pedestal=False),
      dict(num_cells=4, theta_implicit=0.5, set_pedestal=True),
  ])
  def test_calc_coeffs_smoke_test(
      self, num_cells, theta_implicit, set_pedestal
  ):
    sources_config = default_sources.get_default_source_config()
    sources_config['ei_exchange']['Qei_multiplier'] = 0.0
    sources_config['generic_heat']['P_total'] = 0.0
    sources_config['fusion']['mode'] = 'ZERO'
    sources_config['ohmic']['mode'] = 'ZERO'
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(
                set_pedestal=set_pedestal, model_name='set_T_ped_n_ped'
            ),
            sources=sources_config,
            solver=dict(
                use_predictor_corrector=False, theta_implicit=theta_implicit
            ),
            transport=dict(model_name='constant', chi_min=0, chi_i=1),
            time_step_calculator=dict(),
        )
    )
    physics_models = torax_config.build_physics_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    evolving_names = tuple(['T_i'])
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        source_models=physics_models.source_models,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        neoclassical_models=physics_models.neoclassical_models,
        explicit=True,
    )
    calc_coeffs.calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        physics_models=physics_models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )

  def test_calc_coeffs_hash(self):
    def create_coeffs_callback(
        torax_config: model_config.ToraxConfig,
    ) -> calc_coeffs.CoeffsCallback:
      physics_models = torax_config.build_physics_models()
      evolving_names = tuple(['T_i'])
      return calc_coeffs.CoeffsCallback(
          physics_models=physics_models,
          evolving_names=evolving_names,
      )

    torax_config = {
        'pedestal': {},
        'transport': {},
        'solver': {},
        'profile_conditions': {},
        'numerics': {},
        'sources': {},
        'geometry': {'geometry_type': 'circular'},
        'plasma_composition': {},
    }
    torax_config_with_pedestal = copy.deepcopy(torax_config)
    torax_config_with_pedestal['pedestal'] = {'model_name': 'set_T_ped_n_ped'}

    with self.subTest('same_coeffs_callback_hash_equal'):
      self.assertEqual(
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
      )
    with self.subTest('same_coeffs_callback_equal'):
      self.assertEqual(
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
      )
    with self.subTest('different_coeffs_callback_hash_not_equal'):
      self.assertNotEqual(
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config_with_pedestal)
              )
          ),
          hash(
              create_coeffs_callback(
                  model_config.ToraxConfig.from_dict(torax_config)
              )
          ),
      )
    with self.subTest('different_coeffs_callback_not_equal'):
      self.assertNotEqual(
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config)
          ),
          create_coeffs_callback(
              model_config.ToraxConfig.from_dict(torax_config_with_pedestal)
          ),
      )


class TransitionCalculationsTest(parameterized.TestCase):

  def test_pedestal_transition_state_initial_state(self):
    state = pedestal_transition_state.PedestalTransitionState.initial_state()
    self.assertTrue(jnp.isneginf(state.transition_start_time))
    self.assertEqual(state.T_i_ped_L_mode, 0.0)
    self.assertFalse(state.in_H_mode)

  def test_compute_ramp_fraction_ramp(self):
    state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(0.0),
        T_e_ped_L_mode=jnp.array(0.0),
        n_e_ped_L_mode=jnp.array(0.0),
        in_H_mode=jnp.array(True),
    )
    # transition_time_width = 1.0. Start at 1.0.
    # Clip at both ends
    self.assertEqual(
        calc_coeffs._compute_ramp_fraction(state, 1.0, 0.5), 0.0
    )  # t < start
    self.assertEqual(
        calc_coeffs._compute_ramp_fraction(state, 1.0, 1.0), 0.0
    )  # t = start
    self.assertEqual(
        calc_coeffs._compute_ramp_fraction(state, 1.0, 1.5), 0.5
    )  # t = start + 0.5
    self.assertEqual(
        calc_coeffs._compute_ramp_fraction(state, 1.0, 2.0), 1.0
    )  # t = start + 1.0
    self.assertEqual(
        calc_coeffs._compute_ramp_fraction(state, 1.0, 2.5), 1.0
    )  # t = start + 1.5

  def test_apply_transition_ramp_scaling_l_to_h(self):
    l_mode_baseline = 1.0
    h_mode_target = 3.0

    state = pedestal_transition_state.PedestalTransitionState(
        transition_start_time=jnp.array(1.0),
        T_i_ped_L_mode=jnp.array(l_mode_baseline),
        T_e_ped_L_mode=jnp.array(l_mode_baseline),
        n_e_ped_L_mode=jnp.array(l_mode_baseline),
        in_H_mode=jnp.array(True),  # L -> H
    )

    pedestal_model_output = pedestal_model_output_lib.PedestalModelOutput(
        T_i_ped=h_mode_target,
        T_e_ped=h_mode_target,
        n_e_ped=h_mode_target,
        rho_norm_ped_top=0.5,
        rho_norm_ped_top_idx=1,
        transport_multipliers=mock.Mock(),
    )

    scaled_pedestal_model_output = calc_coeffs._apply_transition_ramp_scaling(  # pytype: disable=wrong-arg-types
        pedestal_model_output=pedestal_model_output,
        pedestal_transition_state=state,
        ramp_fraction=0.5,
    )

    # Expected: 1.0 + 0.5 * (3.0 - 1.0) = 2.0
    self.assertTrue(jnp.allclose(scaled_pedestal_model_output.T_i_ped, 2.0))


if __name__ == '__main__':
  absltest.main()
