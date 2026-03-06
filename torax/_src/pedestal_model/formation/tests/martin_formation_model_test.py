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
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.pedestal_model.formation import martin_formation_model
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class MartinFormationModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    # Switch to use the Martin formation model.
    config['pedestal'] = {
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
        'formation_model': {'model_name': 'martin'},
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
    step_fn = run_simulation.make_step_fn(self.torax_config)
    self.initial_state, self.initial_post_processed_outputs = (
        initial_state.get_initial_state_and_post_processed_outputs(step_fn)
    )
    self.runtime_params = step_fn.runtime_params_provider(t=0.0)

  def test_calculate_P_SOL_total(self):
    P_SOL_total = martin_formation_model._calculate_P_SOL_total(
        self.initial_state.core_profiles.internal_plasma_energy,
        self.initial_state.core_sources,
        self.initial_state.geometry,
    )

    np.testing.assert_allclose(
        P_SOL_total, self.initial_post_processed_outputs.P_SOL_total
    )

  @parameterized.named_parameters(
      dict(
          # If P_sol >> P_LH, we expect the suppression multiplier to be
          # base_multiplier.
          testcase_name='above_threshold',
          power=1e6,
          expected_multiplier=1e-6,
      ),
      dict(
          # If P_sol << P_LH, we expect the suppression multiplier to be 1.0
          # (no suppression).
          testcase_name='below_threshold',
          # We set the aux power to be negative (equivalent to a heat sink)
          # to make sure that P_sol < P_LH.
          power=-1e6,
          expected_multiplier=1.0,
      ),
  )
  def test_martin_formation_model_suppression(self, power, expected_multiplier):
    formation_model = martin_formation_model.MartinFormationModel()

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
              f' {expected_multiplier}.'
          ),
      )


if __name__ == '__main__':
  absltest.main()
