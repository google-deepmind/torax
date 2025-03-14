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

from absl.testing import absltest
from absl.testing import parameterized
from torax.config import build_sim
from torax.pedestal_model import set_tped_nped
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import runtime_params as source_runtime_params_lib
from torax.stepper import linear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import qlknn_transport_model


class BuildSimTest(parameterized.TestCase):

  def test_build_sim_raises_error_with_missing_keys(self):
    with self.assertRaises(ValueError):
      build_sim.build_sim_from_config({})

  def test_build_sim_with_full_config(self):
    """Tests building Sim with a more complete config."""
    sim = build_sim.build_sim_from_config(
        dict(
            runtime_params=dict(
                plasma_composition=dict(
                    Ai_override=0.1,
                ),
                profile_conditions=dict(
                    ne_is_fGW=False,
                ),
            ),
            geometry=dict(
                geometry_type='circular',
                n_rho=5,
            ),
            sources=dict(
                pellet_source=dict(
                    mode='ZERO',
                ),
            ),
            transport=dict(
                transport_model='qlknn',
                include_ITG=False,
            ),
            stepper=dict(
                stepper_type='linear',
                theta_imp=0.5,
            ),
            pedestal=dict(
                rho_norm_ped_top=0.6,
                Teped=5.9,
                Tiped=4.8,
                neped=0.8,
                neped_is_fGW=True,
            ),
            time_step_calculator=dict(
                calculator_type='fixed',
            ),
        )
    )
    dynamic_runtime_params_slice = sim.dynamic_runtime_params_slice_provider(
        t=sim.initial_state.t,
    )
    with self.subTest('runtime_params'):
      self.assertEqual(
          dynamic_runtime_params_slice.plasma_composition.main_ion.avg_A, 0.1
      )
      self.assertEqual(
          dynamic_runtime_params_slice.profile_conditions.ne_is_fGW,
          False,
      )
    with self.subTest('geometry'):
      geo = sim.geometry_provider(sim.initial_state.t)
      self.assertEqual(geo.torax_mesh.nx, 5)
    with self.subTest('sources'):
      self.assertEqual(
          sim.static_runtime_params_slice.sources['pellet_source'].mode,
          source_runtime_params_lib.Mode.ZERO.value,
      )
    with self.subTest('transport'):
      self.assertIsInstance(
          sim.transport_model, qlknn_transport_model.QLKNNTransportModel
      )
      self.assertIsInstance(
          dynamic_runtime_params_slice.transport,
          qlknn_transport_model.DynamicRuntimeParams,
      )
      # pytype: disable=attribute-error
      self.assertEqual(
          dynamic_runtime_params_slice.transport.include_ITG, False
      )
      # pytype: enable=attribute-error
    with self.subTest('stepper'):
      self.assertIsInstance(sim.stepper, linear_theta_method.LinearThetaMethod)
      self.assertEqual(sim.static_runtime_params_slice.stepper.theta_imp, 0.5)
    with self.subTest('pedestal'):
      self.assertIsInstance(
          dynamic_runtime_params_slice.pedestal,
          set_tped_nped.DynamicRuntimeParams,
      )
      # pytype: disable=attribute-error
      self.assertEqual(
          dynamic_runtime_params_slice.pedestal.rho_norm_ped_top, 0.6
      )
      self.assertEqual(dynamic_runtime_params_slice.pedestal.Teped, 5.9)
      self.assertEqual(dynamic_runtime_params_slice.pedestal.Tiped, 4.8)
      self.assertEqual(dynamic_runtime_params_slice.pedestal.neped, 0.8)
      self.assertEqual(dynamic_runtime_params_slice.pedestal.neped_is_fGW, True)
      # pytype: enable=attribute-error
    with self.subTest('time_step_calculator'):
      self.assertIsInstance(
          sim.time_step_calculator,
          fixed_time_step_calculator.FixedTimeStepCalculator,
      )

  def test_empty_source_config_only_has_defaults_turned_off(self):
    """Tests that an empty source config has all sources turned off."""
    sources = sources_pydantic_model.Sources.from_dict({})
    self.assertEqual(
        sources.source_model_config['j_bootstrap'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.source_model_config['generic_current_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.source_model_config['qei_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertLen(sources.source_model_config, 3)

  def test_unknown_stepper_type_raises_error(self):
    with self.assertRaises(ValueError):
      stepper_pydantic_model.Stepper.from_dict({'stepper_type': 'foo'})


if __name__ == '__main__':
  absltest.main()
