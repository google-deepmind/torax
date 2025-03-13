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
import numpy as np
from torax.config import build_runtime_params
from torax.config import build_sim
from torax.config import runtime_params as runtime_params_lib
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import set_tped_nped
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import runtime_params as source_runtime_params_lib
from torax.stepper import linear_theta_method
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import constant as constant_transport
from torax.transport_model import critical_gradient as critical_gradient_transport
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
                qlknn_params=dict(
                    include_ITG=False,
                ),
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

  def test_build_runtime_params_from_empty_config(self):
    """An empty config should return all defaults."""
    runtime_params = build_sim.build_runtime_params_from_config({})
    defaults = runtime_params_lib.GeneralRuntimeParams()
    self.assertEqual(runtime_params, defaults)

  def test_build_runtime_params_raises_error_with_incorrect_args(self):
    """If an incorrect key is provided, an error should be raised."""
    with self.assertRaises(KeyError):
      build_sim.build_runtime_params_from_config({'incorrect_key': 'value'})

  def test_general_runtime_params_with_time_dependent_args(self):
    """Tests that we can build all types of attributes in the runtime params."""
    runtime_params = build_sim.build_runtime_params_from_config({
        'plasma_composition': {
            'main_ion': 'D',
            'Zeff': {
                0: {0: 0.1, 1: 0.1},
                1: {0: 0.2, 1: 0.2},
                2: {0: 0.3, 1: 0.3},
            },  # time-dependent with constant radial profile.
        },
        'profile_conditions': {
            'ne_is_fGW': False,  # scalar fields.
            'Ip_tot': {0: 0.2, 1: 0.4, 2: 0.6},  # time-dependent.
        },
        'numerics': {
            # Designate the interpolation mode, as well, setting to "step".
            'resistivity_mult': ({0: 0.3, 1: 0.6, 2: 0.9}, 'step'),
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(runtime_params.plasma_composition.main_ion, 'D')
    self.assertEqual(runtime_params.profile_conditions.ne_is_fGW, False)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=1.5,
        )
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.plasma_composition.Zeff, 0.25
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ip_tot, 0.5
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.numerics.resistivity_mult, 0.6
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

  def test_missing_transport_model_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_transport_model_builder_from_config({})

  @parameterized.named_parameters(
      dict(
          testcase_name='constant',
          name='constant',
          expected_type=constant_transport.ConstantTransportModel,
      ),
      dict(
          testcase_name='critical_gradient',
          name='CGM',
          expected_type=critical_gradient_transport.CriticalGradientTransportModel,
      ),
      dict(
          testcase_name='qlknn',
          name='qlknn',
          expected_type=qlknn_transport_model.QLKNNTransportModel,
      ),
  )
  def test_build_transport_models(self, name, expected_type):
    """Tests that we can build a transport model from the config."""
    transport_model_builder = (
        build_sim.build_transport_model_builder_from_config({
            'transport_model': name,
            'chimin': 1.23,
            'constant_params': {
                'chii_const': 4.56,
            },
            'cgm_params': {
                'alpha': 7.89,
            },
            'qlknn_params': {
                'coll_mult': 10.11,
            },
        })
    )
    transport_model = transport_model_builder()
    self.assertIsInstance(transport_model, expected_type)
    self.assertEqual(transport_model_builder.runtime_params.chimin, 1.23)
    if name == 'constant':
      assert isinstance(
          transport_model_builder.runtime_params,
          constant_transport.RuntimeParams,
      )
      self.assertEqual(transport_model_builder.runtime_params.chii_const, 4.56)
    elif name == 'CGM':
      assert isinstance(
          transport_model_builder.runtime_params,
          critical_gradient_transport.RuntimeParams,
      )
      self.assertEqual(transport_model_builder.runtime_params.alpha, 7.89)
    elif name == 'qlknn':
      assert isinstance(
          transport_model_builder.runtime_params,
          qlknn_transport_model.RuntimeParams,
      )
      self.assertEqual(transport_model_builder.runtime_params.coll_mult, 10.11)
    else:
      self.fail(f'Unknown transport model: {name}')

  def test_unknown_stepper_type_raises_error(self):
    with self.assertRaises(ValueError):
      stepper_pydantic_model.Stepper.from_dict({'stepper_type': 'foo'})

  def test_missing_time_step_calculator_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_time_step_calculator_from_config({})

  def test_unknown_time_step_calculator_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_time_step_calculator_from_config({'calculator_type': 'x'})

  @parameterized.named_parameters(
      dict(
          testcase_name='fixed',
          calculator_type='fixed',
          expected_type=fixed_time_step_calculator.FixedTimeStepCalculator,
      ),
      dict(
          testcase_name='chi',
          calculator_type='chi',
          expected_type=chi_time_step_calculator.ChiTimeStepCalculator,
      ),
  )
  def test_build_time_step_calculator_from_config(
      self, calculator_type, expected_type
  ):
    """Builds a time step calculator from the config."""
    time_stepper = build_sim.build_time_step_calculator_from_config(
        calculator_type
    )
    self.assertIsInstance(time_stepper, expected_type)


if __name__ == '__main__':
  absltest.main()
