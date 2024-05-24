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

"""Unit tests for torax.config.build_sim."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import geometry
from torax.config import build_sim
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.sources import formula_config
from torax.sources import formulas
from torax.sources import runtime_params as source_runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.time_step_calculator import array_time_step_calculator
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import constant as constant_transport
from torax.transport_model import critical_gradient as critical_gradient_transport
from torax.transport_model import qlknn_wrapper


class BuildSimTest(parameterized.TestCase):
  """Unit tests for the `torax.config.build_sim` module."""

  def test_build_sim_raises_error_with_missing_keys(self):
    with self.assertRaises(ValueError):
      build_sim.build_sim_from_config({})

  def test_build_sim_with_full_config(self):
    """Tests building Sim with a more complete config."""
    sim = build_sim.build_sim_from_config(
        dict(
            runtime_params=dict(
                plasma_composition=dict(
                    Ai=0.1,
                ),
                profile_conditions=dict(
                    nbar_is_fGW=False,
                ),
                numerics=dict(
                    q_correction_factor=0.2,
                ),
            ),
            geometry=dict(
                geometry_type='circular',
                nr=5,
            ),
            sources=dict(
                pellet_source=dict(
                    mode='zero',
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
            time_step_calculator=dict(
                calculator_type='fixed',
            ),
        )
    )
    dynamic_runtime_params_slice = sim.dynamic_runtime_params_slice_provider(
        sim.initial_state.t
    )
    with self.subTest('runtime_params'):
      self.assertEqual(dynamic_runtime_params_slice.plasma_composition.Ai, 0.1)
      self.assertEqual(
          dynamic_runtime_params_slice.profile_conditions.nbar_is_fGW,
          False,
      )
      self.assertEqual(
          dynamic_runtime_params_slice.numerics.q_correction_factor,
          0.2,
      )
    with self.subTest('geometry'):
      geo = sim.geometry_provider(sim.initial_state)
      self.assertIsInstance(geo, geometry.CircularGeometry)
      self.assertEqual(geo.mesh.nx, 5)
    with self.subTest('sources'):
      self.assertEqual(
          sim.source_models.sources['pellet_source'].runtime_params.mode,
          source_runtime_params_lib.Mode.ZERO,
      )
    with self.subTest('transport'):
      self.assertIsInstance(
          sim.transport_model, qlknn_wrapper.QLKNNTransportModel
      )
      self.assertIsInstance(
          dynamic_runtime_params_slice.transport,
          qlknn_wrapper.DynamicRuntimeParams,
      )
      # pytype: disable=attribute-error
      self.assertEqual(
          dynamic_runtime_params_slice.transport.include_ITG, False
      )
      # pytype: enable=attribute-error
    with self.subTest('stepper'):
      self.assertIsInstance(sim.stepper, linear_theta_method.LinearThetaMethod)
      self.assertEqual(sim.static_runtime_params_slice.stepper.theta_imp, 0.5)
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
            'Ai': 0.1,  # scalar fields.
            'Zeff': {0: 0.1, 1: 0.2, 2: 0.3},  # time-dependent.
        },
        'profile_conditions': {
            'nbar_is_fGW': False,  # scalar fields.
            'Ip': {0: 0.2, 1: 0.4, 2: 0.6},  # time-dependent.
        },
        'numerics': {
            'q_correction_factor': 0.2,  # scalar fields.
            # Designate the interpolation mode, as well, setting to "step".
            'resistivity_mult': ({0: 0.3, 1: 0.6, 2: 0.9}, 'step'),
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(runtime_params.plasma_composition.Ai, 0.1)
    self.assertEqual(runtime_params.profile_conditions.nbar_is_fGW, False)
    self.assertEqual(runtime_params.numerics.q_correction_factor, 0.2)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params, t=1.5
        )
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.plasma_composition.Zeff, 0.25
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ip, 0.5
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.numerics.resistivity_mult, 0.6
    )

  def test_missing_geometry_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_geometry_from_config({})

  def test_build_circular_geometry(self):
    geo = build_sim.build_geometry_from_config({
        'geometry_type': 'circular',
        'nr': 5,  # override a default.
    })
    self.assertIsInstance(geo, geometry.CircularGeometry)
    np.testing.assert_array_equal(geo.mesh.nx, 5)
    np.testing.assert_array_equal(geo.B0, 5.3)  # test a default.

  def test_build_chease_geometry(self):
    geo = build_sim.build_geometry_from_config(
        {
            'geometry_type': 'chease',
            'nr': 5,  # override a default.
        },
        runtime_params=runtime_params_lib.GeneralRuntimeParams(),
    )
    self.assertIsInstance(geo, geometry.CHEASEGeometry)
    np.testing.assert_array_equal(geo.mesh.nx, 5)

  # pylint: disable=invalid-name
  def test_chease_geometry_updates_Ip(self):
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    original_Ip = runtime_params.profile_conditions.Ip
    geo = build_sim.build_geometry_from_config({
        'geometry_type': 'chease',
        'runtime_params': runtime_params,
        'Ip_from_parameters': False,  # this will force update runtime_params.Ip
    })
    self.assertIsInstance(geo, geometry.CHEASEGeometry)
    self.assertNotEqual(runtime_params.profile_conditions.Ip, original_Ip)
    # pylint: enable=invalid-name

  def test_empty_source_config_only_has_defaults_turned_off(self):
    """Tests that an empty source config has all sources turned off."""
    source_models = source_models_lib.SourceModelsBuilder({})()
    self.assertEqual(
        source_models.j_bootstrap.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models.jext.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models.qei_source.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertLen(source_models.sources, 3)
    self.assertEmpty(source_models.standard_sources)

  def test_adding_standard_source_via_config(self):
    """Tests that a source can be added with overriding defaults."""
    source_models = source_models_lib.SourceModelsBuilder({
        'gas_puff_source': {
            'puff_decay_length': 1.23,
        },
        'ohmic_heat_source': {
            'is_explicit': True,
            'mode': 'zero',  # turn it off.
        },
    })()
    # The non-standard ones are still off.
    self.assertEqual(
        source_models.j_bootstrap.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models.jext.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models.qei_source.runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    # But these new sources have been added.
    self.assertLen(source_models.sources, 5)
    self.assertLen(source_models.standard_sources, 2)
    # With the overriding params.
    # pytype: disable=attribute-error
    self.assertEqual(
        source_models.sources[
            'gas_puff_source'
        ].runtime_params.puff_decay_length,
        1.23,
    )
    # pytype: enable=attribute-error
    self.assertEqual(
        source_models.sources['gas_puff_source'].runtime_params.mode,
        source_runtime_params_lib.Mode.FORMULA_BASED,  # On by default.
    )
    self.assertEqual(
        source_models.sources['ohmic_heat_source'].runtime_params.mode,
        source_runtime_params_lib.Mode.ZERO,
    )

  def test_updating_formula_via_source_config(self):
    """Tests that we can set the formula type and params via the config."""
    source_models = source_models_lib.SourceModelsBuilder({
        'gas_puff_source': {
            'formula_type': 'gaussian',
            'total': 1,
            'c1': 2,
            'c2': 3,
        }
    })()
    gas_source = source_models.sources['gas_puff_source']
    self.assertIsInstance(gas_source.formula, formulas.Gaussian)
    self.assertIsInstance(
        gas_source.runtime_params.formula, formula_config.Gaussian
    )
    # pytype: disable=attribute-error
    self.assertEqual(gas_source.runtime_params.formula.total, 1)
    self.assertEqual(gas_source.runtime_params.formula.c1, 2)
    self.assertEqual(gas_source.runtime_params.formula.c2, 3)
    # pytype: enable=attribute-error

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
          expected_type=critical_gradient_transport.CriticalGradientModel,
      ),
      dict(
          testcase_name='qlknn',
          name='qlknn',
          expected_type=qlknn_wrapper.QLKNNTransportModel,
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
                'CGMalpha': 7.89,
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
      self.assertEqual(transport_model_builder.runtime_params.CGMalpha, 7.89)
    elif name == 'qlknn':
      assert isinstance(
          transport_model_builder.runtime_params, qlknn_wrapper.RuntimeParams
      )
      self.assertEqual(transport_model_builder.runtime_params.coll_mult, 10.11)
    else:
      self.fail(f'Unknown transport model: {name}')

  def test_missing_stepper_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_stepper_builder_from_config({})

  def test_unknown_stepper_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_stepper_builder_from_config({'stepper_type': 'foo'})

  @parameterized.named_parameters(
      dict(
          testcase_name='linear',
          stepper_type='linear',
          expected_type=linear_theta_method.LinearThetaMethod,
      ),
      dict(
          testcase_name='newton_raphson',
          stepper_type='newton_raphson',
          expected_type=nonlinear_theta_method.NewtonRaphsonThetaMethod,
      ),
      dict(
          testcase_name='optimizer',
          stepper_type='optimizer',
          expected_type=nonlinear_theta_method.OptimizerThetaMethod,
      ),
  )
  def test_build_stepper_builder_from_config(self, stepper_type, expected_type):
    """Builds a stepper from the config."""
    stepper_builder = build_sim.build_stepper_builder_from_config({
        'stepper_type': stepper_type,
        'theta_imp': 0.5,
    })
    transport_model_builder = (
        build_sim.build_transport_model_builder_from_config('constant')
    )
    transport_model = transport_model_builder()
    stepper = stepper_builder(
        transport_model=transport_model,
        source_models=source_models_lib.SourceModelsBuilder({})(),
    )
    self.assertIsInstance(stepper, expected_type)
    self.assertEqual(stepper_builder.runtime_params.theta_imp, 0.5)

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

  def test_build_array_time_step_calculator(self):
    time_stepper = build_sim.build_time_step_calculator_from_config({
        'calculator_type': 'array',
        'init_kwargs': {
            'arr': [0.1, 0.2, 0.3],
        },
    })
    self.assertIsInstance(
        time_stepper, array_time_step_calculator.ArrayTimeStepCalculator
    )


if __name__ == '__main__':
  absltest.main()
