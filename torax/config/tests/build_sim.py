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
from torax import geometry_provider
from torax.config import build_sim
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.pedestal_model import set_tped_nped
from torax.sources import formula_config
from torax.sources import formulas
from torax.sources import runtime_params as source_runtime_params_lib
from torax.stepper import linear_theta_method
from torax.stepper import nonlinear_theta_method
from torax.stepper import runtime_params as stepper_params
from torax.time_step_calculator import array_time_step_calculator
from torax.time_step_calculator import chi_time_step_calculator
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import constant as constant_transport
from torax.transport_model import critical_gradient as critical_gradient_transport
from torax.transport_model import qlknn_wrapper
from torax.transport_model import runtime_params as transport_model_params


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
                    ne_is_fGW=False,
                ),
                numerics=dict(
                    q_correction_factor=0.2,
                ),
            ),
            geometry=dict(
                geometry_type='circular',
                n_rho=5,
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
      self.assertEqual(dynamic_runtime_params_slice.plasma_composition.Ai, 0.1)
      self.assertEqual(
          dynamic_runtime_params_slice.profile_conditions.ne_is_fGW,
          False,
      )
      self.assertEqual(
          dynamic_runtime_params_slice.numerics.q_correction_factor,
          0.2,
      )
    with self.subTest('geometry'):
      geo = sim.geometry_provider(sim.initial_state.t)
      self.assertIsInstance(geo, geometry.CircularAnalyticalGeometry)
      self.assertEqual(geo.torax_mesh.nx, 5)
    with self.subTest('sources'):
      self.assertEqual(
          sim.source_models_builder.runtime_params['pellet_source'].mode,
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
            'Ai': 0.1,  # scalar fields.
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
            'q_correction_factor': 0.2,  # scalar fields.
            # Designate the interpolation mode, as well, setting to "step".
            'resistivity_mult': ({0: 0.3, 1: 0.6, 2: 0.9}, 'step'),
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(runtime_params.plasma_composition.Ai, 0.1)
    self.assertEqual(runtime_params.profile_conditions.ne_is_fGW, False)
    self.assertEqual(runtime_params.numerics.q_correction_factor, 0.2)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
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

  def test_missing_geometry_type_raises_error(self):
    with self.assertRaises(ValueError):
      build_sim.build_geometry_provider_from_config({})

  def test_build_circular_geometry(self):
    geo_provider = build_sim.build_geometry_provider_from_config({
        'geometry_type': 'circular',
        'n_rho': 5,  # override a default.
    })
    self.assertIsInstance(
        geo_provider, geometry_provider.ConstantGeometryProvider
    )
    geo = geo_provider(t=0)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 5)
    self.assertIsInstance(geo, geometry.CircularAnalyticalGeometry)
    np.testing.assert_array_equal(geo.B0, 5.3)  # test a default.

  def test_build_geometry_from_chease(self):
    geo_provider = build_sim.build_geometry_provider_from_config(
        {
            'geometry_type': 'chease',
            'n_rho': 5,  # override a default.
        },
    )
    self.assertIsInstance(
        geo_provider, geometry_provider.ConstantGeometryProvider
    )
    self.assertIsInstance(geo_provider(t=0), geometry.StandardGeometry)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 5)

  def test_build_time_dependent_geometry_from_chease(self):
    """Tests correctness of config constraints with time-dependent geometry."""

    base_config = {
        'geometry_type': 'chease',
        'Ip_from_parameters': True,
        'n_rho': 10,  # overrides the default
        'geometry_configs': {
            0.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
            1.0: {
                'geometry_file': 'ITER_hybrid_citrin_equil_cheasedata.mat2cols',
                'Rmaj': 6.2,
                'Rmin': 2.0,
                'B0': 5.3,
            },
        },
    }

    # Test valid config
    geo_provider = build_sim.build_geometry_provider_from_config(base_config)
    self.assertIsInstance(geo_provider, geometry.StandardGeometryProvider)
    self.assertIsInstance(geo_provider(t=0), geometry.StandardGeometry)
    np.testing.assert_array_equal(geo_provider.torax_mesh.nx, 10)

    # Test invalid configs:
    for param, value in zip(
        ['n_rho', 'Ip_from_parameters', 'geometry_dir'], [5, True, '.']
    ):
      for time_key in [0.0, 1.0]:
        invalid_config = base_config.copy()
        invalid_config['geometry_configs'][time_key][param] = value
        with self.assertRaises(ValueError):
          build_sim.build_geometry_provider_from_config(invalid_config)

  # pylint: disable=invalid-name
  def test_chease_geometry_updates_Ip(self):
    """Tests that the Ip is updated when using chease geometry."""
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    original_Ip_tot = runtime_params.profile_conditions.Ip_tot
    geo_provider = build_sim.build_geometry_provider_from_config({
        'geometry_type': 'chease',
        'Ip_from_parameters': (
            False
        ),  # this will force update runtime_params.Ip_tot
    })
    runtime_params_provider = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport=transport_model_params.RuntimeParams(),
            sources={},
            stepper=stepper_params.RuntimeParams(),
            torax_mesh=geo_provider.torax_mesh,
        )
    )
    geo = geo_provider(t=0)
    dynamic_runtime_params_slice = runtime_params_provider(
        t=0,
    )
    dynamic_slice, geo = runtime_params_slice.make_ip_consistent(
        dynamic_runtime_params_slice, geo
    )
    self.assertIsInstance(geo, geometry.StandardGeometry)
    self.assertIsNotNone(dynamic_slice)
    self.assertNotEqual(
        dynamic_slice.profile_conditions.Ip_tot, original_Ip_tot
    )
    # pylint: enable=invalid-name

  def test_empty_source_config_only_has_defaults_turned_off(self):
    """Tests that an empty source config has all sources turned off."""
    source_models_builder = build_sim.build_sources_builder_from_config({})
    source_models = source_models_builder()
    self.assertEqual(
        source_models_builder.runtime_params['j_bootstrap'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models_builder.runtime_params['generic_current_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models_builder.runtime_params['qei_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertLen(source_models.sources, 3)
    self.assertLen(source_models.standard_sources, 1)

  def test_adding_standard_source_via_config(self):
    """Tests that a source can be added with overriding defaults."""
    source_models_builder = build_sim.build_sources_builder_from_config({
        'gas_puff_source': {
            'puff_decay_length': 1.23,
        },
        'ohmic_heat_source': {
            'is_explicit': True,
            'mode': 'zero',  # turn it off.
        },
    })
    source_models = source_models_builder()
    # The non-standard ones are still off.
    self.assertEqual(
        source_models_builder.runtime_params['j_bootstrap'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models_builder.runtime_params['generic_current_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        source_models_builder.runtime_params['qei_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    # But these new sources have been added.
    self.assertLen(source_models.sources, 5)
    self.assertLen(source_models.standard_sources, 3)
    # With the overriding params.
    # pytype: disable=attribute-error
    self.assertEqual(
        source_models_builder.runtime_params[
            'gas_puff_source'
        ].puff_decay_length,
        1.23,
    )
    # pytype: enable=attribute-error
    self.assertEqual(
        source_models_builder.runtime_params['gas_puff_source'].mode,
        source_runtime_params_lib.Mode.FORMULA_BASED,  # On by default.
    )
    self.assertEqual(
        source_models_builder.runtime_params['ohmic_heat_source'].mode,
        source_runtime_params_lib.Mode.ZERO,
    )

  def test_updating_formula_via_source_config(self):
    """Tests that we can set the formula type and params via the config."""
    source_models_builder = build_sim.build_sources_builder_from_config({
        'gas_puff_source': {
            'formula_type': 'gaussian',
            'total': 1,
            'c1': 2,
            'c2': 3,
        }
    })
    source_models = source_models_builder()
    gas_source = source_models.sources['gas_puff_source']
    self.assertIsInstance(gas_source.formula, formulas.Gaussian)
    gas_source_runtime_params = source_models_builder.runtime_params[
        'gas_puff_source'
    ]
    self.assertIsInstance(
        gas_source_runtime_params.formula,
        formula_config.Gaussian,
    )
    # pytype: disable=attribute-error
    self.assertEqual(gas_source_runtime_params.formula.total, 1)
    self.assertEqual(gas_source_runtime_params.formula.c1, 2)
    self.assertEqual(gas_source_runtime_params.formula.c2, 3)
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
    pedestal_model_builder = build_sim.build_pedestal_model_builder_from_config(
        {}
    )
    pedestal_model = pedestal_model_builder()
    source_models_builder = build_sim.build_sources_builder_from_config({})
    source_models = source_models_builder()
    stepper = stepper_builder(
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
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
