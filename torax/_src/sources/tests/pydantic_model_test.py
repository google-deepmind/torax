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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import numpy as np
from torax._src import jax_utils
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.sources import base
from torax._src.sources import fusion_heat_source
from torax._src.sources import gas_puff_source
from torax._src.sources import generic_current_source
from torax._src.sources import pydantic_model
from torax._src.sources import qei_source
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax._src.torax_pydantic import torax_pydantic


class PydanticModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.neoclassical = neoclassical_pydantic_model.Neoclassical.from_dict({})

  @parameterized.parameters(
      dict(
          config={
              'gas_puff': {
                  'puff_decay_length': 0.3,
                  'S_total': 0.0,
              }
          },
          expected_sources_model=gas_puff_source.GasPuffSourceConfig,
      ),
      dict(
          config={
              'fusion': {},
          },
          expected_sources_model=fusion_heat_source.FusionHeatSourceConfig,
      ),
      dict(
          config={
              'impurity_radiation': {'model_name': 'mavrin_fit'},
          },
          expected_sources_model=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
      ),
      dict(
          config={
              'impurity_radiation': {'model_name': 'P_in_scaled_flat_profile'},
          },
          expected_sources_model=impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
      ),
  )
  def test_correct_source_model(
      self,
      config: dict[str, Any],
      expected_sources_model: type[base.SourceModelBase],
  ):
    sources_model = pydantic_model.Sources.from_dict(config)
    self.assertIsInstance(
        sources_model.source_model_config[list(config.keys())[0]],
        expected_sources_model,
    )
    # Check that the 3 default sources are always present.
    for key in [
        qei_source.QeiSource.SOURCE_NAME,
        generic_current_source.GenericCurrentSource.SOURCE_NAME,
    ]:
      self.assertIn(key, sources_model.source_model_config.keys())

  def test_adding_standard_source_via_config(self):
    """Tests that a source can be added with overriding defaults."""
    sources = pydantic_model.Sources.from_dict({
        'gas_puff': {
            'puff_decay_length': 1.23,
        },
        'ohmic': {
            'is_explicit': True,
            'mode': 'ZERO',  # turn it off.
        },
    })
    source_models = sources.build_models()
    # The non-standard ones are still off.
    self.assertEqual(
        sources.generic_current.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.ei_exchange.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    # But these new sources have been added.
    self.assertLen(source_models.standard_sources, 3)
    # With the overriding params.
    gas_puff_config = sources.gas_puff
    self.assertIsNotNone(gas_puff_config)
    self.assertEqual(
        gas_puff_config.puff_decay_length.get_value(0.0),
        1.23,
    )
    self.assertEqual(
        gas_puff_config.mode,
        source_runtime_params_lib.Mode.MODEL_BASED,  # On by default.
    )

    ohmic_config = sources.ohmic
    self.assertIsNotNone(ohmic_config)
    self.assertEqual(
        ohmic_config.mode,
        source_runtime_params_lib.Mode.ZERO,
    )

  def test_empty_source_config_only_has_defaults_turned_off(self):
    """Tests that an empty source config has all sources turned off."""
    sources = pydantic_model.Sources.from_dict({})
    self.assertEqual(
        sources.generic_current.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    self.assertEqual(
        sources.ei_exchange.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    source_models = sources.build_models()
    self.assertLen(source_models.standard_sources, 1)

  def test_adding_a_source_with_prescribed_values(self):
    """Tests that a source can be added with overriding defaults."""
    sources = pydantic_model.Sources.from_dict({
        'generic_current': {
            'mode': 'PRESCRIBED',
            'prescribed_values': (
                (
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 0.5, 1.0]),
                    np.full([4, 3], 42),
                ),
            ),
        },
        'ecrh': {
            'mode': 'PRESCRIBED',
            'prescribed_values': (
                3.0,
                4.0,
            ),
        },
    })
    mesh = torax_pydantic.Grid1D(nx=4,)
    torax_pydantic.set_grid(sources, mesh)
    source = sources.generic_current
    self.assertLen(source.prescribed_values, 1)
    self.assertIsInstance(
        source.prescribed_values[0], torax_pydantic.TimeVaryingArray
    )

    source = sources.ecrh
    self.assertIsNotNone(source)
    self.assertLen(source.prescribed_values, 2)
    self.assertIsInstance(
        source.prescribed_values[0], torax_pydantic.TimeVaryingArray
    )
    self.assertIsInstance(
        source.prescribed_values[1], torax_pydantic.TimeVaryingArray
    )
    value = source.prescribed_values[0].get_value(0.0)
    np.testing.assert_equal(value, 3.0)
    value = source.prescribed_values[1].get_value(0.0)
    np.testing.assert_equal(value, 4.0)

  def test_bremsstrahlung_and_mavrin_validator_with_bremsstrahlung_zero(self):
    valid_config = {
        'bremsstrahlung': {'mode': 'ZERO'},
        'impurity_radiation': {
            'mode': 'PRESCRIBED',
            'model_name': 'mavrin_fit',
        },
    }
    pydantic_model.Sources.from_dict(valid_config)

  def test_bremsstrahlung_and_mavrin_validator_with_mavrin_zero(self):
    valid_config = {
        'bremsstrahlung': {'mode': 'PRESCRIBED'},
        'impurity_radiation': {
            'mode': 'ZERO',
            'model_name': 'mavrin_fit',
        },
    }
    pydantic_model.Sources.from_dict(valid_config)

  def test_bremsstrahlung_and_mavrin_validator_with_constant_fraction(self):
    valid_config = {
        'bremsstrahlung': {'mode': 'PRESCRIBED'},
        'impurity_radiation': {
            'mode': 'PRESCRIBED',
            'model_name': 'P_in_scaled_flat_profile',
        },
    }
    pydantic_model.Sources.from_dict(valid_config)

  def test_bremsstrahlung_and_mavrin_validator_with_invalid_config(self):
    invalid_config = {
        'bremsstrahlung': {'mode': 'PRESCRIBED'},
        'impurity_radiation': {
            'mode': 'PRESCRIBED',
            'model_name': 'mavrin_fit',
        },
    }
    with self.assertRaisesRegex(
        ValueError,
        'Both bremsstrahlung and impurity_radiation',
    ):
      pydantic_model.Sources.from_dict(invalid_config)

  @parameterized.named_parameters(
      dict(
          testcase_name='bremsstrahlung',
          source_name='bremsstrahlung',
          model_name='wesson',
          field_to_update='use_relativistic_correction',
          initial_value=False,
          updated_value=True,
      ),
      dict(
          testcase_name='cyclotron_radiation',
          source_name='cyclotron_radiation',
          model_name='albajar_artaud',
          field_to_update='wall_reflection_coeff',
          initial_value=0.9,
          updated_value=0.8,
      ),
      dict(
          testcase_name='ecrh',
          source_name='ecrh',
          model_name='gaussian_lin_liu',
          field_to_update='P_total',
          initial_value=0.0,
          updated_value=1e6,
      ),
      dict(
          testcase_name='ei_exchange',
          source_name='ei_exchange',
          model_name=None,
          field_to_update='Qei_multiplier',
          initial_value=1.0,
          updated_value=2.0,
      ),
      dict(
          testcase_name='gas_puff',
          source_name='gas_puff',
          model_name='exponential',
          field_to_update='S_total',
          initial_value=1e22,
          updated_value=2e22,
      ),
      dict(
          testcase_name='generic_current',
          source_name='generic_current',
          model_name='gaussian',
          field_to_update='use_absolute_current',
          initial_value=False,
          updated_value=True,
      ),
      dict(
          testcase_name='generic_heat',
          source_name='generic_heat',
          model_name='gaussian',
          field_to_update='P_total',
          initial_value=120e6,
          updated_value=130e6,
      ),
      dict(
          testcase_name='generic_particle',
          source_name='generic_particle',
          model_name='gaussian',
          field_to_update='S_total',
          initial_value=1e22,
          updated_value=2e22,
      ),
      dict(
          testcase_name='icrh',
          source_name='icrh',
          model_name='toric_nn',
          field_to_update='wall_inner',
          initial_value=1.24,
          updated_value=1.25,
      ),
      dict(
          testcase_name='impurity_radiation_mavrin_fit',
          source_name='impurity_radiation',
          model_name='mavrin_fit',
          field_to_update='radiation_multiplier',
          initial_value=1.0,
          updated_value=2.0,
      ),
      dict(
          testcase_name='impurity_radiation_constant_fraction',
          source_name='impurity_radiation',
          model_name='P_in_scaled_flat_profile',
          field_to_update='fraction_P_heating',
          initial_value=0.1,
          updated_value=0.2,
      ),
      dict(
          testcase_name='pellet',
          source_name='pellet',
          model_name='gaussian',
          field_to_update='S_total',
          initial_value=2e22,
          updated_value=3e22,
      ),
  )
  def test_sources_under_jit(
      self,
      source_name,
      model_name,
      field_to_update,
      initial_value,
      updated_value,
  ):
    config_dict = {
        source_name: {field_to_update: initial_value}
    }
    if model_name is not None:
      config_dict[source_name]['model_name'] = model_name
    sm = pydantic_model.Sources.from_dict(config_dict)
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    torax_pydantic.set_grid(sm, geo.torax_mesh)

    @jax.jit
    def f(sources: pydantic_model.Sources):
      source = sources.source_model_config[source_name]
      return source.build_dynamic_params(t=0.0)

    with self.subTest('first_jit_compiles_and_returns_expected_value'):
      output = f(sm)
      output_field = getattr(output, field_to_update)
      chex.assert_trees_all_close(output_field, initial_value)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    with self.subTest('second_jit_updates_value_without_recompile'):
      sm._update_fields({f'{source_name}.{field_to_update}': updated_value})
      # Update the grid for any recreated TimeVaryingArrays.
      torax_pydantic.set_grid(sm, geo.torax_mesh, mode='relaxed')
      output = f(sm)
      output_field = getattr(output, field_to_update)
      chex.assert_trees_all_close(output_field, updated_value)
      self.assertEqual(jax_utils.get_number_of_compiles(f), 1)


if __name__ == '__main__':
  absltest.main()
