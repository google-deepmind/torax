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
import pydantic
from torax._src import jax_utils
from torax._src.geometry import circular_geometry
from torax._src.neoclassical import config as neoclassical_config
from torax._src.sources import base
from torax._src.sources import config as sources_config
from torax._src.sources import fusion_heat_source
from torax._src.sources import gas_puff_source
from torax._src.sources import qei_source
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_constant_fraction
from torax._src.sources.impurity_radiation_heat_sink import impurity_radiation_mavrin_fit
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic


class SourcesConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.neoclassical = (
        neoclassical_config.AnalyticalNeoclassicalConfig.from_dict({})
    )

  @parameterized.parameters(
      dict(
          config_dict={
              'gas_puff': {
                  'puff_decay_length': 0.3,
                  'S_total': 0.0,
              }
          },
          expected_sources_model=gas_puff_source.GasPuffSourceConfig,
      ),
      dict(
          config_dict={
              'fusion': {},
          },
          expected_sources_model=fusion_heat_source.FusionHeatSourceConfig,
      ),
      dict(
          config_dict={
              'impurity_radiation': {'model_name': 'mavrin_fit'},
          },
          expected_sources_model=impurity_radiation_mavrin_fit.ImpurityRadiationHeatSinkMavrinFitConfig,
      ),
      dict(
          config_dict={
              'impurity_radiation': {
                  'model_name': 'P_in_scaled_flat_profile',
                  'fraction_P_heating': 0.1,
              },
          },
          expected_sources_model=impurity_radiation_constant_fraction.ImpurityRadiationHeatSinkConstantFractionConfig,
      ),
  )
  def test_correct_source_model(
      self,
      config_dict: dict[str, Any],
      expected_sources_model: type[base.SourceConfigBase],
  ):
    sources_model = sources_config.SourcesConfig.from_dict(config_dict)
    self.assertIsInstance(
        sources_model.source_model_config[list(config_dict.keys())[0]],
        expected_sources_model,
    )
    # Check that ei_exchange is always present by default.
    self.assertIn(
        qei_source.QeiSource.SOURCE_ID,
        sources_model.source_model_config.keys(),
    )

  def test_adding_standard_source_via_config(self):
    """Tests that a source can be added with overriding defaults."""
    sources = sources_config.SourcesConfig.from_dict({
        'gas_puff': {
            'is_explicit': True,
            'puff_decay_length': 1.23,
            'S_total': 1e22,
        },
        'ohmic': {
            'mode': 'zero',  # turn it off.
        },
    })
    source_models = sources.build_models()
    # The default baseline source is still off.
    self.assertEqual(
        sources.ei_exchange.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    # But these new sources have been added.
    self.assertLen(source_models.standard_sources, 2)
    # With the overriding params.
    gas_puff_config = sources.gas_puff
    self.assertIsNotNone(gas_puff_config)
    assert gas_puff_config.puff_decay_length is not None
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
    sources = sources_config.SourcesConfig.from_dict({})
    self.assertEqual(
        sources.ei_exchange.mode,
        source_runtime_params_lib.Mode.ZERO,
    )
    source_models = sources.build_models()
    self.assertEmpty(source_models.standard_sources)

  def test_adding_a_source_with_prescribed_values(self):
    """Tests that a source can be added with overriding defaults."""
    sources = sources_config.SourcesConfig.from_dict({
        'generic_current': {
            'mode': 'prescribed',
            'prescribed_values': {
                'j_parallel': (
                    np.array([0.0, 1.0, 2.0, 3.0]),
                    np.array([0.0, 0.5, 1.0]),
                    np.full([4, 3], 42),
                ),
            },
        },
        'ecrh': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_e': 3.0,
                'j_parallel': 4.0,
            },
        },
    })
    mesh = torax_pydantic.Grid1D(
        face_centers=interpolated_param_2d.get_face_centers(nx=4)
    )
    torax_pydantic.set_grid(sources, mesh)
    source = sources.generic_current
    self.assertIsNotNone(source)
    self.assertIsInstance(
        source.prescribed_values.j_parallel, torax_pydantic.TimeVaryingArray
    )
    self.assertIsNone(source.prescribed_values.power_density_i)
    self.assertIsNone(source.prescribed_values.power_density_e)
    self.assertIsNone(source.prescribed_values.particle_source_e)

    source = sources.ecrh
    self.assertIsNotNone(source)
    self.assertIsInstance(
        source.prescribed_values.power_density_e,
        torax_pydantic.TimeVaryingArray,
    )
    self.assertIsInstance(
        source.prescribed_values.j_parallel, torax_pydantic.TimeVaryingArray
    )
    value = source.prescribed_values.power_density_e.get_value(0.0)
    np.testing.assert_equal(value, 3.0)
    value = source.prescribed_values.j_parallel.get_value(0.0)
    np.testing.assert_equal(value, 4.0)

  def test_bremsstrahlung_and_mavrin_both_active_sets_exclude_flag(self):
    """When both bremsstrahlung and Mavrin are active, exclude flag is set."""
    config_dict = {
        'bremsstrahlung': {'mode': 'prescribed'},
        'impurity_radiation': {
            'mode': 'prescribed',
            'model_name': 'mavrin_fit',
        },
    }
    sources = sources_config.SourcesConfig.from_dict(config_dict)
    assert sources.bremsstrahlung is not None
    self.assertTrue(sources.bremsstrahlung.exclude_impurity_bremsstrahlung)

  def test_bremsstrahlung_active_mavrin_zero_does_not_set_exclude_flag(self):
    config_dict = {
        'bremsstrahlung': {'mode': 'prescribed'},
        'impurity_radiation': {
            'mode': 'zero',
            'model_name': 'mavrin_fit',
        },
    }
    sources = sources_config.SourcesConfig.from_dict(config_dict)
    assert sources.bremsstrahlung is not None
    self.assertFalse(sources.bremsstrahlung.exclude_impurity_bremsstrahlung)

  def test_bremsstrahlung_zero_mavrin_active_does_not_set_exclude_flag(self):
    config_dict = {
        'bremsstrahlung': {'mode': 'zero'},
        'impurity_radiation': {
            'mode': 'prescribed',
            'model_name': 'mavrin_fit',
        },
    }
    sources = sources_config.SourcesConfig.from_dict(config_dict)
    assert sources.bremsstrahlung is not None
    self.assertFalse(sources.bremsstrahlung.exclude_impurity_bremsstrahlung)

  def test_bremsstrahlung_with_constant_fraction_does_not_set_exclude_flag(
      self,
  ):
    config_dict = {
        'bremsstrahlung': {'mode': 'prescribed'},
        'impurity_radiation': {
            'mode': 'prescribed',
            'model_name': 'P_in_scaled_flat_profile',
        },
    }
    sources = sources_config.SourcesConfig.from_dict(config_dict)
    assert sources.bremsstrahlung is not None
    self.assertFalse(sources.bremsstrahlung.exclude_impurity_bremsstrahlung)

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
          extra_fields={'gaussian_width': 0.1, 'gaussian_location': 0.0},
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
          extra_fields={'puff_decay_length': 0.05},
      ),
      dict(
          testcase_name='generic_current',
          source_name='generic_current',
          model_name='gaussian',
          field_to_update='fraction_of_total_current',
          initial_value=0.2,
          updated_value=0.3,
          extra_fields={
              'gaussian_width': 0.05,
              'gaussian_location': 0.4,
          },
      ),
      dict(
          testcase_name='generic_heat',
          source_name='generic_heat',
          model_name='gaussian',
          field_to_update='P_total',
          initial_value=120e6,
          updated_value=130e6,
          extra_fields={'gaussian_width': 0.25, 'gaussian_location': 0.0},
      ),
      dict(
          testcase_name='generic_particle',
          source_name='generic_particle',
          model_name='gaussian',
          field_to_update='S_total',
          initial_value=1e22,
          updated_value=2e22,
          extra_fields={'particle_width': 0.25, 'deposition_location': 0.0},
      ),
      dict(
          testcase_name='icrh_scaled_profile',
          source_name='icrh',
          model_name='scaled_profile',
          field_to_update='reference_B0',
          initial_value=12.2,
          updated_value=8.71,
          extra_fields={'minority_species': 'He3', 'P_total': 10e6},
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
          extra_fields={
              'pellet_width': 0.1,
              'pellet_deposition_location': 0.85,
          },
      ),
  )
  def test_sources_under_jit(
      self,
      source_name,
      model_name,
      field_to_update,
      initial_value,
      updated_value,
      extra_fields=None,
  ):
    config_dict = {source_name: {field_to_update: initial_value}}
    if model_name is not None:
      config_dict[source_name]['model_name'] = model_name
    if extra_fields is not None:
      config_dict[source_name].update(extra_fields)
    sm = sources_config.SourcesConfig.from_dict(config_dict)
    geo = circular_geometry.CircularConfig().build_geometry()
    torax_pydantic.set_grid(sm, geo.torax_mesh)

    @jax.jit
    def f(sources: sources_config.SourcesConfig):
      source = sources.source_model_config[source_name]
      return source.build_runtime_params(t=0.0)

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

  def test_generic_current_invalid_model_name_error_message(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_name' found using 'model_name' does not match any"
        r' of the expected tags',
    ):
      sources_config.SourcesConfig.from_dict(
          {'generic_current': {'model_name': 'invalid_name'}}
      )

  def test_prescribed_fast_ions_default_to_none_and_empty_tuple(self):
    cfg = fusion_heat_source.FusionHeatSourceConfig.from_dict(
        {'mode': 'prescribed'}
    )
    self.assertIsNone(cfg.prescribed_values.fast_ions)
    params = cfg.build_runtime_params(t=0.0)
    self.assertEqual(params.prescribed_values.fast_ions, ())

    empty_dict_cfg = fusion_heat_source.FusionHeatSourceConfig.from_dict(
        {'mode': 'prescribed', 'prescribed_values': {'fast_ions': {}}}
    )
    self.assertEqual(
        empty_dict_cfg.build_runtime_params(t=0.0).prescribed_values.fast_ions,
        (),
    )

  def test_prescribed_fast_ions_single_and_multiple_species_under_jit(self):
    sm = sources_config.SourcesConfig.from_dict({
        'fusion': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 1.0e6,
                'power_density_e': 2.0e6,
                'fast_ions': {
                    'He4': {
                        'n': {0.0: {0.0: 2.0e19, 1.0: 1.0e18}},
                        'T': {0.0: {0.0: 80.0, 1.0: 5.0}},
                    },
                    'D': {
                        'n': 5.0e18,
                        'T': 40.0,
                    },
                },
            },
        },
        'icrh': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 3.0e6,
                'power_density_e': 4.0e6,
                'fast_ions': {
                    'He3': {
                        'n': 1.0e19,
                        'T': 100.0,
                    },
                },
            },
        },
    })
    geo = circular_geometry.CircularConfig().build_geometry()
    torax_pydantic.set_grid(sm, geo.torax_mesh)

    @jax.jit
    def f(sources: sources_config.SourcesConfig):
      assert sources.fusion is not None
      assert sources.icrh is not None
      return (
          sources.fusion.build_runtime_params(t=0.0),
          sources.icrh.build_runtime_params(t=0.0),
      )

    fusion_params, icrh_params = f(sm)
    self.assertLen(fusion_params.prescribed_values.fast_ions, 2)
    by_species = {
        fi.species: fi for fi in fusion_params.prescribed_values.fast_ions
    }
    self.assertEqual(set(by_species.keys()), {'D', 'He4'})
    he4 = by_species['He4']
    self.assertEqual(he4.source, 'fusion')
    expected_n = np.interp(geo.rho_norm, [0.0, 1.0], [2.0e19, 1.0e18])
    expected_t = np.interp(geo.rho_norm, [0.0, 1.0], [80.0, 5.0])
    chex.assert_trees_all_close(he4.n.value, expected_n)
    chex.assert_trees_all_close(he4.n.right_face_constraint, 1.0e18)
    chex.assert_trees_all_close(he4.T.value, expected_t)
    chex.assert_trees_all_close(he4.T.right_face_constraint, 5.0)
    self.assertEqual(by_species['D'].source, 'fusion')
    chex.assert_trees_all_close(
        by_species['D'].n.value, np.full_like(geo.rho_norm, 5.0e18)
    )
    chex.assert_trees_all_close(
        by_species['D'].T.value, np.full_like(geo.rho_norm, 40.0)
    )

    self.assertLen(icrh_params.prescribed_values.fast_ions, 1)
    he3 = icrh_params.prescribed_values.fast_ions[0]
    self.assertEqual(he3.species, 'He3')
    self.assertEqual(he3.source, 'icrh')
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

    sm._update_fields({
        'fusion.prescribed_values.fast_ions.He4.n': {
            0.0: {0.0: 3.0e19, 1.0: 2.0e18}
        },
        'fusion.prescribed_values.fast_ions.He4.T': {
            0.0: {0.0: 60.0, 1.0: 3.0}
        },
    })
    torax_pydantic.set_grid(sm, geo.torax_mesh, mode='relaxed')
    updated_fusion_params, _ = f(sm)
    updated_he4 = {
        fi.species: fi
        for fi in updated_fusion_params.prescribed_values.fast_ions
    }['He4']
    expected_updated_n = np.interp(geo.rho_norm, [0.0, 1.0], [3.0e19, 2.0e18])
    expected_updated_t = np.interp(geo.rho_norm, [0.0, 1.0], [60.0, 3.0])
    chex.assert_trees_all_close(updated_he4.n.value, expected_updated_n)
    chex.assert_trees_all_close(updated_he4.n.right_face_constraint, 2.0e18)
    chex.assert_trees_all_close(updated_he4.T.value, expected_updated_t)
    chex.assert_trees_all_close(updated_he4.T.right_face_constraint, 3.0)
    self.assertEqual(jax_utils.get_number_of_compiles(f), 1)

  @parameterized.named_parameters(
      dict(testcase_name='missing_T', entry={'n': 1.0e19}),
      dict(testcase_name='missing_n', entry={'T': 50.0}),
      dict(testcase_name='negative_n', entry={'n': -1.0, 'T': 50.0}),
      dict(testcase_name='negative_T', entry={'n': 1.0e19, 'T': -5.0}),
  )
  def test_prescribed_fast_ions_invalid_entry_raises_error(self, entry):
    with self.assertRaises(pydantic.ValidationError):
      base.PrescribedValues.from_dict({'fast_ions': {'D': entry}})

  def test_prescribed_fast_ions_invalid_species_raises_error(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError, 'not a supported species'
    ):
      base.PrescribedValues.from_dict(
          {'fast_ions': {'Xe': {'n': 1.0e19, 'T': 50.0}}}
      )

  def test_prescribed_fast_ions_on_non_fast_ion_source_raises_error(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'ElectronCyclotronSource does not support fast_ions',
    ):
      sources_config.SourcesConfig.from_dict({
          'ecrh': {
              'mode': 'prescribed',
              'prescribed_values': {
                  'power_density_e': 1.0e6,
                  'j_parallel': 1.0e5,
                  'fast_ions': {'D': {'n': 1.0e19, 'T': 50.0}},
              },
          },
      })

if __name__ == '__main__':
  absltest.main()
