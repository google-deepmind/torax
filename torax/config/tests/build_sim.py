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
from torax.sources import formula_config
from torax.sources import formulas
from torax.sources import runtime_params as source_runtime_params_lib


class BuildSimTest(parameterized.TestCase):
  """Unit tests for the `torax.config.build_sim` module."""

  def test_build_sim_raises_error_with_missing_keys(self):
    with self.assertRaises(ValueError):
      build_sim.build_sim_from_config({})

  def test_build_sim(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(ValueError):
      build_sim.build_sim_from_config({
          'runtime_params': {},
          'geometry': {},
          'sources': {},
          'transport': {},
          'stepper': {},
          'time_step_calculator': {},
      })

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
            'resistivity_mult': {0: 0.3, 1: 0.6, 2: 0.9},  # time-dependent.
        },
        'output_dir': '/tmp/this/is/a/test',
    })
    self.assertEqual(runtime_params.plasma_composition.Ai, 0.1)
    self.assertEqual(runtime_params.plasma_composition.Zeff[0], 0.1)
    self.assertEqual(runtime_params.profile_conditions.nbar_is_fGW, False)
    self.assertEqual(runtime_params.profile_conditions.Ip[1], 0.4)
    self.assertEqual(runtime_params.numerics.q_correction_factor, 0.2)
    self.assertEqual(runtime_params.numerics.resistivity_mult[2], 0.9)
    self.assertEqual(runtime_params.output_dir, '/tmp/this/is/a/test')

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
    source_models = build_sim.build_sources_from_config({})
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
    source_models = build_sim.build_sources_from_config({
        'gas_puff_source': {
            'puff_decay_length': 1.23,
        },
        'ohmic_heat_source': {
            'is_explicit': True,
            'mode': 'zero',  # turn it off.
        },
    })
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
    source_models = build_sim.build_sources_from_config(
        {
            'gas_puff_source': {
                'func': 'gauss',
                'total': 1,
                'c1': 2,
                'c2': 3,
            }
        }
    )
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

  def test_build_transport_model_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_transport_model_from_config({})

  def test_build_stepper_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_stepper_from_config({})

  def test_build_time_step_calculator_from_config(self):
    # TODO(b/323504363): Update once implemented.
    with self.assertRaises(NotImplementedError):
      build_sim.build_time_step_calculator_from_config({})


if __name__ == '__main__':
  absltest.main()
