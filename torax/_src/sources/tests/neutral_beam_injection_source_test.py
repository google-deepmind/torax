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

from absl.testing import absltest
import jax
import numpy as np
import torax
from torax._src.sources import core_sources
from torax._src.sources import neutral_beam_injection_source as nbi_lib
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources.tests import test_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config as torax_pydantic_config

# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class NeutralBeamInjectionSourceTest(test_lib.SourceTestCase):
  """Tests for NeutralBeamInjectionSource."""

  source_name = nbi_lib.NeutralBeamInjectionSource.SOURCE_ID
  source_config_class = nbi_lib.NeutralBeamInjectionSourceConfig

  def test_source_name(self):
    self.assertEqual(nbi_lib.NeutralBeamInjectionSource.SOURCE_ID, 'nbi')

  def test_populated_source_outputs(self):
    expected = (
        core_sources.PopulatedSourceOutput.POWER_DENSITY_I,
        core_sources.PopulatedSourceOutput.POWER_DENSITY_E,
        core_sources.PopulatedSourceOutput.J_PARALLEL,
        core_sources.PopulatedSourceOutput.PARTICLE_SOURCE_E,
    )
    self.assertEqual(
        nbi_lib.NeutralBeamInjectionSource.POPULATED_SOURCE_OUTPUTS, expected
    )

  def test_default_config_values(self):
    config = nbi_lib.NeutralBeamInjectionSourceConfig.from_dict({})
    self.assertEqual(config.mode, runtime_params_lib.Mode.PRESCRIBED)
    self.assertEqual(config.model_name, 'prescribed')
    self.assertIsNone(config.prescribed_values.power_density_i)
    self.assertIsNone(config.prescribed_values.power_density_e)
    self.assertIsNone(config.prescribed_values.j_parallel)
    self.assertIsNone(config.prescribed_values.particle_source_e)
    self.assertIsNone(config.prescribed_values.fast_ions)

  def test_prescribed_get_value_without_fast_ions(self):
    config = default_configs.get_default_config_dict()
    config['sources'] = {
        'nbi': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 1.0e6,
                'power_density_e': 2.0e6,
                'j_parallel': 3.0e5,
                'particle_source_e': 4.0e19,
            },
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source = torax_config.sources.build_models().standard_sources['nbi']
    assert isinstance(source, nbi_lib.NeutralBeamInjectionSource)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    assert output.power_density_i is not None
    assert output.power_density_e is not None
    assert output.j_parallel is not None
    assert output.particle_source_e is not None
    np.testing.assert_allclose(output.power_density_i, 1.0e6)
    np.testing.assert_allclose(output.power_density_e, 2.0e6)
    np.testing.assert_allclose(output.j_parallel, 3.0e5)
    np.testing.assert_allclose(output.particle_source_e, 4.0e19)
    self.assertEqual(output.fast_ions, ())

  def test_prescribed_get_value_with_fast_ion_profiles(self):
    config = default_configs.get_default_config_dict()
    config['sources'] = {
        'nbi': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 1.0e6,
                'power_density_e': 2.0e6,
                'j_parallel': 3.0e5,
                'particle_source_e': 4.0e19,
                'fast_ions': {
                    'T': {
                        'n': {0.0: {0.0: 2.0e19, 1.0: 1.0e18}},
                        'T': {0.0: {0.0: 80.0, 1.0: 5.0}},
                    },
                    'D': {
                        'n': 5.0e18,
                        'T': 40.0,
                    },
                },
            },
        }
    }
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source = torax_config.sources.build_models().standard_sources['nbi']
    assert isinstance(source, nbi_lib.NeutralBeamInjectionSource)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    self.assertLen(output.fast_ions, 2)
    by_species = {fi.species: fi for fi in output.fast_ions}
    self.assertEqual(set(by_species.keys()), {'D', 'T'})
    fast_ion_t = by_species['T']
    self.assertEqual(fast_ion_t.source, 'nbi')
    expected_n = np.interp(geo.rho_norm, [0.0, 1.0], [2.0e19, 1.0e18])
    expected_t = np.interp(geo.rho_norm, [0.0, 1.0], [80.0, 5.0])
    np.testing.assert_allclose(fast_ion_t.n.value, expected_n)
    np.testing.assert_allclose(fast_ion_t.T.value, expected_t)
    assert fast_ion_t.n.right_face_constraint is not None
    assert fast_ion_t.T.right_face_constraint is not None
    np.testing.assert_allclose(fast_ion_t.n.right_face_constraint, 1.0e18)
    np.testing.assert_allclose(fast_ion_t.T.right_face_constraint, 5.0)

    fast_ion_d = by_species['D']
    self.assertEqual(fast_ion_d.source, 'nbi')
    np.testing.assert_allclose(fast_ion_d.n.value, 5.0e18)
    np.testing.assert_allclose(fast_ion_d.T.value, 40.0)

  def test_zero_mode(self):
    config = default_configs.get_default_config_dict()
    config['sources'] = {'nbi': {'mode': 'zero'}}
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    source = torax_config.sources.build_models().standard_sources['nbi']
    assert isinstance(source, nbi_lib.NeutralBeamInjectionSource)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        solver_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
    )
    assert output.power_density_i is not None
    assert output.power_density_e is not None
    assert output.j_parallel is not None
    assert output.particle_source_e is not None
    np.testing.assert_allclose(output.power_density_i, 0.0, atol=1e-15)
    np.testing.assert_allclose(output.power_density_e, 0.0, atol=1e-15)
    np.testing.assert_allclose(output.j_parallel, 0.0, atol=1e-15)
    np.testing.assert_allclose(output.particle_source_e, 0.0, atol=1e-15)
    self.assertEqual(output.fast_ions, ())

  def test_model_based_mode_raises_not_implemented(self):
    config = default_configs.get_default_config_dict()
    config['sources'] = {'nbi': {'mode': 'model_based'}}
    torax_config = torax_pydantic_config.ToraxConfig.from_dict(config)
    with self.assertRaisesRegex(
        NotImplementedError, 'Model-based NBI source is not yet implemented'
    ):
      core_profile_helpers.make_initial_inputs(torax_config)

  def test_make_initial_inputs_populates_nbi_fast_ion(self):
    config = default_configs.get_default_config_dict()
    config['numerics'] = {
        'enable_fast_ions': True,
    }
    config['sources'] = {
        'nbi': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 1.0e5,
                'power_density_e': 2.0e5,
                'j_parallel': 3.0e4,
                'particle_source_e': 4.0e18,
                'fast_ions': {
                    'D': {
                        'n': {0.0: {0.0: 2.0e19, 1.0: 1.0e18}},
                        'T': {0.0: {0.0: 60.0, 1.0: 2.0}},
                    },
                },
            },
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    core_profiles, _, geo = core_profile_helpers.make_initial_inputs(
        torax_config
    )
    expected_n_cell = np.interp(geo.rho_norm, [0.0, 1.0], [2.0e19, 1.0e18])
    expected_t_cell = np.interp(geo.rho_norm, [0.0, 1.0], [60.0, 2.0])

    self.assertLen(core_profiles.fast_ions, 1)
    fi = core_profiles.fast_ions[0]
    self.assertEqual(fi.source, 'nbi')
    self.assertEqual(fi.species, 'D')
    np.testing.assert_allclose(fi.n.value, expected_n_cell)
    assert fi.n.right_face_constraint is not None
    np.testing.assert_allclose(fi.n.right_face_constraint, 1.0e18)
    np.testing.assert_allclose(fi.T.value, expected_t_cell)
    assert fi.T.right_face_constraint is not None
    np.testing.assert_allclose(fi.T.right_face_constraint, 2.0)

  def test_profile_conditions_fast_ions_overrides_nbi_fast_ion(self):
    config = default_configs.get_default_config_dict()
    config['numerics'] = {
        'enable_fast_ions': True,
    }
    config['profile_conditions']['fast_ions'] = [{
        'source': 'nbi',
        'species': 'D',
        'n': {0.0: 3.0e19},
        'n_right_bc': 0.5e18,
        'T': {0.0: 90.0},
        'T_right_bc': 1.0,
    }]
    config['sources'] = {
        'nbi': {
            'mode': 'prescribed',
            'prescribed_values': {
                'power_density_i': 1.0e5,
                'power_density_e': 2.0e5,
                'j_parallel': 3.0e4,
                'particle_source_e': 4.0e18,
                'fast_ions': {
                    'D': {
                        'n': 1.0e19,
                        'T': 40.0,
                    },
                },
            },
        }
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    core_profiles, _, _ = core_profile_helpers.make_initial_inputs(torax_config)
    self.assertLen(core_profiles.fast_ions, 1)
    fi = core_profiles.fast_ions[0]
    self.assertEqual(fi.source, 'nbi')
    self.assertEqual(fi.species, 'D')
    np.testing.assert_allclose(fi.n.value, 3.0e19)
    assert fi.n.right_face_constraint is not None
    np.testing.assert_allclose(fi.n.right_face_constraint, 0.5e18)
    np.testing.assert_allclose(fi.T.value, 90.0)
    assert fi.T.right_face_constraint is not None
    np.testing.assert_allclose(fi.T.right_face_constraint, 1.0)


if __name__ == '__main__':
  absltest.main()
