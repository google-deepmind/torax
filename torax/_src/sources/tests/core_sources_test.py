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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import core_sources as core_sources_lib


# pylint: disable=invalid-name
class CoreSourcesTest(parameterized.TestCase):

  def test_source_output_zeros_all_profiles(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    all_zeros = core_sources_lib.SourceOutput.zeros(geo)
    self.assertIsNotNone(all_zeros.power_density_i)
    self.assertIsNotNone(all_zeros.power_density_e)
    self.assertIsNotNone(all_zeros.j_parallel)
    self.assertIsNotNone(all_zeros.particle_source_e)
    np.testing.assert_allclose(all_zeros.power_density_i, 0.0)
    np.testing.assert_allclose(all_zeros.power_density_e, 0.0)
    np.testing.assert_allclose(all_zeros.j_parallel, 0.0)
    np.testing.assert_allclose(all_zeros.particle_source_e, 0.0)

  def test_source_output_zeros_populated_source_outputs_subset(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    partial_zeros = core_sources_lib.SourceOutput.zeros(
        geo,
        populated_source_outputs=(
            core_sources_lib.PopulatedSourceOutput.POWER_DENSITY_E,
            core_sources_lib.PopulatedSourceOutput.J_PARALLEL,
        ),
    )
    self.assertIsNone(partial_zeros.power_density_i)
    self.assertIsNone(partial_zeros.particle_source_e)
    self.assertIsNotNone(partial_zeros.power_density_e)
    self.assertIsNotNone(partial_zeros.j_parallel)
    np.testing.assert_allclose(partial_zeros.power_density_e, 0.0)
    np.testing.assert_allclose(partial_zeros.j_parallel, 0.0)

  def test_summed_power_density_profiles_dont_change_when_jitting(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry

    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones_like(geo.rho)
    profiles = core_sources_lib.CoreSources(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=core_sources_lib.QeiInfo.zeros(geo),
        sources={
            'generic_heat': core_sources_lib.SourceOutput(
                power_density_i=ones,
                power_density_e=ones * 2,
            ),
            'fusion': core_sources_lib.SourceOutput(
                power_density_i=ones * 3,
                power_density_e=ones * 4,
            ),
            'bremsstrahlung': core_sources_lib.SourceOutput(
                power_density_e=-ones,
            ),
            'ohmic': core_sources_lib.SourceOutput(
                power_density_e=ones * 5,
            ),
        },
    )
    with self.subTest('without_jit'):
      summed_T_i = profiles.total_sources('power_density_i', geo)
      np.testing.assert_allclose(summed_T_i, ones * 4 * geo.vpr)
      summed_T_e = profiles.total_sources('power_density_e', geo)
      np.testing.assert_allclose(summed_T_e, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp = jax.jit(profiles.total_sources, static_argnames='source_type')
      jitted_T_i = sum_temp('power_density_i', geo)
      np.testing.assert_allclose(jitted_T_i, ones * 4 * geo.vpr)
      jitted_T_e = sum_temp('power_density_e', geo)
      np.testing.assert_allclose(jitted_T_e, ones * 10 * geo.vpr)

  def test_from_source_outputs(self):
    geo = (
        circular_geometry.CircularConfig(n_rho=10)
        .build_runtime_params(0.0)
        .geometry
    )
    ones = jnp.ones(10)
    mock_fast_ion = fast_ion_lib.FastIon(
        species='He3',
        source='heat',
        n=cell_variable.CellVariable(
            value=ones,
            face_centers=geo.rho_face_norm,
        ),
        T=cell_variable.CellVariable(
            value=ones * 10.0,
            face_centers=geo.rho_face_norm,
        ),
    )
    bootstrap_current = bootstrap_current_base.BootstrapCurrent.zeros(geo)
    qei = core_sources_lib.QeiInfo.zeros(geo)
    standard_sources = {
        'heat': object(),
        'particle': object(),
        'current': object(),
    }
    source_outputs = {
        'heat': core_sources_lib.SourceOutput(
            power_density_i=ones * 2.0,
            power_density_e=ones * 3.0,
            fast_ions=(mock_fast_ion,),
        ),
        'particle': core_sources_lib.SourceOutput(
            particle_source_e=ones * 4.0,
        ),
        'current': core_sources_lib.SourceOutput(
            j_parallel=ones * 5.0,
        ),
        'ignored_nonstandard': core_sources_lib.SourceOutput(
            power_density_e=ones * 99.0,
        ),
    }
    profiles = core_sources_lib.CoreSources.from_source_outputs(
        source_outputs=source_outputs,
        standard_sources=standard_sources,
        bootstrap_current=bootstrap_current,
        qei=qei,
    )
    self.assertIn('heat', profiles.sources)
    self.assertIn('particle', profiles.sources)
    self.assertIn('current', profiles.sources)
    self.assertNotIn('ignored_nonstandard', profiles.sources)
    self.assertIsNotNone(profiles.sources['heat'].power_density_i)
    self.assertIsNotNone(profiles.sources['heat'].power_density_e)
    self.assertIsNotNone(profiles.sources['particle'].particle_source_e)
    self.assertIsNotNone(profiles.sources['current'].j_parallel)
    np.testing.assert_allclose(
        profiles.sources['heat'].power_density_i, ones * 2.0
    )
    np.testing.assert_allclose(
        profiles.sources['heat'].power_density_e, ones * 3.0
    )
    np.testing.assert_allclose(
        profiles.sources['particle'].particle_source_e, ones * 4.0
    )
    np.testing.assert_allclose(
        profiles.sources['current'].j_parallel, ones * 5.0
    )
    self.assertEqual(profiles.sources['heat'].fast_ions, (mock_fast_ion,))

  def test_update_sources(self):
    geo = (
        circular_geometry.CircularConfig(n_rho=10)
        .build_runtime_params(0.0)
        .geometry
    )
    ones = jnp.ones(10)
    standard_sources = {
        'heat': object(),
        'particle': object(),
        'radiation': object(),
    }
    source_models = mock.Mock(standard_sources=standard_sources)
    profiles = core_sources_lib.CoreSources.from_source_outputs(
        source_outputs={
            'heat': core_sources_lib.SourceOutput(
                power_density_e=ones * 2.0,
            ),
            'particle': core_sources_lib.SourceOutput(
                particle_source_e=ones * 4.0,
            ),
        },
        standard_sources=standard_sources,
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=core_sources_lib.QeiInfo.zeros(geo),
    )
    updated = profiles.update_sources(
        {
            'heat': core_sources_lib.SourceOutput(
                power_density_e=ones * 7.0,
            ),
            'radiation': core_sources_lib.SourceOutput(
                power_density_e=-ones,
            ),
            'ignored_nonstandard': core_sources_lib.SourceOutput(
                power_density_e=ones * 99.0,
            ),
        },
        source_models,
    )
    self.assertNotIn('ignored_nonstandard', updated.sources)
    self.assertIsNotNone(profiles.sources['heat'].power_density_e)
    self.assertIsNotNone(updated.sources['heat'].power_density_e)
    self.assertIsNotNone(updated.sources['particle'].particle_source_e)
    self.assertIsNotNone(updated.sources['radiation'].power_density_e)
    np.testing.assert_allclose(
        profiles.sources['heat'].power_density_e, ones * 2.0
    )
    np.testing.assert_allclose(
        updated.sources['heat'].power_density_e, ones * 7.0
    )
    np.testing.assert_allclose(
        updated.sources['particle'].particle_source_e, ones * 4.0
    )
    np.testing.assert_allclose(
        updated.sources['radiation'].power_density_e, -ones
    )

  def test_extract_psi_sources(self):
    ones = np.ones(10)
    source_models = mock.Mock(
        standard_sources={'heat': object(), 'current': object()}
    )
    psi_sources = core_sources_lib.extract_psi_sources(
        {
            'heat': core_sources_lib.SourceOutput(power_density_e=ones * 2.0),
            'current': core_sources_lib.SourceOutput(j_parallel=ones * 5.0),
            'ignored_nonstandard': core_sources_lib.SourceOutput(
                j_parallel=ones * 9.0
            ),
        },
        source_models,
    )
    self.assertEqual(list(psi_sources.keys()), ['current'])
    self.assertIsInstance(psi_sources['current'], jax.Array)
    np.testing.assert_allclose(psi_sources['current'], ones * 5.0)

  def test_qei_info_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    times = np.array([0.0, 1.0])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], geo.rho_norm, [1.0]]
        ),
    )
    p_ei = jnp.ones((2, geo.rho_norm.size)) * 5.0
    qei = core_sources_lib.QeiInfo(
        implicit_ii=jnp.zeros((2, geo.rho_norm.size)),
        explicit_i=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ee=jnp.zeros((2, geo.rho_norm.size)),
        explicit_e=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ie=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ei=jnp.zeros((2, geo.rho_norm.size)),
        p_ei=p_ei,
    )
    out_dict = qei.to_output_dict(context)
    self.assertIn('p_ei_exchange_i', out_dict)
    self.assertIn('p_ei_exchange_e', out_dict)
    dims_i, data_i, attrs_i = out_dict['p_ei_exchange_i']
    self.assertEqual(dims_i, (output_keys.TIME, output_keys.RHO_CELL_NORM))
    np.testing.assert_allclose(data_i, p_ei)
    self.assertEqual(attrs_i, {'units': output_keys.Units.WATT_PER_CUBIC_METER})
    dims_e, data_e, attrs_e = out_dict['p_ei_exchange_e']
    self.assertEqual(dims_e, (output_keys.TIME, output_keys.RHO_CELL_NORM))
    np.testing.assert_allclose(data_e, -p_ei)
    self.assertEqual(attrs_e, {'units': output_keys.Units.WATT_PER_CUBIC_METER})

  def test_bootstrap_current_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    times = np.array([0.0, 1.0])
    rho_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_norm,
    )
    j_cell = jnp.ones((2, geo.rho_norm.size)) * 3.0
    j_face = jnp.ones((2, geo.rho_face_norm.size)) * 4.0
    bc = bootstrap_current_base.BootstrapCurrent(
        j_parallel_bootstrap=j_cell,
        j_parallel_bootstrap_face=j_face,
    )
    out_dict = bc.to_output_dict(context)
    self.assertIn(str(output_keys.J_PARALLEL_BOOTSTRAP), out_dict)
    dims, data, attrs = out_dict[str(output_keys.J_PARALLEL_BOOTSTRAP)]
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_NORM))
    self.assertEqual(data.shape, (2, rho_norm.size))
    self.assertEqual(
        attrs, {'units': output_keys.Units.AMPERE_PER_SQUARE_METER}
    )

  def test_core_sources_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    times = np.array([0.0, 1.0])
    rho_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_norm,
    )
    n_cell = geo.rho_norm.size
    profiles = core_sources_lib.CoreSources(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.ones((2, n_cell)),
            j_parallel_bootstrap_face=jnp.ones((2, geo.rho_face_norm.size)),
        ),
        qei=core_sources_lib.QeiInfo(
            implicit_ii=jnp.zeros((2, n_cell)),
            explicit_i=jnp.zeros((2, n_cell)),
            implicit_ee=jnp.zeros((2, n_cell)),
            explicit_e=jnp.zeros((2, n_cell)),
            implicit_ie=jnp.zeros((2, n_cell)),
            implicit_ei=jnp.zeros((2, n_cell)),
            p_ei=jnp.ones((2, n_cell)) * 5.0,
        ),
        sources={
            'fusion': core_sources_lib.SourceOutput(
                power_density_i=jnp.ones((2, n_cell)) * 10.0
            ),
            'ecrh': core_sources_lib.SourceOutput(
                power_density_e=jnp.ones((2, n_cell)) * 20.0
            ),
            'eccd': core_sources_lib.SourceOutput(
                j_parallel=jnp.ones((2, n_cell)) * 30.0
            ),
            'pellet': core_sources_lib.SourceOutput(
                particle_source_e=jnp.ones((2, n_cell)) * 40.0
            ),
        },
    )

    out_dict = profiles.to_output_dict(context)
    self.assertIn(str(output_keys.J_PARALLEL_BOOTSTRAP), out_dict)
    self.assertIn('p_ei_exchange_i', out_dict)
    self.assertIn('p_ei_exchange_e', out_dict)
    self.assertIn('p_alpha_i', out_dict)
    self.assertIn('p_ecrh_e', out_dict)
    self.assertIn('j_parallel_eccd', out_dict)
    self.assertIn('s_pellet', out_dict)

    # Check dimensions and data
    self.assertEqual(
        out_dict['p_alpha_i'][0], (output_keys.TIME, output_keys.RHO_CELL_NORM)
    )
    np.testing.assert_allclose(
        out_dict['p_alpha_i'][1], np.ones((2, n_cell)) * 10.0
    )
    self.assertEqual(
        out_dict['p_alpha_i'][2],
        {'units': output_keys.Units.WATT_PER_CUBIC_METER},
    )

  def test_core_sources_to_output_dict_empty_channels(self):
    geo = circular_geometry.CircularConfig().build_runtime_params(0.0).geometry
    times = np.array([0.0, 1.0])
    rho_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_norm,
    )
    n_cell = geo.rho_norm.size
    profiles = core_sources_lib.CoreSources(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.zeros((2, n_cell)),
            j_parallel_bootstrap_face=jnp.zeros((2, geo.rho_face_norm.size)),
        ),
        qei=core_sources_lib.QeiInfo(
            implicit_ii=jnp.zeros((2, n_cell)),
            explicit_i=jnp.zeros((2, n_cell)),
            implicit_ee=jnp.zeros((2, n_cell)),
            explicit_e=jnp.zeros((2, n_cell)),
            implicit_ie=jnp.zeros((2, n_cell)),
            implicit_ei=jnp.zeros((2, n_cell)),
            p_ei=jnp.zeros((2, n_cell)),
        ),
    )
    out_dict = profiles.to_output_dict(context)
    self.assertCountEqual(
        out_dict.keys(),
        [
            output_keys.J_PARALLEL_BOOTSTRAP,
            'p_ei_exchange_i',
            'p_ei_exchange_e',
        ],
    )

  def test_iterate_channels_for_output_renaming(self):
    n_cell = 10
    profiles = core_sources_lib.CoreSources(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.zeros((1, n_cell)),
            j_parallel_bootstrap_face=jnp.zeros((1, n_cell + 1)),
        ),
        qei=core_sources_lib.QeiInfo(
            implicit_ii=jnp.zeros((1, n_cell)),
            explicit_i=jnp.zeros((1, n_cell)),
            implicit_ee=jnp.zeros((1, n_cell)),
            explicit_e=jnp.zeros((1, n_cell)),
            implicit_ie=jnp.zeros((1, n_cell)),
            implicit_ei=jnp.zeros((1, n_cell)),
            p_ei=jnp.zeros((1, n_cell)),
        ),
        sources={
            'fusion': core_sources_lib.SourceOutput(
                power_density_i=jnp.ones((1, n_cell))
            ),
            'ecrh': core_sources_lib.SourceOutput(
                power_density_e=jnp.ones((1, n_cell))
            ),
            'eccd_custom': core_sources_lib.SourceOutput(
                j_parallel=jnp.ones((1, n_cell))
            ),
            'pellet_custom': core_sources_lib.SourceOutput(
                particle_source_e=jnp.ones((1, n_cell))
            ),
        },
    )
    keys = [
        key
        for key, _ in profiles._iterate_channels_for_output(
            renames={
                'fusion': 'alpha',
                'eccd_custom': 'eccd',
                'pellet_custom': 'pellet',
            }
        )
    ]
    self.assertEqual(
        keys,
        ['p_alpha_i', 'p_ecrh_e', 'j_parallel_eccd', 's_pellet'],
    )
    self.assertEqual(
        [k.units for k in keys],
        [
            output_keys.Units.WATT_PER_CUBIC_METER,
            output_keys.Units.WATT_PER_CUBIC_METER,
            output_keys.Units.AMPERE_PER_SQUARE_METER,
            output_keys.Units.INVERSE_CUBIC_METER_PER_SECOND,
        ],
    )


if __name__ == '__main__':
  absltest.main()
