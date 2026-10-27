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
import dataclasses
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
from torax._src.sources import source_profiles as source_profiles_lib


# pylint: disable=invalid-name
class SourceProfilesTest(parameterized.TestCase):

  def test_summed_T_i_profiles_dont_change_when_jitting(self):
    geo = circular_geometry.CircularConfig().build_geometry()

    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones_like(geo.rho)
    profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        T_i={
            'generic_heat': ones,
            'fusion': ones * 3,
        },
        T_e={
            'generic_heat': ones * 2,
            'fusion': ones * 4,
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
        },
        n_e={},
        psi={},
    )
    with self.subTest('without_jit'):
      summed_T_i = profiles.total_sources('T_i', geo)
      np.testing.assert_allclose(summed_T_i, ones * 4 * geo.vpr)
      summed_T_e = profiles.total_sources('T_e', geo)
      np.testing.assert_allclose(summed_T_e, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp = jax.jit(profiles.total_sources, static_argnames='source_type')
      jitted_T_i = sum_temp('T_i', geo)
      np.testing.assert_allclose(jitted_T_i, ones * 4 * geo.vpr)
      jitted_T_e = sum_temp('T_e', geo)
      np.testing.assert_allclose(jitted_T_e, ones * 10 * geo.vpr)

  def test_from_source_outputs(self):
    geo = circular_geometry.CircularConfig(n_rho=10).build_geometry()
    ones = jnp.ones(10)

    @dataclasses.dataclass(frozen=True)
    class DummySource:
      AFFECTED_CORE_PROFILES: tuple[
          source_profiles_lib.AffectedCoreProfile, ...
      ]

    standard_sources = {
        'heat': DummySource((
            source_profiles_lib.AffectedCoreProfile.TEMP_ION,
            source_profiles_lib.AffectedCoreProfile.TEMP_EL,
        )),
        'particle': DummySource((source_profiles_lib.AffectedCoreProfile.NE,)),
        'current': DummySource((source_profiles_lib.AffectedCoreProfile.PSI,)),
    }
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
    qei = source_profiles_lib.QeiInfo.zeros(geo)
    source_outputs = {
        'heat': source_profiles_lib.SourceOutput(
            profiles=(ones * 2.0, ones * 3.0),
            fast_ions=(mock_fast_ion,),
        ),
        'particle': source_profiles_lib.SourceOutput(
            profiles=(ones * 4.0,),
        ),
        'current': source_profiles_lib.SourceOutput(
            profiles=(ones * 5.0,),
        ),
        'ignored_nonstandard': source_profiles_lib.SourceOutput(
            profiles=(ones * 99.0,),
        ),
    }
    profiles = source_profiles_lib.SourceProfiles.from_source_outputs(
        source_outputs=source_outputs,
        standard_sources=standard_sources,
        bootstrap_current=bootstrap_current,
        qei=qei,
    )
    self.assertIn('heat', profiles.T_i)
    self.assertIn('heat', profiles.T_e)
    self.assertIn('particle', profiles.n_e)
    self.assertIn('current', profiles.psi)
    self.assertNotIn('ignored_nonstandard', profiles.psi)
    np.testing.assert_allclose(profiles.T_i['heat'], ones * 2.0)
    np.testing.assert_allclose(profiles.T_e['heat'], ones * 3.0)
    np.testing.assert_allclose(profiles.n_e['particle'], ones * 4.0)
    np.testing.assert_allclose(profiles.psi['current'], ones * 5.0)
    self.assertEqual(profiles.fast_ions['heat'], (mock_fast_ion,))

  def test_qei_info_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_geometry()
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
    qei = source_profiles_lib.QeiInfo(
        implicit_ii=jnp.zeros((2, geo.rho_norm.size)),
        explicit_i=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ee=jnp.zeros((2, geo.rho_norm.size)),
        explicit_e=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ie=jnp.zeros((2, geo.rho_norm.size)),
        implicit_ei=jnp.zeros((2, geo.rho_norm.size)),
        p_ei=p_ei,
    )
    out_dict = qei.to_output_dict(context)
    self.assertIn(str(output_keys.EI_EXCHANGE), out_dict)
    dims, data, attrs = out_dict[str(output_keys.EI_EXCHANGE)]
    self.assertEqual(dims, (output_keys.TIME, output_keys.RHO_CELL_NORM))
    np.testing.assert_allclose(data, p_ei)
    self.assertEqual(
        attrs, {'units': output_keys.Units.WATT_PER_CUBIC_METER}
    )

  def test_bootstrap_current_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_geometry()
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

  def test_source_profiles_to_output_dict(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    times = np.array([0.0, 1.0])
    rho_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_norm,
    )
    n_cell = geo.rho_norm.size
    profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.ones((2, n_cell)),
            j_parallel_bootstrap_face=jnp.ones((2, geo.rho_face_norm.size)),
        ),
        qei=source_profiles_lib.QeiInfo(
            implicit_ii=jnp.zeros((2, n_cell)),
            explicit_i=jnp.zeros((2, n_cell)),
            implicit_ee=jnp.zeros((2, n_cell)),
            explicit_e=jnp.zeros((2, n_cell)),
            implicit_ie=jnp.zeros((2, n_cell)),
            implicit_ei=jnp.zeros((2, n_cell)),
            p_ei=jnp.ones((2, n_cell)) * 5.0,
        ),
        T_i={'fusion': jnp.ones((2, n_cell)) * 10.0},
        T_e={'ecrh': jnp.ones((2, n_cell)) * 20.0},
        psi={'eccd': jnp.ones((2, n_cell)) * 30.0},
        n_e={'pellet': jnp.ones((2, n_cell)) * 40.0},
    )

    out_dict = profiles.to_output_dict(context)
    self.assertIn(str(output_keys.J_PARALLEL_BOOTSTRAP), out_dict)
    self.assertIn(str(output_keys.EI_EXCHANGE), out_dict)
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

  def test_source_profiles_to_output_dict_empty_channels(self):
    geo = circular_geometry.CircularConfig().build_geometry()
    times = np.array([0.0, 1.0])
    rho_norm = np.concatenate([[0.0], geo.rho_norm, [1.0]])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=rho_norm,
    )
    n_cell = geo.rho_norm.size
    profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.zeros((2, n_cell)),
            j_parallel_bootstrap_face=jnp.zeros((2, geo.rho_face_norm.size)),
        ),
        qei=source_profiles_lib.QeiInfo(
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
        [output_keys.J_PARALLEL_BOOTSTRAP, output_keys.EI_EXCHANGE],
    )

  def test_iterate_channels_for_output_renaming(self):
    n_cell = 10
    profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent(
            j_parallel_bootstrap=jnp.zeros((1, n_cell)),
            j_parallel_bootstrap_face=jnp.zeros((1, n_cell + 1)),
        ),
        qei=source_profiles_lib.QeiInfo(
            implicit_ii=jnp.zeros((1, n_cell)),
            explicit_i=jnp.zeros((1, n_cell)),
            implicit_ee=jnp.zeros((1, n_cell)),
            explicit_e=jnp.zeros((1, n_cell)),
            implicit_ie=jnp.zeros((1, n_cell)),
            implicit_ei=jnp.zeros((1, n_cell)),
            p_ei=jnp.zeros((1, n_cell)),
        ),
        T_i={'fusion': jnp.ones((1, n_cell))},
        T_e={'ecrh': jnp.ones((1, n_cell))},
        psi={'eccd_custom': jnp.ones((1, n_cell))},
        n_e={'pellet_custom': jnp.ones((1, n_cell))},
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
