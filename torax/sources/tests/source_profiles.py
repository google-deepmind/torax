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

"""Tests for sources and source profiles."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torax  # useful for setting up jax properly.
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax.sources import bootstrap_current_source
from torax.sources import source as source_lib
from torax.sources import source_config
from torax.sources import source_profiles
from torax.time_step_calculator import fixed_time_step_calculator


class SourcesTest(parameterized.TestCase):
  """Tests for Sources."""

  def test_default_constructor_works(self):
    """Tests that you can initialize the Sources class with no args."""
    _ = source_profiles.Sources()


class SourceProfilesTest(parameterized.TestCase):
  """Tests for computing source profiles."""

  def test_computing_source_profiles_works_with_all_defaults(self):
    """Tests that you can compute source profiles with all defaults."""
    config = torax.Config()
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    geo = torax.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sources = source_profiles.Sources()
    sim_state = initial_states.get_initial_sim_state(
        config, geo, ts_calculator, sources
    )
    _ = source_profiles.build_source_profiles(
        sources, dynamic_config_slice, geo, sim_state, explicit=True
    )
    _ = source_profiles.build_source_profiles(
        sources, dynamic_config_slice, geo, sim_state, explicit=False
    )

  def test_summed_temp_ion_profiles_dont_change_when_jitting(self):
    """Test that sum_sources_temp_{ion|el} works with jitting."""
    config = torax.Config()
    geo = torax.build_circular_geometry(config)

    # Use the default sources where the generic_ion_el_heat_source,
    # fusion_heat_source, and ohmic_heat_source are included and produce
    # profiles for ion and electron heat.
    # temperature.
    sources = source_profiles.Sources()
    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones(source_lib.ProfileType.CELL.get_profile_shape(geo))
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=_zero_bootstrap_profile(geo),
        profiles={
            'generic_ion_el_heat_source': jnp.stack([ones, ones * 2]),
            'fusion_heat_source': jnp.stack([ones * 3, ones * 4]),
            'ohmic_heat_source': ones * 5,  # only used for electron temp.
        },
    )
    with self.subTest('without_jit'):
      summed_temp_ion = source_profiles.sum_sources_temp_ion(
          sources, profiles, geo
      )
      np.testing.assert_allclose(summed_temp_ion, ones * 4 * geo.vpr)
      summed_temp_el = source_profiles.sum_sources_temp_el(
          sources, profiles, geo
      )
      np.testing.assert_allclose(summed_temp_el, ones * 11 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp_ion = jax.jit(
          source_profiles.sum_sources_temp_ion,
          static_argnames=['sources'],
      )
      jitted_temp_ion = sum_temp_ion(sources, profiles, geo)
      np.testing.assert_allclose(jitted_temp_ion, ones * 4 * geo.vpr)
      sum_temp_el = jax.jit(
          source_profiles.sum_sources_temp_el,
          static_argnames=['sources'],
      )
      jitted_temp_el = sum_temp_el(sources, profiles, geo)
      np.testing.assert_allclose(jitted_temp_el, ones * 11 * geo.vpr)

  def test_custom_source_profiles_dont_change_when_jitted(self):
    """Test that custom source profiles don't change profiles when jitted."""
    source_name = 'foo'

    def foo_formula(unused_dcs, geo: geometry.Geometry, unused_state):
      return jnp.stack([
          jnp.zeros(source_lib.ProfileType.CELL.get_profile_shape(geo)),
          jnp.ones(source_lib.ProfileType.CELL.get_profile_shape(geo)),
      ])

    foo_source = source_lib.Source(
        name=source_name,
        # Test a fake source that somehow affects both electron temp and
        # electron density.
        affected_mesh_states=(
            source_lib.AffectedMeshStateAttribute.TEMP_EL,
            source_lib.AffectedMeshStateAttribute.NE,
        ),
        supported_types=(source_config.SourceType.FORMULA_BASED,),
        output_shape_getter=lambda _0, geo, _1: (2,)
        + source_lib.ProfileType.CELL.get_profile_shape(geo),
        formula=foo_formula,
    )
    sources = source_profiles.Sources(
        additional_sources=[foo_source],
    )
    zero_config = source_config.SourceConfig(
        source_type=source_config.SourceType.ZERO
    )
    config = torax.Config(
        sources=dict(
            # Turn off all the other ne sources.
            gas_puff_source=zero_config,
            nbi_particle_source=zero_config,
            pellet_source=zero_config,
            # And turn off the temp sources.
            generic_ion_el_heat_source=zero_config,
            fusion_heat_source=zero_config,
            ohmic_heat_source=zero_config,
            # But for the custom source, leave that on.
            foo=source_config.SourceConfig(
                source_type=source_config.SourceType.FORMULA_BASED,
            ),
        )
    )
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    geo = torax.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config, geo, ts_calculator, sources
    )

    def compute_and_sum_profiles():
      profiles = source_profiles.build_source_profiles(
          sources=sources,
          dynamic_config_slice=dynamic_config_slice,
          geo=geo,
          sim_state=sim_state,
          # Configs set sources to implicit by default, so set this to False to
          # calculate the custom source's profile.
          explicit=False,
      )
      ne = source_profiles.sum_sources_ne(sources, profiles, geo)
      temp_el = source_profiles.sum_sources_temp_el(sources, profiles, geo)
      return (ne, temp_el)

    expected_ne = (
        jnp.ones(source_lib.ProfileType.CELL.get_profile_shape(geo)) * geo.vpr
    )
    expected_temp_el = jnp.zeros(
        source_lib.ProfileType.CELL.get_profile_shape(geo)
    )
    with self.subTest('without_jit'):
      (ne, temp_el) = compute_and_sum_profiles()
      np.testing.assert_allclose(ne, expected_ne)
      np.testing.assert_allclose(temp_el, expected_temp_el)
    with self.subTest('with_jit'):
      jitted_compute_and_sum = jax.jit(compute_and_sum_profiles)
      (ne, temp_el) = jitted_compute_and_sum()
      np.testing.assert_allclose(ne, expected_ne)
      np.testing.assert_allclose(temp_el, expected_temp_el)


def _zero_bootstrap_profile(
    geo: torax.Geometry,
) -> bootstrap_current_source.BootstrapCurrentProfile:
  """Returns a dummy boostrap current profile with everything set to zero."""
  cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
  face = source_lib.ProfileType.FACE.get_profile_shape(geo)
  return bootstrap_current_source.BootstrapCurrentProfile(
      sigma=jnp.zeros(cell),
      j_bootstrap=jnp.zeros(cell),
      j_bootstrap_face=jnp.zeros(face),
      I_bootstrap=jnp.zeros((1,)),
  )


if __name__ == '__main__':
  absltest.main()
