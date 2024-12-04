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

"""Tests for SourceModels and functions computing the source profiles."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import torax  # useful for setting up jax properly.
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.stepper import runtime_params as stepper_runtime_params_lib
from torax.tests.test_lib import default_sources


@dataclasses.dataclass(frozen=True)
class FooSource(source_lib.Source):
  """A test source."""

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (
        source_lib.AffectedCoreProfile.TEMP_EL,
        source_lib.AffectedCoreProfile.NE,
    )

  @property
  def output_shape_getter(self) -> source_lib.SourceOutputShapeFunction:
    return source_lib.get_ion_el_output_shape

  @property
  def supported_modes(self) -> tuple[runtime_params_lib.Mode, ...]:
    return (
        runtime_params_lib.Mode.FORMULA_BASED,
    )


_FooSourceBuilder = source_lib.make_source_builder(
    FooSource,
)


class SourceModelsTest(parameterized.TestCase):
  """Tests for SourceModels."""

  def test_default_constructor_works(self):
    """Tests that you can initialize the SourceModels class with no args."""
    _ = source_models_lib.SourceModels()


class SourceProfilesTest(parameterized.TestCase):
  """Tests for computing source profiles."""

  def test_computing_source_profiles_works_with_all_defaults(self):
    """Tests that you can compute source profiles with all defaults."""
    runtime_params = torax.GeneralRuntimeParams()
    geo = torax.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    stepper_params = stepper_runtime_params_lib.RuntimeParams()
    static_runtime_params_slice = (
        runtime_params_slice.build_static_runtime_params_slice(
            runtime_params,
            stepper=stepper_params,
            source_runtime_params=source_models_builder.runtime_params,
        )
    )
    _ = source_models_lib.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        source_models,
        explicit=True,
    )
    _ = source_models_lib.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        source_models,
        explicit=False,
    )

  def test_summed_temp_ion_profiles_dont_change_when_jitting(self):
    """Test that sum_sources_temp_{ion|el} works with jitting."""
    geo = torax.build_circular_geometry()

    # Use the default sources where the generic_ion_el_heat_source,
    # fusion_heat_source, and ohmic_heat_source are included and produce
    # profiles for ion and electron heat.
    # temperature.
    source_models_builder = default_sources.get_default_sources_builder()
    source_models = source_models_builder()
    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones(source_lib.ProfileType.CELL.get_profile_shape(geo))
    profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        profiles={
            'generic_ion_el_heat_source': jnp.stack([ones, ones * 2]),
            'fusion_heat_source': jnp.stack([ones * 3, ones * 4]),
            'bremsstrahlung_heat_sink': -ones,
            'ohmic_heat_source': ones * 5,  # only used for electron temp.
        },
    )
    with self.subTest('without_jit'):
      summed_temp_ion = source_models_lib.sum_sources_temp_ion(
          geo, profiles, source_models
      )
      np.testing.assert_allclose(summed_temp_ion, ones * 4 * geo.vpr)
      summed_temp_el = source_models_lib.sum_sources_temp_el(
          geo, profiles, source_models
      )
      np.testing.assert_allclose(summed_temp_el, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp_ion = jax.jit(
          source_models_lib.sum_sources_temp_ion,
          static_argnames=['source_models'],
      )
      jitted_temp_ion = sum_temp_ion(geo, profiles, source_models)
      np.testing.assert_allclose(jitted_temp_ion, ones * 4 * geo.vpr)
      sum_temp_el = jax.jit(
          source_models_lib.sum_sources_temp_el,
          static_argnames=['source_models'],
      )
      jitted_temp_el = sum_temp_el(geo, profiles, source_models)
      np.testing.assert_allclose(jitted_temp_el, ones * 10 * geo.vpr)

  def test_custom_source_profiles_dont_change_when_jitted(self):
    """Test that custom source profiles don't change profiles when jitted."""
    source_name = 'foo'

    def foo_formula(
        unused_dcs,
        unused_sc,
        unused_static_runtime_params_slice,
        unused_static_source_runtime_params,
        geo: geometry.Geometry,
        unused_state,
        unused_source_models,
    ):
      return jnp.stack([
          jnp.zeros(source_lib.ProfileType.CELL.get_profile_shape(geo)),
          jnp.ones(source_lib.ProfileType.CELL.get_profile_shape(geo)),
      ])

    foo_source_builder = _FooSourceBuilder(
        formula=foo_formula,
    )
    foo_source_builder.affected_core_profiles = (
        source_lib.AffectedCoreProfile.TEMP_EL,
        source_lib.AffectedCoreProfile.NE,
    )
    # Set the source mode to FORMULA.
    foo_source_builder.runtime_params.mode = (
        runtime_params_lib.Mode.FORMULA_BASED
    )
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {source_name: foo_source_builder},
    )
    source_models = source_models_builder()
    runtime_params = torax.GeneralRuntimeParams()
    geo = torax.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    def compute_and_sum_profiles():
      profiles = source_models_lib.build_source_profiles(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          static_runtime_params_slice=static_slice,
          geo=geo,
          core_profiles=core_profiles,
          source_models=source_models,
          # Configs set sources to implicit by default, so set this to False to
          # calculate the custom source's profile.
          explicit=False,
      )
      ne = source_models_lib.sum_sources_ne(geo, profiles, source_models)
      temp_el = source_models_lib.sum_sources_temp_el(
          geo, profiles, source_models
      )
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


if __name__ == '__main__':
  absltest.main()
