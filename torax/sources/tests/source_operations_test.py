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
from torax import core_profile_setters
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.geometry import geometry
from torax.sources import runtime_params as source_runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_operations
from torax.sources import source_profile_builders
from torax.sources import source_profiles as source_profiles_lib


@dataclasses.dataclass(frozen=True)
class FooSource(source_lib.Source):
  """A test source."""

  @property
  def source_name(self) -> str:
    return 'foo'

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (
        source_lib.AffectedCoreProfile.TEMP_EL,
        source_lib.AffectedCoreProfile.NE,
    )


class SourceOperationsTest(parameterized.TestCase):

  def test_summed_temp_ion_profiles_dont_change_when_jitting(self):
    geo = circular_geometry.build_circular_geometry()

    # Make some dummy source profiles that could have come from these sources.
    ones = jnp.ones_like(geo.rho)
    profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
        temp_ion={
            'generic_ion_el_heat_source': ones,
            'fusion_heat_source': ones * 3,
        },
        temp_el={
            'generic_ion_el_heat_source': ones * 2,
            'fusion_heat_source': ones * 4,
            'bremsstrahlung_heat_sink': -ones,
            'ohmic_heat_source': ones * 5,
        },
        ne={},
        psi={},
    )
    with self.subTest('without_jit'):
      summed_temp_ion = source_operations.sum_sources_temp_ion(geo, profiles)
      np.testing.assert_allclose(summed_temp_ion, ones * 4 * geo.vpr)
      summed_temp_el = source_operations.sum_sources_temp_el(geo, profiles)
      np.testing.assert_allclose(summed_temp_el, ones * 10 * geo.vpr)

    with self.subTest('with_jit'):
      sum_temp_ion = jax.jit(
          source_operations.sum_sources_temp_ion,
      )
      jitted_temp_ion = sum_temp_ion(geo, profiles)
      np.testing.assert_allclose(jitted_temp_ion, ones * 4 * geo.vpr)
      sum_temp_el = jax.jit(
          source_operations.sum_sources_temp_el,
      )
      jitted_temp_el = sum_temp_el(geo, profiles)
      np.testing.assert_allclose(jitted_temp_el, ones * 10 * geo.vpr)

  def test_custom_source_profiles_dont_change_when_jitted(self):
    source_name = 'foo'

    def foo_formula(
        unused_dcs,
        unused_static_runtime_params_slice,
        geo: geometry.Geometry,
        unused_source_name: str,
        unused_state,
        unused_calculated_source_profiles,
        unused_source_models,
    ):
      return jnp.stack([
          jnp.zeros_like(geo.rho),
          jnp.ones_like(geo.rho),
      ])

    foo_source_builder = source_lib.make_source_builder(
        FooSource, model_func=foo_formula
    )()
    # Set the source mode to MODEL_BASED.
    foo_source_builder.runtime_params.mode = (
        source_runtime_params_lib.Mode.MODEL_BASED
    )
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {source_name: foo_source_builder},
    )
    source_models = source_models_builder()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    geo = circular_geometry.build_circular_geometry()
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
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    def compute_and_sum_profiles():
      profiles = source_profile_builders.build_source_profiles(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          static_runtime_params_slice=static_slice,
          geo=geo,
          core_profiles=core_profiles,
          source_models=source_models,
          # Configs set sources to implicit by default, so set this to False to
          # calculate the custom source's profile.
          explicit=False,
      )
      ne = source_operations.sum_sources_ne(geo, profiles)
      temp_el = source_operations.sum_sources_temp_el(
          geo, profiles
      )
      return (ne, temp_el)

    expected_ne = jnp.full(geo.rho.shape, geo.vpr)
    expected_temp_el = jnp.zeros_like(geo.rho)
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
