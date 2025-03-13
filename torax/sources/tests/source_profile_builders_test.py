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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax.config import build_runtime_params
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.stepper import pydantic_model as stepper_pydantic_model


class SourceModelsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = geometry_pydantic_model.CircularConfig().build_geometry()

  def test_computing_source_profiles_works_with_all_defaults(self):
    """Tests that you can compute source profiles with all defaults."""
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    sources = source_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            sources=sources,
            torax_mesh=self.geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=self.geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=self.geo,
        source_models=source_models,
    )
    stepper_params = stepper_pydantic_model.Stepper()
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=self.geo.torax_mesh,
            stepper=stepper_params,
        )
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        self.geo,
        core_profiles,
        source_models,
        explicit=True,
    )
    source_profile_builders.build_source_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        self.geo,
        core_profiles,
        source_models,
        explicit=False,
        explicit_source_profiles=explicit_source_profiles,
    )

  def test_computing_standard_source_profiles_for_single_affected_core_profile(
      self,
  ):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
    class TestSource(source.Source):

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(
          self,
      ) -> tuple[source.AffectedCoreProfile, ...]:
        return (source.AffectedCoreProfile.PSI,)

    test_source = TestSource(
        model_func=lambda *args: (jnp.ones(self.geo.rho.shape),)
    )
    source_models = mock.create_autospec(
        source_models_lib.SourceModels, standard_sources={'foo': test_source}
    )
    test_source_runtime_params = source_runtime_params.StaticRuntimeParams(
        mode='MODEL_BASED', is_explicit=True
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={'foo': test_source_runtime_params},
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            'foo': source_runtime_params.DynamicRuntimeParams(
                prescribed_values=jnp.ones(self.geo.rho.shape)
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_params,
        dynamic_runtime_params_slice=dynamic_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        explicit=True,
        calculated_source_profiles=profiles,
    )
    psi_profiles = profiles.psi
    self.assertLen(psi_profiles, 1)
    self.assertIn('foo', psi_profiles)
    np.testing.assert_equal(psi_profiles['foo'].shape, self.geo.rho.shape)

  def test_computing_standard_source_profiles_for_multiple_affected_core_profile(
      self,
  ):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
    class TestSource(source.Source):

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(
          self,
      ) -> tuple[source.AffectedCoreProfile, ...]:
        return (
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        )

    test_source = TestSource(
        model_func=lambda *args: (jnp.ones_like(self.geo.rho),) * 2
    )
    source_models = mock.create_autospec(
        source_models_lib.SourceModels, standard_sources={'foo': test_source}
    )
    test_source_runtime_params = source_runtime_params.StaticRuntimeParams(
        mode='MODEL_BASED', is_explicit=True
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={'foo': test_source_runtime_params},
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            'foo': source_runtime_params.DynamicRuntimeParams(
                prescribed_values=jnp.ones(self.geo.rho.shape)
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_params,
        dynamic_runtime_params_slice=dynamic_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        explicit=True,
        calculated_source_profiles=profiles,
    )

    # Check that a single profile is returned for each affected core profile.
    # These profiles should be the same shape as the geo.rho.
    ion_profiles = profiles.temp_ion
    self.assertLen(ion_profiles, 1)
    self.assertIn('foo', ion_profiles)
    np.testing.assert_equal(ion_profiles['foo'].shape, self.geo.rho.shape)

    el_profiles = profiles.temp_el
    self.assertLen(el_profiles, 1)
    self.assertIn('foo', el_profiles)
    np.testing.assert_equal(el_profiles['foo'].shape, self.geo.rho.shape)

  @parameterized.parameters(
      dict(
          calculate_anyway=True,
          is_explicit=True,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=True,
          is_explicit=False,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=False,
          is_explicit=True,
          expected_calculate=True,
      ),
      dict(
          calculate_anyway=False,
          is_explicit=False,
          expected_calculate=False,
      ),
  )
  def test_build_standard_source_profiles_calculate_anyway(
      self, calculate_anyway, is_explicit, expected_calculate
  ):
    @dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
    class TestSource(source.Source):

      @property
      def source_name(self) -> str:
        return 'foo'

      @property
      def affected_core_profiles(
          self,
      ) -> tuple[source.AffectedCoreProfile, ...]:
        return (source.AffectedCoreProfile.PSI,)

    test_source = TestSource(
        model_func=lambda *args: (jnp.ones(self.geo.rho.shape),)
    )
    source_models = mock.create_autospec(
        source_models_lib.SourceModels, standard_sources={'foo': test_source}
    )
    test_source_runtime_params = source_runtime_params.StaticRuntimeParams(
        mode='MODEL_BASED', is_explicit=True  # Set the source to be explicit.
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={'foo': test_source_runtime_params},
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        sources={
            'foo': source_runtime_params.DynamicRuntimeParams(
                prescribed_values=jnp.ones(self.geo.rho.shape)
            )
        },
    )
    profiles = source_profiles.SourceProfiles(
        j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles.QeiInfo.zeros(self.geo),
    )
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_params,
        dynamic_runtime_params_slice=dynamic_params,
        geo=self.geo,
        core_profiles=mock.ANY,
        source_models=source_models,
        explicit=is_explicit,
        calculated_source_profiles=profiles,
        calculate_anyway=calculate_anyway,
    )

    if expected_calculate:
      self.assertIn('foo', profiles.psi)
    else:
      self.assertNotIn('foo', profiles.psi)


if __name__ == '__main__':
  absltest.main()
