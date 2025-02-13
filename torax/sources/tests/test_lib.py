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

"""Utilities to help with testing sources."""

from typing import Type

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import circular_geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles


# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class TestSource(source_lib.Source):
  """A test source."""

  @property
  def source_name(self) -> str:
    return 'foo'

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (source_lib.AffectedCoreProfile.NE,)


TestSourceBuilder = source_lib.make_source_builder(TestSource)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  _source_class: Type[source_lib.Source]
  _source_class_builder: source_lib.SourceBuilderProtocol
  _config_attr_name: str
  _source_name: str
  _runtime_params_class: Type[runtime_params_lib.RuntimeParams]
  _needs_source_models: bool

  @classmethod
  def setUpClass(
      cls,
      source_class: Type[source_lib.Source],
      runtime_params_class: Type[runtime_params_lib.RuntimeParams],
      source_name: str,
      model_func: source_lib.SourceProfileFunction | None,
      needs_source_models: bool = False,
      source_class_builder: source_lib.SourceBuilderProtocol | None = None,
  ):
    super().setUpClass()
    cls._source_class = source_class
    if source_class_builder is None:
      cls._source_class_builder = source_lib.make_source_builder(
          source_type=source_class,
          runtime_params_type=runtime_params_class,
          model_func=model_func,
      )
    else:
      cls._source_class_builder = source_class_builder
    cls._runtime_params_class = runtime_params_class
    cls._source_name = source_name
    cls._needs_source_models = needs_source_models

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = self._runtime_params_class()
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)
    geo = circular_geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_params = provider.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_params, runtime_params_lib.DynamicRuntimeParams
    )

  @parameterized.product(
      mode=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.MODEL_BASED,
          runtime_params_lib.Mode.PRESCRIBED,
      ),
      is_explicit=(True, False),
  )
  def test_runtime_params_builds_static_params(
      self, mode: runtime_params_lib.Mode, is_explicit: bool
  ):
    """Tests that the static params are built correctly."""
    runtime_params = self._runtime_params_class()
    runtime_params.mode = mode
    runtime_params.is_explicit = is_explicit
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)
    static_params = runtime_params.build_static_params()
    self.assertIsInstance(static_params, runtime_params_lib.StaticRuntimeParams)
    self.assertEqual(static_params.mode, mode.value)
    self.assertEqual(static_params.is_explicit, is_explicit)


class SingleProfileSourceTestCase(SourceTestCase):
  """Base test class for SingleProfileSource subclasses."""

  def test_source_value_on_the_cell_grid(self):
    """Tests that the source can provide a value by default on the cell grid."""
    # SingleProfileSource subclasses should have default names and be
    # instantiable without any __init__ arguments.
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()
    if not source_lib.is_source_builder(source_builder):
      raise TypeError(f'{type(self)} has a bad _source_class_builder')
    # pylint: enable=missing-kwoa
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {self._source_name: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[self._source_name]
    source_builder.runtime_params.mode = runtime_params_lib.Mode.MODEL_BASED
    self.assertIsInstance(source, source_lib.Source)
    geo = circular_geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
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
    if self._needs_source_models:
      calculated_source_profiles = source_profiles.SourceProfiles(
          j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
          psi={'foo': jnp.full(geo.rho.shape, 13.0)},
          temp_el={'foo_source': jnp.full(geo.rho.shape, 17.0)},
          temp_ion={'foo_sink': jnp.full(geo.rho.shape, 19.0)},
          ne={},
          qei=source_profiles.QeiInfo.zeros(geo)
      )
    else:
      calculated_source_profiles = None
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=calculated_source_profiles,
    )[0]
    chex.assert_rank(value, 1)
    self.assertEqual(value.shape, geo.rho.shape)


class IonElSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_values_on_the_cell_grid(self):
    """Tests that the source can provide values on the cell grid."""
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()
    # pylint: enable=missing-kwoa
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = circular_geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {self._source_name: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[self._source_name]
    self.assertIsInstance(source, source_lib.Source)
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
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
    ion_and_el = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )
    self.assertLen(ion_and_el, 2)
    self.assertEqual(ion_and_el[0].shape, geo.rho.shape)
    self.assertEqual(ion_and_el[1].shape, geo.rho.shape)
