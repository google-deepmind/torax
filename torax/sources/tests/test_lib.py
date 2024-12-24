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

from typing import Sequence, Type

from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib


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

  @property
  def supported_modes(self) -> tuple[runtime_params_lib.Mode, ...]:
    return (
        runtime_params_lib.Mode.ZERO,
        runtime_params_lib.Mode.FORMULA_BASED,
        runtime_params_lib.Mode.MODEL_BASED,
        runtime_params_lib.Mode.PRESCRIBED,
    )


TestSourceBuilder = source_lib.make_source_builder(TestSource)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  _source_class: Type[source_lib.Source]
  _source_class_builder: source_lib.SourceBuilderProtocol
  _config_attr_name: str
  _unsupported_modes: Sequence[runtime_params_lib.Mode]
  _source_name: str
  _runtime_params_class: Type[runtime_params_lib.RuntimeParams]

  @classmethod
  def setUpClass(
      cls,
      source_class: Type[source_lib.Source],
      runtime_params_class: Type[runtime_params_lib.RuntimeParams],
      unsupported_modes: Sequence[runtime_params_lib.Mode],
      source_name: str,
      model_func: source_lib.SourceProfileFunction | None,
      links_back: bool = False,
      source_class_builder: source_lib.SourceBuilderProtocol | None = None,
  ):
    super().setUpClass()
    cls._source_class = source_class
    if source_class_builder is None:
      cls._source_class_builder = source_lib.make_source_builder(
          source_type=source_class,
          runtime_params_type=runtime_params_class,
          links_back=links_back,
          model_func=model_func,
      )
    else:
      cls._source_class_builder = source_class_builder
    cls._runtime_params_class = runtime_params_class
    cls._unsupported_modes = unsupported_modes
    cls._links_back = links_back
    cls._source_name = source_name

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = self._runtime_params_class()
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)
    geo = geometry.build_circular_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_params = provider.build_dynamic_params(t=0.0)
    self.assertIsInstance(
        dynamic_params, runtime_params_lib.DynamicRuntimeParams
    )

  @parameterized.product(
      mode=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.MODEL_BASED,
          runtime_params_lib.Mode.FORMULA_BASED,
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

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
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
    geo = geometry.build_circular_geometry()
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
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles,
    )
    chex.assert_rank(value, 1)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
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
    for unsupported_mode in self._unsupported_modes:
      source_builder.runtime_params.mode = unsupported_mode
      static_slice = runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          source_runtime_params=source_models_builder.runtime_params,
          torax_mesh=geo.torax_mesh,
      )
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(ValueError):
          source.get_value(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              static_runtime_params_slice=static_slice,
              geo=geo,
              core_profiles=core_profiles,
          )


class IonElSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()
    # pylint: enable=missing-kwoa
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
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
    )
    chex.assert_rank(ion_and_el, 2)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
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
    for unsupported_mode in self._unsupported_modes:
      source_builder.runtime_params.mode = unsupported_mode
      static_slice = runtime_params_slice.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          source_runtime_params=source_models_builder.runtime_params,
          torax_mesh=geo.torax_mesh,
      )
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(ValueError):
          source.get_value(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              static_runtime_params_slice=static_slice,
              geo=geo,
              core_profiles=core_profiles,
          )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    geo = geometry.build_circular_geometry()
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    fake_profile = jnp.stack((jnp.ones(cell), 2 * jnp.ones(cell)))
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_ION.value,
            geo,
        ),
        jnp.ones(cell),
    )
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_EL.value,
            geo,
        ),
        2 * jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.NE.value,
            geo,
        ),
        jnp.zeros(cell),
    )
