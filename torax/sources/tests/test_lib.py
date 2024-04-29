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
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib


# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  _source_class: Type[source_lib.Source]
  _config_attr_name: str
  _unsupported_modes: Sequence[runtime_params_lib.Mode]
  _expected_affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...]

  @classmethod
  def setUpClass(
      cls,
      source_class: Type[source_lib.Source],
      unsupported_modes: Sequence[runtime_params_lib.Mode],
      expected_affected_core_profiles: tuple[
          source_lib.AffectedCoreProfile, ...
      ],
  ):
    super().setUpClass()
    cls._source_class = source_class
    cls._unsupported_modes = unsupported_modes
    cls._expected_affected_core_profiles = expected_affected_core_profiles

  def test_expected_mesh_states(self):
    # Most Source subclasses should have default names and be instantiable
    # without any __init__ arguments.
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertSameElements(
        source.affected_core_profiles,
        self._expected_affected_core_profiles,
    )


class SingleProfileSourceTestCase(SourceTestCase):
  """Base test class for SingleProfileSource subclasses."""

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    # SingleProfileSource subclasses should have default names and be
    # instantiable without any __init__ arguments.
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.SingleProfileSource)
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source.runtime_params.mode = source.supported_modes[0]
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    geo = geometry.build_circular_geometry()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    chex.assert_rank(value, 1)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.SingleProfileSource)
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    for unsupported_mode in self._unsupported_modes:
      source.runtime_params.mode = unsupported_mode
      dynamic_runtime_params_slice = (
          runtime_params_slice.build_dynamic_runtime_params_slice(
              runtime_params=runtime_params,
              sources=source_models.runtime_params,
          )
      )
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_value(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
                  'foo'
              ],
              geo=geo,
              core_profiles=core_profiles,
          )


class IonElSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.IonElectronSource)
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    ion_and_el = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            'foo'
        ],
        geo=geo,
        core_profiles=core_profiles,
    )
    chex.assert_rank(ion_and_el, 2)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.IonElectronSource)
    source_models = source_models_lib.SourceModels(
        sources={'foo': source},
    )
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params=runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    for unsupported_mode in self._unsupported_modes:
      source.runtime_params.mode = unsupported_mode
      dynamic_runtime_params_slice = (
          runtime_params_slice.build_dynamic_runtime_params_slice(
              runtime_params=runtime_params,
              sources=source_models.runtime_params,
          )
      )
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_value(
              dynamic_runtime_params_slice=dynamic_runtime_params_slice,
              dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
                  'foo'
              ],
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
