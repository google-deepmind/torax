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
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax.sources import source as source_lib
from torax.sources import source_config as source_config_lib
from torax.sources import source_profiles
from torax.time_step_calculator import fixed_time_step_calculator


# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class SourceTestCase(parameterized.TestCase):
  """Base test class for sources.

  Extend this class for source-specific tests.
  """

  _source_class: Type[source_lib.Source]
  _config_attr_name: str
  _unsupported_types: Sequence[source_config_lib.SourceType]
  _expected_affected_mesh_states: tuple[
      source_lib.AffectedMeshStateAttribute, ...
  ]

  @classmethod
  def setUpClass(
      cls,
      source_class: Type[source_lib.Source],
      unsupported_types: Sequence[source_config_lib.SourceType],
      expected_affected_mesh_states: tuple[
          source_lib.AffectedMeshStateAttribute, ...
      ],
  ):
    super().setUpClass()
    cls._source_class = source_class
    cls._unsupported_types = unsupported_types
    cls._expected_affected_mesh_states = expected_affected_mesh_states

  def test_expected_mesh_states(self):
    # Most Source subclasses should have default names and be instantiable
    # without any __init__ arguments.
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertSameElements(
        source.affected_mesh_states,
        self._expected_affected_mesh_states,
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
    config = config_lib.Config()
    # Not all sources are in the default config, so add the source in here if
    # it doesn't already exist.
    if source.name not in config.sources:
      supported_types = set(
          [source_type for source_type in source_config_lib.SourceType]
      ) - set(self._unsupported_types)
      supported_type = supported_types.pop()
      config = config_lib.Config(
          sources={
              source.name: source_config_lib.SourceConfig(
                  source_type=supported_type,
              )
          }
      )
      sources = source_profiles.Sources(additional_sources=[source])
    else:
      sources = source_profiles.Sources()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=sources,
    )
    source_type = config.sources[source.name].source_type.value
    value = source.get_value(
        source_type=source_type,
        dynamic_config_slice=(config_slice.build_dynamic_config_slice(config)),
        geo=geo,
        sim_state=sim_state,
    )
    chex.assert_rank(value, 1)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(),  # only need default sources here.
    )
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.SingleProfileSource)
    for unsupported_type in self._unsupported_types:
      with self.subTest(unsupported_type.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_value(
              source_type=unsupported_type.value,
              dynamic_config_slice=(
                  config_slice.build_dynamic_config_slice(config)
              ),
              geo=geo,
              sim_state=sim_state,
          )


class IonElSourceTestCase(SourceTestCase):
  """Base test class for IonElSource subclasses."""

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.IonElectronSource)
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(),  # only need default sources here.
    )
    source_type = config.sources[source.name].source_type.value
    ion_and_el = source.get_value(
        source_type=source_type,
        dynamic_config_slice=(config_slice.build_dynamic_config_slice(config)),
        geo=geo,
        sim_state=sim_state,
    )
    chex.assert_rank(ion_and_el, 2)

  def test_invalid_source_types_raise_errors(self):
    """Tests that using unsupported types raises an error."""
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(),  # only need default sources here.
    )
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    self.assertIsInstance(source, source_lib.IonElectronSource)
    for unsupported_type in self._unsupported_types:
      with self.subTest(unsupported_type.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_value(
              source_type=unsupported_type.value,
              dynamic_config_slice=(
                  config_slice.build_dynamic_config_slice(config)
              ),
              geo=geo,
              sim_state=sim_state,
          )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    # pylint: disable=missing-kwoa
    source = self._source_class()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    fake_profile = jnp.stack((jnp.ones(cell), 2 * jnp.ones(cell)))
    np.testing.assert_allclose(
        source.get_profile_for_affected_state(
            fake_profile,
            source_lib.AffectedMeshStateAttribute.TEMP_ION.value,
            geo,
        ),
        jnp.ones(cell),
    )
    np.testing.assert_allclose(
        source.get_profile_for_affected_state(
            fake_profile,
            source_lib.AffectedMeshStateAttribute.TEMP_EL.value,
            geo,
        ),
        2 * jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_profile_for_affected_state(
            fake_profile,
            source_lib.AffectedMeshStateAttribute.NE.value,
            geo,
        ),
        jnp.zeros(cell),
    )
