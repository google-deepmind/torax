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

"""Tests for qei_source."""

from absl.testing import absltest
import jax
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax.sources import qei_source
from torax.sources import source as source_lib
from torax.sources import source_config
from torax.sources import source_profiles
from torax.sources.tests import test_lib
from torax.time_step_calculator import fixed_time_step_calculator


class QeiSourceTest(test_lib.SourceTestCase):
  """Tests for QeiSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=qei_source.QeiSource,
        unsupported_types=[
            source_config.SourceType.FORMULA_BASED,
        ],
        expected_affected_mesh_states=(
            source_lib.AffectedMeshStateAttribute.TEMP_ION,
            source_lib.AffectedMeshStateAttribute.TEMP_EL,
        ),
    )

  def test_source_value(self):
    """Checks that the default implementation from Sources gives values."""
    source = qei_source.QeiSource()
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(qei_source=source),
    )
    assert isinstance(source, qei_source.QeiSource)  # required for pytype.
    dynamic_slice = config_slice.build_dynamic_config_slice(config)
    static_slice = config_slice.build_static_config_slice(config)
    qei = source.get_qei(
        dynamic_slice.sources[source.name].source_type,
        dynamic_slice,
        static_slice,
        geo,
        sim_state,
    )
    self.assertIsNotNone(qei)

  def test_invalid_source_types_raise_errors(self):
    source = qei_source.QeiSource()
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(qei_source=source),
    )
    dynamic_slice = config_slice.build_dynamic_config_slice(config)
    static_slice = config_slice.build_static_config_slice(config)
    for unsupported_type in self._unsupported_types:
      with self.subTest(unsupported_type.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          source.get_qei(
              unsupported_type.value,
              dynamic_slice,
              static_slice,
              geo,
              sim_state,
          )


if __name__ == '__main__':
  absltest.main()
