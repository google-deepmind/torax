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

import dataclasses
from absl.testing import absltest
import jax
from torax import config as config_lib
from torax import config_slice
from torax import core_profile_setters
from torax import geometry
from torax.sources import qei_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib


class QeiSourceTest(test_lib.SourceTestCase):
  """Tests for QeiSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=qei_source.QeiSource,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
        expected_affected_core_profiles=(
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.TEMP_EL,
        ),
    )

  def test_source_value(self):
    """Checks that the default implementation from Sources gives values."""
    source = qei_source.QeiSource()
    source_models = source_models_lib.SourceModels(
        sources={'qei_source': source}
    )
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    static_slice = config_slice.build_static_config_slice(config)
    dynamic_slice = config_slice.build_dynamic_config_slice(
        config,
        sources=source_models.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_config_slice=static_slice,
        dynamic_config_slice=dynamic_slice,
        geo=geo,
        source_models=source_models,
    )
    assert isinstance(source, qei_source.QeiSource)  # required for pytype.
    qei = source.get_qei(
        static_slice,
        dynamic_slice,
        dynamic_slice.sources['qei_source'],
        geo,
        core_profiles,
    )
    self.assertIsNotNone(qei)

  def test_invalid_source_types_raise_errors(self):
    source = qei_source.QeiSource()
    source_models = source_models_lib.SourceModels(
        sources={'qei_source': source}
    )
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    static_slice = config_slice.build_static_config_slice(config)
    dynamic_slice = config_slice.build_dynamic_config_slice(
        config,
        sources=source_models.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_config_slice=static_slice,
        dynamic_config_slice=dynamic_slice,
        geo=geo,
        source_models=source_models,
    )
    for unsupported_mode in self._unsupported_modes:
      with self.subTest(unsupported_mode.name):
        with self.assertRaises(jax.interpreters.xla.xe.XlaRuntimeError):
          dynamic_slice = config_slice.build_dynamic_config_slice(
              config,
              sources={
                  'qei_source': dataclasses.replace(
                      source.runtime_params, mode=unsupported_mode
                  )
              },
          )
          source.get_qei(
              static_slice,
              dynamic_slice,
              dynamic_slice.sources['qei_source'],
              geo,
              core_profiles,
          )


if __name__ == '__main__':
  absltest.main()
