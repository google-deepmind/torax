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
from absl.testing import absltest
import jax.numpy as jnp
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import electron_cyclotron_source
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.stepper import pydantic_model as stepper_pydantic_model


class ElectronCyclotronSourceTest(test_lib.SourceTestCase):
  """Tests for ElectronCyclotronSource."""

  # pytype: disable=signature-mismatch
  def setUp(self):
    super().setUp(
        source_config_class=electron_cyclotron_source.ElectronCyclotronSourceConfig,
        source_name=electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME,
    )
  # pytype: enable=signature-mismatch

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    sources = source_pydantic_model.Sources.from_dict({self._source_name: {}})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources[self._source_name]
    self.assertIsInstance(source, source_lib.Source)
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
            stepper=stepper_pydantic_model.Stepper(),
        )
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    value = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
    )
    # ElectronCyclotronSource provides TEMP_EL and PSI
    self.assertLen(value, 2)
    # ElectronCyclotronSource default model_func provides sane default values
    self.assertFalse(jnp.any(jnp.isnan(value[0])))
    self.assertFalse(jnp.any(jnp.isnan(value[1])))


if __name__ == "__main__":
  absltest.main()
