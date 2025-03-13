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
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import ion_cyclotron_source
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib

# Internal import.
# Internal import.


_DUMMY_MODEL_PATH = "/tmp/toricnn.json"


class IonCyclotronSourceTest(test_lib.SourceTestCase):
  """Test cases for the ICRH heat source."""

  # pytype: disable=signature-mismatch
  def setUp(self):
    # pylint: disable=protected-access
    # Construct a dummy network and save parameters and model config to JSON.
    toric_nn = ion_cyclotron_source._ToricNN(
        hidden_sizes=[3],
        pca_coeffs=4,
        input_dim=10,
        radial_nodes=ion_cyclotron_source._TORIC_GRID_SIZE,
    )
    model_input = ion_cyclotron_source.ToricNNInputs(
        frequency=120.0e6,
        volume_average_temperature=6,
        volume_average_density=5,
        minority_concentration=2.0,
        gap_inner=0.01,
        gap_outer=0.01,
        z0=0.0,
        temperature_peaking_factor=2,
        density_peaking_factor=2,
        B0=12.2,
    )
    toric_input = jnp.array([
        model_input.frequency,
        model_input.volume_average_temperature,
        model_input.volume_average_density,
        model_input.minority_concentration,
        model_input.gap_inner,
        model_input.gap_outer,
        model_input.z0,
        model_input.temperature_peaking_factor,
        model_input.density_peaking_factor,
        model_input.B0,
    ])
    model_output, params = toric_nn.init_with_output(
        jax.random.PRNGKey(0), toric_input
    )
    model_config = dataclasses.asdict(toric_nn)
    model_config[ion_cyclotron_source._HELIUM3_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    model_config[ion_cyclotron_source._TRITIUM_SECOND_HARMONIC_ID] = (
        jax.tree_util.tree_map(lambda x: x.tolist(), params["params"])
    )
    model_config[ion_cyclotron_source._ELECTRON_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    # pylint: enable=protected-access
    with open(_DUMMY_MODEL_PATH, "w") as f:
      json.dump(model_config, f, indent=4, separators=(",", ":"))
    self.dummy_input = model_input
    self.dummy_output = model_output
    super().setUp(
        source_config_class=ion_cyclotron_source.IonCyclotronSourceConfig,
        source_name=ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME,
    )
    # pytype: enable=signature-mismatch

  def test_toric_nn_loads_and_predicts_with_dummy_model(self):
    """Test that the ToricNNWrapper loads and predicts consistently."""
    # Load the model and verify the prediction are consistent with the output
    # of the dummy network.
    toric_wrapper = ion_cyclotron_source.ToricNNWrapper(path=_DUMMY_MODEL_PATH)
    wrapper_output = toric_wrapper.predict(self.dummy_input)
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_He3,
        self.dummy_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_2T,
        self.dummy_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_e,
        self.dummy_output,
    )
    # pylint: enable=protected-access

  @mock.patch.object(
      ion_cyclotron_source,
      "_get_default_model_path",
      autospec=True,
      return_value=_DUMMY_MODEL_PATH,
  )
  def test_source_value(self, mock_path):
    """Tests that the source can provide a value by default."""
    del mock_path
    sources = sources_pydantic_model.Sources.from_dict({self._source_name: {}})
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    source = source_models.sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, source_lib.Source)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
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


if __name__ == "__main__":
  absltest.main()
