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
"""Tests for ICRH heat source."""

import dataclasses
import json
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
from torax import core_profile_setters
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.sources import ion_cyclotron_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib

# Internal import.


_DUMMY_MODEL_PATH = "/tmp/toricnn.json"


class IonCyclotronSourceTest(test_lib.SourceTestCase):
  """Test cases for the ICRH heat source."""

  @classmethod
  def setUpClass(cls):
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
      json.dump(model_config, f)
    cls.dummy_input = model_input
    cls.dummy_output = model_output
    super().setUpClass(
        source_class=ion_cyclotron_source.IonCyclotronSource,
        runtime_params_class=ion_cyclotron_source.RuntimeParams,
        unsupported_modes=[runtime_params_lib.Mode.FORMULA_BASED],
    )

  @parameterized.product(
      total_power=[1e6, 20e6, 120e6],
      frequency=[119e6, 120e6, 121e6],
      minority_concentration=[1, 2, 3, 4, 5],
  )
  @mock.patch.object(
      ion_cyclotron_source,
      "_get_default_model_path",
      autospec=True,
      return_value=_DUMMY_MODEL_PATH,
  )
  def test_icrh_output_matches_total_power(
      self,
      mock_path,
      total_power: float,
      frequency: float,
      minority_concentration: float,
  ):
    """Test source outputs match the total heating power using dummy model."""
    del mock_path
    source_class_builder = self._source_class_builder()
    source_models_builder = source_models_lib.SourceModelsBuilder({
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME: (
            source_class_builder
        )
    })
    source_models = source_models_builder()
    icrh_source = source_models.sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models_builder.runtime_params[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ].Ptot = total_power
    source_models_builder.runtime_params[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ].frequency = frequency
    source_models_builder.runtime_params[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ].minority_concentration = minority_concentration
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=0.0,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    icrh_output = icrh_source.get_value(
        static_slice,
        static_slice.sources[
            ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
        ],
        dynamic_runtime_params_slice,
        dynamic_runtime_params_slice.sources[
            ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
        ],
        geo,
        core_profiles,
    )
    ion_el_total = np.sum(icrh_output, axis=0)
    # Implicit integration using the trapezoid rule.
    integrated_power = jnp.sum(ion_el_total * geo.vpr * geo.drho_norm)

    np.testing.assert_allclose(
        integrated_power,
        total_power,
    )

  @classmethod
  def test_toric_nn_loads_and_predicts_with_dummy_model(cls):
    """Test that the ToricNNWrapper loads and predicts consistently."""
    # Load the model and verify the prediction are consistent with the output
    # of the dummy network.
    toric_wrapper = ion_cyclotron_source.ToricNNWrapper(path=_DUMMY_MODEL_PATH)
    wrapper_output = toric_wrapper.predict(cls.dummy_input)
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_He3,
        cls.dummy_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_2T,
        cls.dummy_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_e,
        cls.dummy_output,
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
    # pylint: disable=missing-kwoa
    source_builder = self._source_class_builder()  # pytype: disable=missing-parameter
    # pylint: enable=missing-kwoa
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {"foo": source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources["foo"]
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
        runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    ion_and_el = source.get_value(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        dynamic_source_runtime_params=dynamic_runtime_params_slice.sources[
            "foo"
        ],
        static_runtime_params_slice=static_slice,
        static_source_runtime_params=static_slice.sources["foo"],
        geo=geo,
        core_profiles=core_profiles,
    )
    chex.assert_rank(ion_and_el, 2)


if __name__ == "__main__":
  absltest.main()
