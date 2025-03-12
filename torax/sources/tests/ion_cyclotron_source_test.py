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
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax import math_utils

# Internal import.
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
      json.dump(model_config, f, indent=4, separators=(",", ":"))
    cls.dummy_input = model_input
    cls.dummy_output = model_output
    super().setUpClass(
        source_class=ion_cyclotron_source.IonCyclotronSource,
        runtime_params_class=ion_cyclotron_source.RuntimeParams,
        source_class_builder=ion_cyclotron_source.IonCyclotronSourceBuilder,
        source_name=ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME,
        model_func=None,
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
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder(
        {ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME: source_builder},
    )
    source_models = source_models_builder()
    source = source_models.sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, source_lib.Source)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
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

    # Calculate integrated power
    integrated_ion1 = math_utils.volume_integration(ion_and_el[0], geo)
    integrated_ion2 = math_utils.volume_integration(ion_and_el[1], geo)
    integrated_el1 = math_utils.volume_integration(ion_and_el[0], geo)
    integrated_el2 = math_utils.volume_integration(ion_and_el[1], geo)

  @mock.patch.object(
      ion_cyclotron_source,
      "_get_default_model_path",
      autospec=True,
      return_value=_DUMMY_MODEL_PATH,
  )
  def test_absorption_fraction(self, mock_path):
    """Tests that absorption_fraction correctly affects power calculations."""
    # Create a simple geometry
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    
    # Create minimal core profiles
    core_profiles = initialization.build_simple_profiles(geo)
    
    # Set up the toric neural network with dummy model
    toric_nn = ion_cyclotron_source.ToricNNWrapper(_DUMMY_MODEL_PATH)
    
    # Create identical dynamic params but with different absorption fractions
    dynamic_params1 = ion_cyclotron_source.DynamicRuntimeParams(
        frequency=120e6,
        minority_concentration=3.0,
        Ptot=1.0,
        absorption_fraction=1.0,
        wall_inner=1.24,
        wall_outer=2.43,
    )
    
    dynamic_params2 = ion_cyclotron_source.DynamicRuntimeParams(
        frequency=120e6,
        minority_concentration=3.0,
        Ptot=1.0,
        absorption_fraction=0.5,  # Half absorption fraction
        wall_inner=1.24,
        wall_outer=2.43,
    )
    
    # Create simple runtime params slice
    static_params = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=general_runtime_params.GeneralRuntimeParams(),
        source_runtime_params={},
        torax_mesh=geo.torax_mesh,
    )
    
    # Create dynamic runtime params slices
    dynamic_slice1 = {'ion_cyclotron_source': dynamic_params1}
    dynamic_slice2 = {'ion_cyclotron_source': dynamic_params2}
    
    # Call icrh_model_func directly to get ion and electron heat profiles
    ion1, el1 = ion_cyclotron_source.icrh_model_func(
        static_params,
        dynamic_slice1,
        geo,
        'ion_cyclotron_source',
        core_profiles,
        None,
        toric_nn,
    )
    
    ion2, el2 = ion_cyclotron_source.icrh_model_func(
        static_params, 
        dynamic_slice2,
        geo,
        'ion_cyclotron_source',
        core_profiles,
        None,
        toric_nn,
    )
    
    # Integrate the power profiles
    integrated_ion1 = math_utils.volume_integration(ion1, geo)
    integrated_ion2 = math_utils.volume_integration(ion2, geo)
    integrated_el1 = math_utils.volume_integration(el1, geo)
    integrated_el2 = math_utils.volume_integration(el2, geo)
    
    # Second profile should have half the power of the first
    np.testing.assert_allclose(integrated_ion2 / integrated_ion1, 0.5, rtol=1e-5)
    np.testing.assert_allclose(integrated_el2 / integrated_el1, 0.5, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
