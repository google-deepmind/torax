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

from absl.testing import absltest
import jax
from jax import numpy as jnp
import numpy as np
from torax.sources import icrh_heat_source

# Internal import.


class IcrhHeatSourceTest(absltest.TestCase):
  """Test cases for the ICRH heat source."""

  def test_toric_nn_loads_and_predicts_with_dummy_model(self):
    """Test that the ToricNNWrapper loads and predicts consistently."""
    # pylint: disable=protected-access
    # Construct a dummy network and save parameters and model config to JSON.
    toric_nn = icrh_heat_source._ToricNN(
        hidden_sizes=[3],
        pca_coeffs=4,
        input_dim=10,
        radial_nodes=2,
    )
    model_input = icrh_heat_source.ToricNNInputs(
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
    model_config[icrh_heat_source._HELIUM3_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    model_config[icrh_heat_source._TRITIUM_SECOND_HARMONIC_ID] = (
        jax.tree_util.tree_map(lambda x: x.tolist(), params["params"])
    )
    model_config[icrh_heat_source._ELECTRON_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    with open("/tmp/toricnn.json", "w") as f:
      json.dump(model_config, f)

    # Load the model and verify the prediction are consistent with the output
    # of the dummy network.
    toric_wrapper = icrh_heat_source.ToricNNWrapper(path="/tmp/toricnn.json")
    wrapper_output = toric_wrapper.predict(model_input)
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_He3,
        model_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_2T,
        model_output,
    )
    np.testing.assert_array_equal(
        wrapper_output.power_deposition_e,
        model_output,
    )
    # pylint: enable=protected-access


if __name__ == "__main__":
  absltest.main()
