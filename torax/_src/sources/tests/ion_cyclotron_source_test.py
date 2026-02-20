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
import pydantic
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.physics import fast_ions
from torax._src.sources import ion_cyclotron_source
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources.tests import test_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic

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
        minority_concentration=2.0,  # In percentage.
        gap_inner=0.01,
        gap_outer=0.01,
        z0=0.0,
        temperature_peaking_factor=2,
        density_peaking_factor=2,
        B_0=12.2,
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
        model_input.B_0,
    ])
    model_output, params = toric_nn.init_with_output(
        jax.random.PRNGKey(0), toric_input
    )
    config = dataclasses.asdict(toric_nn)
    config[ion_cyclotron_source._HELIUM3_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    config[ion_cyclotron_source._TRITIUM_SECOND_HARMONIC_ID] = (
        jax.tree_util.tree_map(lambda x: x.tolist(), params["params"])
    )
    config[ion_cyclotron_source._ELECTRON_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    # pylint: enable=protected-access
    with open(_DUMMY_MODEL_PATH, "w") as f:
      json.dump(config, f, indent=4, separators=(",", ":"))
    self.dummy_input = model_input
    self.dummy_output = model_output
    super().setUp(
        source_config_class=ion_cyclotron_source.IonCyclotronSourceConfig,
        source_name=ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME,
    )
    # pytype: enable=signature-mismatch

  def config_raises_if_model_path_does_not_exist(self):
    with self.assertRaises(FileNotFoundError):
      ion_cyclotron_source.IonCyclotronSourceConfig.from_dict(
          {"model_path": "/tmp/non_existent_file.json"}
      )

  def test_build_runtime_params(self):
    source = self._source_config_class.from_dict(
        {"model_path": _DUMMY_MODEL_PATH}
    )
    self.assertIsInstance(source, self._source_config_class)
    face_centers = interpolated_param_2d.get_face_centers(nx=4)
    torax_pydantic.set_grid(
        source, torax_pydantic.Grid1D(face_centers=face_centers)
    )
    runtime_params = source.build_runtime_params(t=0.0)
    self.assertIsInstance(runtime_params, runtime_params_lib.RuntimeParams)

  @parameterized.product(
      mode=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.MODEL_BASED,
          runtime_params_lib.Mode.PRESCRIBED,
      ),
      is_explicit=(True, False),
  )
  def test_runtime_params_builds_dynamic_params(
      self, mode: runtime_params_lib.Mode, is_explicit: bool
  ):
    source_config = self._source_config_class.from_dict({
        "mode": mode,
        "is_explicit": is_explicit,
        "model_path": _DUMMY_MODEL_PATH,
    })
    face_centers = interpolated_param_2d.get_face_centers(nx=4)
    torax_pydantic.set_grid(
        source_config, torax_pydantic.Grid1D(face_centers=face_centers),
    )
    dynamic_params = source_config.build_runtime_params(t=0.0)
    self.assertIsInstance(dynamic_params, runtime_params_lib.RuntimeParams)
    self.assertEqual(dynamic_params.mode, mode)
    self.assertEqual(dynamic_params.is_explicit, is_explicit)

  def test_toric_nn_loads_and_predicts_with_dummy_model(self):
    """Test that the ToricNNWrapper loads and predicts consistently."""
    # Load the model and verify the prediction are consistent with the output
    # of the dummy network.
    toric_wrapper = ion_cyclotron_source.ToricNNWrapper(path=_DUMMY_MODEL_PATH)
    wrapper_output = ion_cyclotron_source._toric_nn_predict(
        toric_wrapper, self.dummy_input
    )
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_He3,
        self.dummy_output,
    )
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_2T,
        self.dummy_output,
    )
    np.testing.assert_array_almost_equal(
        wrapper_output.power_deposition_e,
        self.dummy_output,
    )
    # pylint: enable=protected-access

  def test_source_value(self):
    """Tests that the source can provide a value by default."""
    config = default_configs.get_default_config_dict()
    config["sources"] = {self._source_name: {"model_path": _DUMMY_MODEL_PATH}}
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, source_lib.Source)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)
    self.assertEqual(icrh_out[1].shape, geo.rho.shape)

  def test_source_with_minority_species_from_composition(self):
    """Tests ICRH source with minority_species reading from composition."""
    config = default_configs.get_default_config_dict()
    # Add He3 as an impurity in the plasma composition
    config["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"He3": 0.03},  # 3% He3 minority
        },
    }
    # Configure ICRH to use minority_species instead of minority_concentration
    config["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
            # minority_concentration is ignored when minority_species is set
            "minority_concentration": 0.01,  # This should be ignored
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(source, source_lib.Source)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    # Verify minority_species is set in runtime params
    icrh_params = runtime_params.sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    self.assertIsInstance(icrh_params, ion_cyclotron_source.RuntimeParams)
    self.assertEqual(icrh_params.minority_species, "He3")
    # Verify He3 is in impurity composition
    self.assertIn("He3", runtime_params.plasma_composition.impurity_names)
    # Run the source and check it produces valid output
    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)
    self.assertEqual(icrh_out[1].shape, geo.rho.shape)

  def test_source_with_minority_species_main_ion(self):
    """Tests ICRH source with minority_species as a main ion."""
    config = default_configs.get_default_config_dict()
    # Add He3 as a main ion in the plasma composition
    config["plasma_composition"] = {
        "main_ion": {
            "D": 0.47,
            "T": 0.5,
            "He3": 0.03,  # 3% He3 as main ion
        },
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": 0.0},
        },
    }
    # Configure ICRH to use minority_species
    config["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]

    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    # Run the source calculation
    icrh_out = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )

    # Just verify it runs and returns shapes
    self.assertLen(icrh_out, 3)
    self.assertEqual(icrh_out[0].shape, geo.rho.shape)

  def test_source_raises_if_minority_species_not_in_composition(self):
    """Tests that a ValueError is raised if minority species is missing."""
    config = default_configs.get_default_config_dict()
    # D and T only, no He3
    config["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {"impurity_mode": "n_e_ratios", "species": {"Ne": 0.01}},
    }
    config["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
        }
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'The ToricNN ICRH model requires "He3" to be present',
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_minority_concentration_warning_by_default(self):
    with self.assertLogs(level="WARNING") as cm:
      ion_cyclotron_source.IonCyclotronSourceConfig()
    self.assertTrue(
        any("minority_concentration is provided" in o for o in cm.output)
    )

  def test_minority_concentration_warning_when_explicitly_set_to_non_none(self):
    with self.assertLogs(level="WARNING") as cm:
      ion_cyclotron_source.IonCyclotronSourceConfig(minority_concentration=0.05)
    self.assertTrue(
        any("minority_concentration is provided" in o for o in cm.output)
    )

  def test_minority_concentration_no_warning_when_none(self):
    with self.assertNoLogs(level="WARNING"):
      ion_cyclotron_source.IonCyclotronSourceConfig(minority_concentration=None)

  def test_icrh_returns_fast_ion_data(self):
    config = default_configs.get_default_config_dict()
    config["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 10e6}
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(output, 3)
    fast_ion_data = output[2]
    self.assertIsInstance(fast_ion_data, tuple)
    self.assertLen(fast_ion_data, 1)
    fi = fast_ion_data[0]
    self.assertIsInstance(fi, fast_ions.FastIon)
    self.assertEqual(fi.species, "He3")
    self.assertEqual(
        fi.source, ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    )
    self.assertEqual(fi.n.value.shape, geo.rho.shape)
    self.assertEqual(fi.T.value.shape, geo.rho.shape)

  def test_fast_ion_density_conservation(self):
    config = default_configs.get_default_config_dict()
    minority_conc = 0.03
    config["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "P_total": 10e6,
            "minority_concentration": minority_conc,
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    n_tail = fast_ion_data[0].n.value
    n_total = core_profiles.n_e.value * minority_conc
    self.assertTrue(jnp.all(n_tail >= 0))
    self.assertTrue(jnp.all(n_tail <= n_total))

  def test_fast_ion_tail_temperature_above_electron_temp(self):
    config = default_configs.get_default_config_dict()
    config["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 10e6}
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    temperature_tail = fast_ion_data[0].T.value
    self.assertTrue(jnp.all(temperature_tail >= core_profiles.T_e.value))

  def test_zero_power_produces_zero_fast_ions(self):
    config = default_configs.get_default_config_dict()
    config["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 0.0}
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    source = source_models.standard_sources[
        ion_cyclotron_source.IonCyclotronSource.SOURCE_NAME
    ]
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    output = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output[2]
    np.testing.assert_allclose(fast_ion_data[0].n.value, 0.0, atol=1e-9)
    np.testing.assert_allclose(
        fast_ion_data[0].T.value, core_profiles.T_e.value
    )


if __name__ == "__main__":
  absltest.main()
