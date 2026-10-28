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
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import pydantic
from torax._src.config import build_runtime_params
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.sources.ion_cyclotron_source import toric_nn as ion_cyclotron_source
from torax._src.sources.tests import test_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic

# Internal import.
# Internal import.

_DUMMY_MODEL_PATH = "/tmp/toricnn.json"


class ToricNNTest(test_lib.SourceTestCase):
  """Test cases for the ToricNN ICRH model."""

  source_name = icrh_base.IonCyclotronSource.SOURCE_ID
  source_config_class = ion_cyclotron_source.ToricNNIonCyclotronSourceConfig

  def setUp(self):
    super().setUp()
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
    config_dict = dataclasses.asdict(toric_nn)
    config_dict[ion_cyclotron_source._HELIUM3_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    config_dict[ion_cyclotron_source._TRITIUM_SECOND_HARMONIC_ID] = (
        jax.tree_util.tree_map(lambda x: x.tolist(), params["params"])
    )
    config_dict[ion_cyclotron_source._ELECTRON_ID] = jax.tree_util.tree_map(
        lambda x: x.tolist(), params["params"]
    )
    # pylint: enable=protected-access
    with open(_DUMMY_MODEL_PATH, "w") as f:
      json.dump(config_dict, f, indent=4, separators=(",", ":"))
    self.dummy_input = model_input
    self.dummy_output = model_output
    self._source_name = icrh_base.IonCyclotronSource.SOURCE_ID

  def _get_default_config_dict(
      self, minority_concentration: float = 0.03
  ) -> dict[str, Any]:
    config_dict = default_configs.get_default_config_dict()
    config_dict["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"He3": minority_concentration},
        },
    }
    return config_dict

  def test_build_runtime_params(self):
    source = ion_cyclotron_source.ToricNNIonCyclotronSourceConfig.from_dict(
        {"model_path": _DUMMY_MODEL_PATH}
    )
    self.assertIsInstance(
        source, ion_cyclotron_source.ToricNNIonCyclotronSourceConfig
    )
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
    source_config = (
        ion_cyclotron_source.ToricNNIonCyclotronSourceConfig.from_dict({
            "mode": mode,
            "is_explicit": is_explicit,
            "model_path": _DUMMY_MODEL_PATH,
        })
    )
    face_centers = interpolated_param_2d.get_face_centers(nx=4)
    torax_pydantic.set_grid(
        source_config,
        torax_pydantic.Grid1D(face_centers=face_centers),
    )
    dynamic_params = source_config.build_runtime_params(t=0.0)
    self.assertIsInstance(dynamic_params, runtime_params_lib.RuntimeParams)
    self.assertEqual(dynamic_params.mode, mode)
    self.assertEqual(dynamic_params.is_explicit, is_explicit)

  def test_config_raises_if_model_path_does_not_exist(self):
    source_config = ion_cyclotron_source.ToricNNIonCyclotronSourceConfig(
        model_path="/tmp/non_existent_file.json"
    )
    # We catch generic Exception because different environments may raise
    # different errors.
    with self.assertRaises(Exception):
      _ = source_config.build_source()

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
    config_dict = self._get_default_config_dict()
    config_dict["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH}
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    self.assertIsInstance(source, source_lib.Source)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out.profiles, 2)
    self.assertEqual(icrh_out.profiles[0].shape, geo.rho.shape)
    self.assertEqual(icrh_out.profiles[1].shape, geo.rho.shape)

  def test_source_with_minority_species_from_composition(self):
    """Tests ICRH source with minority_species reading from composition."""
    config_dict = self._get_default_config_dict()
    config_dict["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    self.assertIsInstance(source, source_lib.Source)
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    # Verify minority_species is set in runtime params
    icrh_params = runtime_params.sources[icrh_base.IonCyclotronSource.SOURCE_ID]
    self.assertIsInstance(icrh_params, ion_cyclotron_source.RuntimeParams)
    self.assertEqual(icrh_params.minority_species, "He3")
    # Verify He3 is in impurity composition
    self.assertIn("He3", runtime_params.plasma_composition.impurity_names)
    # Run the source and check it produces valid output
    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(icrh_out.profiles, 2)
    self.assertEqual(icrh_out.profiles[0].shape, geo.rho.shape)
    self.assertEqual(icrh_out.profiles[1].shape, geo.rho.shape)

  def test_source_with_minority_species_main_ion(self):
    """Tests ICRH source with minority_species as a main ion."""
    config_dict = default_configs.get_default_config_dict()
    # Add He3 as a main ion in the plasma composition
    config_dict["plasma_composition"] = {
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
    config_dict["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]

    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )

    # Run the source calculation
    icrh_out, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )

    # Just verify it runs and returns shapes
    self.assertLen(icrh_out.profiles, 2)
    self.assertEqual(icrh_out.profiles[0].shape, geo.rho.shape)

  def test_source_raises_if_minority_species_not_in_composition(self):
    """Tests that a ValueError is raised if minority species is missing."""
    config_dict = default_configs.get_default_config_dict()
    # D and T only, no He3
    config_dict["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {"impurity_mode": "n_e_ratios", "species": {"Ne": 0.01}},
    }
    config_dict["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "minority_species": "He3",
        }
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        'The ICRH source requires "He3" to be present',
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_invalid_minority_species_raises(self):
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        "Input should be 'He3'",
    ):
      ion_cyclotron_source.ToricNNIonCyclotronSourceConfig(
          minority_species="D"  # pyrefly: ignore[bad-argument-type]
      )

  def test_icrh_returns_fast_ion_data(self):
    config_dict = self._get_default_config_dict()
    config_dict["numerics"] = {"enable_fast_ions": True}
    config_dict["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 10e6}
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    self.assertLen(output.profiles, 2)
    fast_ion_data = output.fast_ions
    self.assertIsInstance(fast_ion_data, tuple)
    self.assertLen(fast_ion_data, 1)
    fi = fast_ion_data[0]
    self.assertIsInstance(fi, fast_ion_lib.FastIon)
    self.assertEqual(fi.species, "He3")
    self.assertEqual(fi.source, icrh_base.IonCyclotronSource.SOURCE_ID)
    self.assertEqual(fi.n.value.shape, geo.rho.shape)
    self.assertEqual(fi.T.value.shape, geo.rho.shape)

  def test_fast_ion_density_conservation(self):
    minority_conc = 0.03
    config_dict = self._get_default_config_dict(
        minority_concentration=minority_conc
    )
    config_dict["numerics"] = {"enable_fast_ions": True}
    config_dict["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 10e6}
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output.fast_ions
    n_tail = fast_ion_data[0].n.value
    n_total = core_profiles.n_e.value * minority_conc
    self.assertTrue(jnp.all(n_tail >= 0))
    self.assertTrue(jnp.all(n_tail <= n_total))

  def test_fast_ion_tail_temperature_above_electron_temp(self):
    config_dict = self._get_default_config_dict()
    config_dict["numerics"] = {"enable_fast_ions": True}
    config_dict["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 10e6}
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output.fast_ions
    temperature_tail = fast_ion_data[0].T.value
    self.assertTrue(jnp.all(temperature_tail >= core_profiles.T_e.value))

  def test_zero_power_produces_zero_fast_ions(self):
    config_dict = self._get_default_config_dict()
    config_dict["numerics"] = {"enable_fast_ions": True}
    config_dict["sources"] = {
        self._source_name: {"model_path": _DUMMY_MODEL_PATH, "P_total": 0.0}
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    core_profiles, runtime_params, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output, _ = source.get_value(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_state=source.initial_state(runtime_params),
        calculated_source_profiles=None,
        conductivity=None,
    )
    fast_ion_data = output.fast_ions
    np.testing.assert_allclose(fast_ion_data[0].n.value, 0.0, atol=1e-9)
    np.testing.assert_allclose(
        fast_ion_data[0].T.value, core_profiles.T_i.value
    )

  @parameterized.parameters(
      (20e6, 0.5),
      (40e6, 0.25),
  )
  def test_absorption_fraction_scales_heating_and_fast_ions(
      self, p_total: float, absorption_fraction: float
  ):
    config_dict = self._get_default_config_dict()
    config_dict["numerics"] = {"enable_fast_ions": True}
    config_dict["sources"] = {
        self._source_name: {
            "model_path": _DUMMY_MODEL_PATH,
            "P_total": p_total,
            "absorption_fraction": absorption_fraction,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    source_models = torax_config.sources.build_models()
    source = source_models.standard_sources[
        icrh_base.IonCyclotronSource.SOURCE_ID
    ]
    core_profiles, runtime_params_partial, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    output_partial, _ = source.get_value(
        runtime_params=runtime_params_partial,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source.initial_state(runtime_params_partial),
    )
    source_ion_partial, source_el_partial = output_partial.profiles
    fast_ions_partial = output_partial.fast_ions

    torax_config.update_fields({
        f"sources.{self._source_name}.P_total": p_total * absorption_fraction,
        f"sources.{self._source_name}.absorption_fraction": 1.0,
    })
    runtime_params_full = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)(
            t=torax_config.numerics.t_initial
        )
    )
    output_full, _ = source.get_value(
        runtime_params=runtime_params_full,
        geo=geo,
        core_profiles=core_profiles,
        calculated_source_profiles=None,
        conductivity=None,
        source_state=source.initial_state(runtime_params_full),
    )
    source_ion_full, source_el_full = output_full.profiles
    fast_ions_full = output_full.fast_ions

    np.testing.assert_allclose(source_ion_partial, source_ion_full)
    np.testing.assert_allclose(source_el_partial, source_el_full)
    np.testing.assert_allclose(
        fast_ions_partial[0].n.value,
        fast_ions_full[0].n.value,
    )
    np.testing.assert_allclose(
        fast_ions_partial[0].T.value,
        fast_ions_full[0].T.value,
    )


if __name__ == "__main__":
  absltest.main()
