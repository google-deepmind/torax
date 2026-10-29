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
import copy
import json
import logging
import os
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
import eqdsk as eqdsk_lib
import numpy as np
import pydantic
from torax._src import version
from torax._src.config import config_loader
from torax._src.geometry import eqdsk
from torax._src.geometry import geometry_loader
from torax._src.geometry import get_example_L_LY_data
from torax._src.test_utils import default_configs
from torax._src.test_utils import paths
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


def get_unique_objects(x: Any, object_ids: list[int]) -> list[int]:
  if isinstance(x, dict):
    for _, value in x.items():
      object_ids = get_unique_objects(value, object_ids)
  elif isinstance(x, (list, tuple)):
    for value in x:
      object_ids = get_unique_objects(value, object_ids)
  object_ids.append(id(x))
  return object_ids


class ConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      "test_implicit",
      "test_bohmgyrobohm_all",
      "test_iterhybrid_predictor_corrector",
      "test_iterhybrid_rampup",
      "test_iterhybrid_rampup_restart",
  )
  def test_full_config_construction(self, config_name):
    """Test for basic config construction."""

    config_dict = config_loader.import_module(
        f"tests/test_data/{config_name}.py",
    )["CONFIG"]
    unique_objects_before = get_unique_objects(config_dict, list())
    config_dict_copy = copy.deepcopy(config_dict)  # Keep a copy for comparison.
    torax_config = config.ToraxConfig.from_dict(config_dict)

    with self.subTest("original_config_dict_unchanged"):
      chex.assert_trees_all_equal(config_dict, config_dict_copy)
      # And the object ids should be unchanged.
      unique_objects_after = get_unique_objects(config_dict, list())
      self.assertListEqual(unique_objects_before, unique_objects_after)

    with self.subTest("has_unique_submodels"):
      self.assertTrue(torax_config._has_unique_submodels)

    self.assertEqual(
        torax_config.time_step_calculator.calculator_type,
        config_dict["time_step_calculator"]["calculator_type"],
    )
    with self.subTest("pedestal_model_set"):
      pedestal_dict = config_dict["pedestal"]
      expected_model_name = pedestal_dict.get("structure_model", {}).get(
          "model_name", "no_pedestal"
      )
      self.assertEqual(
          torax_config.pedestal.structure_model.model_name,
          expected_model_name,
      )
    with self.subTest("transport_model_set"):
      self.assertNotEmpty(
          torax_config.transport.core_transport_models,
      )
    # The full model should always be serializable.
    config_json = torax_config.model_dump_json()
    with self.subTest("json_roundtrip"):
      torax_config_roundtrip = config.ToraxConfig.model_validate_json(
          config_json
      )
      chex.assert_trees_all_equal(torax_config, torax_config_roundtrip)

    with self.subTest("json_has_torax_version"):
      self.assertEqual(
          json.loads(config_json)["torax_version"], version.TORAX_VERSION
      )

    with self.subTest("geometry_grid_set"):
      mesh = torax_config.geometry.build_provider.torax_mesh
      time_varying_arrays = [
          m
          for m in torax_config.submodels
          if isinstance(m, torax_pydantic.TimeVaryingArray)
      ]
      assert time_varying_arrays  # Should be non-empty.
      for m in time_varying_arrays:
        chex.assert_trees_all_equal(mesh.face_centers, m.grid.face_centers)  # pyrefly: ignore[missing-attribute]
        chex.assert_trees_all_equal(mesh.cell_centers, m.grid.cell_centers)  # pyrefly: ignore[missing-attribute]

  def test_geometry_direct_nrho_hires_factor_update(self):

    config_dict = config_loader.import_module(
        "tests/test_data/test_iterhybrid_predictor_corrector.py",
    )["CONFIG"]
    torax_config = config.ToraxConfig.from_dict(config_dict)

    assert not isinstance(torax_config.geometry.geometry_configs, dict)
    assert torax_config.geometry.geometry_configs.n_rho is not None
    new_n_rho = torax_config.geometry.geometry_configs.n_rho * 2
    new_hires_factor = torax_config.geometry.geometry_configs.hires_factor * 2

    # Check that the caches are invalidated.
    torax_config.plasma_composition.Z_eff.get_value(t=0.2, grid_type="cell")
    torax_config.plasma_composition.Z_eff.get_value(t=0.2, grid_type="face")

    torax_config.update_fields({
        "geometry.geometry_configs.n_rho": new_n_rho,
        "geometry.geometry_configs.hires_factor": new_hires_factor,
    })

    self.assertEqual(
        torax_config.geometry.geometry_configs.n_rho,
        new_n_rho,
    )
    self.assertEqual(
        torax_config.geometry.geometry_configs.hires_factor,
        new_hires_factor,
    )

    with self.subTest("nrho_updated_reset_mesh_cache"):
      v1_cell = torax_config.plasma_composition.Z_eff.get_value(
          t=0.2, grid_type="cell"
      )
      v1_face = torax_config.plasma_composition.Z_eff.get_value(
          t=0.2, grid_type="face"
      )
      self.assertLen(v1_cell, new_n_rho)
      self.assertLen(v1_face, new_n_rho + 1)

  def test_n_rho_updated_from_full_geometry_dict_update(self):

    config_dict = config_loader.import_module(
        "tests/test_data/test_iterhybrid_predictor_corrector.py",
    )["CONFIG"]
    torax_config = config.ToraxConfig.from_dict(config_dict)

    assert not isinstance(torax_config.geometry.geometry_configs, dict)
    assert torax_config.geometry.geometry_configs.n_rho is not None
    new_n_rho = torax_config.geometry.geometry_configs.n_rho * 2

    torax_config.update_fields({
        "geometry": {
            "n_rho": new_n_rho,
            "geometry_type": "circular",
        },
        "profile_conditions.initial_psi_source": "j",
    })

    self.assertEqual(
        torax_config.geometry.geometry_configs.n_rho,
        new_n_rho,
    )

    with self.subTest("nrho_updated_reset_mesh_cache"):
      v1_cell = torax_config.plasma_composition.Z_eff.get_value(
          t=0.2, grid_type="cell"
      )
      self.assertLen(v1_cell, new_n_rho)

  @parameterized.named_parameters(
      ("const_lin_no_per", "constant", "linear", None, False, False),
      ("qlknn_lin_no_per", "qlknn", "linear", None, False, True),
      ("cgm_lin_no_per", "CGM", "linear", None, False, True),
      ("qlknn_lin_per", "qlknn", "linear", None, True, False),
      ("qualikiz_lin_per", "qualikiz", "linear", None, True, False),
      (
          "qlknn_newton_no_per_x_old",
          "qlknn",
          "newton_raphson",
          "x_old",
          False,
          False,
      ),
      (
          "qlknn_newton_no_per_linear",
          "qlknn",
          "newton_raphson",
          "linear",
          False,
          True,
      ),
      ("qlknn_newton_per_1", "qlknn", "newton_raphson", "linear", True, False),
  )
  def test_pereverzev_warning(
      self,
      transport_model,
      solver_type,
      initial_guess_mode,
      use_pereverzev,
      expect_warning,
  ):
    # Use a basic config and modify it to test the warning.
    config_dict = default_configs.get_default_config_dict()

    if transport_model == "combined":
      config_dict["transport"] = {}
    else:
      config_dict["transport"] = {
          "core_transport_models": {
              transport_model: {"model_name": transport_model}
          },
      }
    config_dict["solver"] = {
        "solver_type": solver_type,
        "use_pereverzev": use_pereverzev,
    }
    if initial_guess_mode is not None:
      config_dict["solver"]["initial_guess_mode"] = initial_guess_mode

    warning_snippet = "use_pereverzev=False in a configuration where setting"

    self._assert_warning_logged(warning_snippet, expect_warning, config_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name="psidot_none_evolve_current_true",
          psidot=None,
          evolve_current=True,
          expect_warning=False,
      ),
      dict(
          testcase_name="psidot_set_evolve_current_false",
          psidot={0: {0: 0.1, 1: 0.2}},
          evolve_current=False,
          expect_warning=False,
      ),
      dict(
          testcase_name="psidot_set_evolve_current_true",
          psidot={0: {0: 0.1, 1: 0.2}},
          evolve_current=True,
          expect_warning=True,
      ),
  )
  def test_psidot_evolve_current_warning(
      self, psidot, evolve_current, expect_warning
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict["profile_conditions"]["psidot"] = psidot
    config_dict["numerics"]["evolve_current"] = evolve_current

    warning_snippet = "profile_conditions.psidot input is ignored"
    self._assert_warning_logged(warning_snippet, expect_warning, config_dict)

  def test_missing_initial_psi_source_raises_error(self):
    config_dict = default_configs.get_default_config_dict()
    del config_dict["profile_conditions"]["initial_psi_source"]
    with self.assertRaisesRegex(pydantic.ValidationError, "initial_psi_source"):
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name="psi_none_no_restart",
          psi=None,
          do_restart=None,
          expect_error=True,
      ),
      dict(
          testcase_name="psi_none_do_restart_false",
          psi=None,
          do_restart=False,
          expect_error=True,
      ),
      dict(
          testcase_name="psi_none_do_restart_true",
          psi=None,
          do_restart=True,
          expect_error=False,
      ),
      dict(
          testcase_name="psi_provided_no_restart",
          psi={0: {0: 0.0, 1: 1.0}},
          do_restart=None,
          expect_error=False,
      ),
  )
  def test_initial_psi_source_profile_conditions_requires_psi_unless_restarting(
      self, psi, do_restart, expect_error
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict["profile_conditions"][
        "initial_psi_source"
    ] = "profile_conditions"
    config_dict["profile_conditions"]["psi"] = psi
    if do_restart is not None:
      config_dict["restart"] = {
          "filename": os.path.join(
              paths.test_data_dir(), "test_iterhybrid_rampup.nc"
          ),
          "time": 40,
          "do_restart": do_restart,
          "stitch": True,
      }
    if expect_error:
      with self.assertRaisesRegex(
          ValueError,
          "profile_conditions.psi must be provided when"
          " profile_conditions.initial_psi_source is 'profile_conditions'",
      ):
        config.ToraxConfig.from_dict(config_dict)
    else:
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name="circular_geometry_raises",
          geometry={"geometry_type": "circular", "n_rho": 4},
          expect_error=True,
      ),
      dict(
          testcase_name="chease_geometry_passes",
          geometry={
              "geometry_type": "chease",
              "geometry_file": "iterhybrid.mat2cols",
          },
          expect_error=False,
      ),
  )
  def test_initial_psi_source_geometry_validation(self, geometry, expect_error):
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = geometry
    config_dict["profile_conditions"]["initial_psi_source"] = "geometry"
    if expect_error:
      with self.assertRaisesRegex(
          ValueError,
          "profile_conditions.initial_psi_source='geometry' is not supported"
          " for CircularGeometry.",
      ):
        config.ToraxConfig.from_dict(config_dict)
    else:
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name="psi_set_source_j_evolve_current_true",
          initial_psi_source="j",
          geometry_type="circular",
          psi={0: {0: 0.0, 1: 1.0}},
          evolve_current=True,
          expect_warning=True,
      ),
      dict(
          testcase_name="psi_set_source_geometry_evolve_current_true",
          initial_psi_source="geometry",
          geometry_type="chease",
          psi={0: {0: 0.0, 1: 1.0}},
          evolve_current=True,
          expect_warning=True,
      ),
      dict(
          testcase_name="psi_set_source_j_evolve_current_false",
          initial_psi_source="j",
          geometry_type="circular",
          psi={0: {0: 0.0, 1: 1.0}},
          evolve_current=False,
          expect_warning=False,
      ),
      dict(
          testcase_name="psi_set_source_geometry_evolve_current_false",
          initial_psi_source="geometry",
          geometry_type="chease",
          psi={0: {0: 0.0, 1: 1.0}},
          evolve_current=False,
          expect_warning=False,
      ),
      dict(
          testcase_name="psi_set_source_profile_conditions_evolve_current_true",
          initial_psi_source="profile_conditions",
          geometry_type="circular",
          psi={0: {0: 0.0, 1: 1.0}},
          evolve_current=True,
          expect_warning=False,
      ),
      dict(
          testcase_name="psi_none_source_j_evolve_current_true",
          initial_psi_source="j",
          geometry_type="circular",
          psi=None,
          evolve_current=True,
          expect_warning=False,
      ),
  )
  def test_psi_ignored_warning(
      self,
      initial_psi_source,
      geometry_type,
      psi,
      evolve_current,
      expect_warning,
  ):
    config_dict = default_configs.get_default_config_dict()
    if geometry_type == "chease":
      config_dict["geometry"] = {
          "geometry_type": "chease",
          "geometry_file": "iterhybrid.mat2cols",
      }
    config_dict["profile_conditions"]["initial_psi_source"] = initial_psi_source
    config_dict["profile_conditions"]["psi"] = psi
    config_dict["numerics"]["evolve_current"] = evolve_current

    warning_snippet = "profile_conditions.psi is provided but will be ignored"
    self._assert_warning_logged(warning_snippet, expect_warning, config_dict)

  @parameterized.named_parameters(
      ("uniform_no_ped", True, False, False),
      ("non_uniform_no_ped", False, False, False),
      ("uniform_ped", True, True, False),
      ("non_uniform_ped", False, True, True),
  )
  def test_pedestal_non_uniform_grid_warning(
      self, is_uniform, use_pedestal, expect_warning
  ):
    config_dict = default_configs.get_default_config_dict()
    if not is_uniform:
      # non-uniform grid
      config_dict["geometry"]["face_centers"] = np.array(
          [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
      )
    if use_pedestal:
      config_dict["pedestal"] = {
          "structure_model": {"model_name": "set_T_ped_n_ped"}
      }
    else:
      config_dict["pedestal"] = {
          "structure_model": {"model_name": "no_pedestal"}
      }

    warning_snippet = "Config has both a pedestal model and a non-uniform grid."
    self._assert_warning_logged(warning_snippet, expect_warning, config_dict)

  def _assert_warning_logged(
      self, warning_snippet, expect_warning, config_dict
  ):
    # Avoid assertion failure when no warnings are logged at all.
    try:
      with self.assertLogs(level=logging.WARNING) as cm:
        config.ToraxConfig.from_dict(config_dict)
        warnings = "\n".join(cm.output)
    except Exception:  # pylint: disable=broad-except
      warnings = ""
    if expect_warning:
      self.assertIn(warning_snippet, warnings)
    else:
      self.assertNotIn(warning_snippet, warnings)

  def test_edge_model_with_circular_geometry_raises_error(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "circular",
        "n_rho": 33,
    }
    config_dict["edge"] = {
        "model_name": "extended_lengyel",
    }
    with self.assertRaisesRegex(
        ValueError,
        "Edge models are not supported for use with CircularGeometry.",
    ):
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      ("good_config", None, False, ""),
      (
          "bad_config",
          True,
          True,
          "`diverted` must NOT be set when using FBT geometry",
      ),
  )
  def test_edge_diverted_status_validation_with_fbt(
      self, diverted, expect_error, error_regex
  ):
    L, LY = get_example_L_LY_data.get_example_L_LY_data(10, 1)
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "fbt",
        "L_object": L,
        "LY_object": LY,
    }
    config_dict["edge"] = {
        "computation_mode": "forward",
        "model_name": "extended_lengyel",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "diverted": diverted,
    }
    # Need valid plasma composition for extended lengyel
    config_dict["plasma_composition"]["impurity"] = {
        "impurity_mode": "n_e_ratios",
        "species": {"Ne": 0.01},
    }
    if expect_error:
      with self.assertRaisesRegex(ValueError, error_regex):
        config.ToraxConfig.from_dict(config_dict)
    else:
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      ("good_config", True, False, ""),
      (
          "bad_config",
          None,
          True,
          "`diverted` MUST be set when using non-FBT geometry",
      ),
  )
  def test_edge_diverted_status_validation_with_chease(
      self, diverted, expect_error, error_regex
  ):
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "forward",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "diverted": diverted,
    }
    # Need valid plasma composition for extended lengyel
    config_dict["plasma_composition"]["impurity"] = {
        "impurity_mode": "n_e_ratios",
        "species": {"Ne": 0.01},
    }
    if expect_error:
      with self.assertRaisesRegex(ValueError, error_regex):
        config.ToraxConfig.from_dict(config_dict)
    else:
      config.ToraxConfig.from_dict(config_dict)

  def test_edge_core_impurity_consistency_overlap(self):
    """Tests that overlapping fixed and seeded impurities raise an error."""
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    # Inverse mode logic checks that fixed and seed sets are disjoint.
    config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        # 'N' is present in both fixed and seed dictionaries.
        "fixed_impurity_concentrations": {"N": 0.01},
        "seed_impurity_weights": {"N": 1.0},
        "enrichment_factor": {"N": 1.0},
        "diverted": True,
    }
    config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"N": 0.01},
        }
    }

    with self.assertRaisesRegex(
        ValueError, "Edge fixed and seeded impurities must be disjoint"
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_edge_core_impurity_consistency_mismatch(self):
    """Tests that mismatched core and edge species raise an error."""
    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        # Edge has 'N' and 'Ar'
        "fixed_impurity_concentrations": {"N": 0.01},
        "seed_impurity_weights": {"Ar": 1.0},
        "enrichment_factor": {"N": 1.0, "Ar": 1.0},
        "diverted": True,
    }

    # Core has 'N', 'Ar', and 'Ne'. 'Ne' is missing from the edge config.
    config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"N": 0.01, "Ar": 0.01, "Ne": 0.01},
        }
    }

    with self.assertRaisesRegex(
        ValueError, "Mismatch between core plasma composition impurities"
    ):
      config.ToraxConfig.from_dict(config_dict)

  @parameterized.named_parameters(
      (
          "toric_nn",
          {"model_name": "toric_nn", "model_path": "/tmp/dummy_path.json"},
          "He3",
      ),
      (
          "scaled_profile",
          {"model_name": "scaled_profile"},
          "Li",
      ),
  )
  def test_validate_icrh_requires_minority_species_in_composition(
      self, icrh_kwargs, minority_species
  ):
    """Tests that ICRH models require minority_species in plasma composition."""
    config_dict = default_configs.get_default_config_dict()
    config_dict["sources"] = {
        "icrh": {
            **icrh_kwargs,
            "minority_species": minority_species,
        }
    }

    # Case 1: minority_species is missing -> INVALID
    config_dict["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": 0.01},
        },
    }
    with self.assertRaisesRegex(
        ValueError,
        f'The ICRH source requires "{minority_species}" to be present',
    ):
      config.ToraxConfig.from_dict(config_dict)

    # Case 1b: minority_species is missing in ZERO or PRESCRIBED mode -> Valid
    for mode in ("zero", "prescribed"):
      non_model_config_dict = copy.deepcopy(config_dict)
      non_model_config_dict["sources"]["icrh"]["mode"] = mode
      # Should not raise
      config.ToraxConfig.from_dict(non_model_config_dict)

    # Case 2: minority_species is present as main ion -> Valid
    config_dict["plasma_composition"] = {
        "main_ion": {"D": 0.47, "T": 0.5, minority_species: 0.03},
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": 0.01},
        },
    }
    config.ToraxConfig.from_dict(config_dict)  # Should not raise

    # Case 3: minority_species is present as impurity -> Valid
    config_dict["plasma_composition"] = {
        "main_ion": {"D": 0.5, "T": 0.5},
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {minority_species: 0.03},
        },
    }
    config.ToraxConfig.from_dict(config_dict)  # Should not raise

  def test_torax_config_eqdsk_roundtrip(self):
    geo_dir = geometry_loader.get_geometry_dir()
    file_name = "iterhybrid_cocos11.eqdsk"
    file_path = f"{geo_dir}/{file_name}"
    eqdsk_obj = eqdsk_lib.EQDSKInterface.from_file(file_path, from_cocos=11)

    config_dict = default_configs.get_default_config_dict()
    config_dict["geometry"] = {
        "geometry_type": "eqdsk",
        "cocos": 11,
        "eqdsk_object": eqdsk_obj,
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    dumped_config_dict = torax_config.model_dump()

    torax_config_restored = config.ToraxConfig.from_dict(dumped_config_dict)
    self.assertIsInstance(
        torax_config_restored.geometry.geometry_configs, eqdsk.EQDSKConfig
    )
    assert isinstance(
        torax_config_restored.geometry.geometry_configs, eqdsk.EQDSKConfig
    )
    assert (
        torax_config_restored.geometry.geometry_configs.eqdsk_object is not None
    )
    self.assertIsInstance(
        torax_config_restored.geometry.geometry_configs.eqdsk_object,
        eqdsk_lib.EQDSKInterface,
    )
    provider = torax_config_restored.geometry.build_provider
    self.assertIsNotNone(provider(0.0))


class ExtendedLengyelImpurityModeValidationTest(parameterized.TestCase):
  """Tests validation logic for Extended Lengyel and impurity mode."""

  def setUp(self):
    super().setUp()
    self.config_dict = default_configs.get_default_config_dict()
    self.config_dict["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }

  def test_inverse_mode_bad_impurity_mode(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 5.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Default plasma_composition uses 'fractions', which is incompatible
    # with extended lengyel.
    with self.assertRaisesRegex(
        ValueError,
        "Edge models require"
        " plasma_composition.impurity_mode to be 'n_e_ratios'",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_inverse_mode_good_impurity_mode(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 5.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": 0.01},
        }
    }
    # Should not raise
    config.ToraxConfig.from_dict(self.config_dict)

  def test_forward_mode_bad_impurity_mode(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "forward",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Default plasma_composition uses 'fractions', which is incompatible
    # with extended lengyel.
    with self.assertRaisesRegex(
        ValueError,
        "Edge models require"
        " plasma_composition.impurity_mode to be 'n_e_ratios'",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_forward_mode_good_impurity_mode(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "forward",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": 0.01},
        }
    }
    # Should not raise
    config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_lcfs_raises_error_for_seeded_impurity(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Profile that goes to zero at rho_norm=1.0.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.0}}},
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "has a zero or negative n_e_ratio at rho_norm=1.0",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_positive_n_e_ratio_at_lcfs_passes_for_seeded_impurity(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Profile with a small but positive value at rho_norm=1.0.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.001}}},
        }
    }
    # Should not raise.
    config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_lcfs_passes_for_fixed_impurity_with_core_sot(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "forward",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Profile that goes to zero at rho_norm=1.0.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.0}}},
        }
    }
    # Should not raise because Ne is a fixed impurity in forward mode,
    # and SoT is core.
    config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_lcfs_raises_error_for_fixed_impurity_with_edge_sot(
      self,
  ):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "forward",
        "impurity_sot": "edge",
        "fixed_impurity_concentrations": {"Ne": 0.01},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Profile that goes to zero at rho_norm=1.0.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.0}}},
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "has a zero or negative n_e_ratio at rho_norm=1.0",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_extrapolated_lcfs_raises_error(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "diverted": True,
    }
    # Profile where the rightmost anchor is at 0.8 and is zero.
    # Because of constant extrapolation, this means the value at 1.0 is also
    # zero.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.8: 0.0}}},
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "has a zero or negative n_e_ratio at rho_norm=1.0",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_lcfs_passes_when_update_n_impurities_false(self):
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "update_n_impurities": False,
        "diverted": True,
    }
    # Profile that goes to zero at rho_norm=1.0.
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.0}}},
        }
    }
    # Should not raise because update_n_impurities is False, so the edge
    # model won't rescale impurity profiles.
    config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_lcfs_passes_when_update_n_impurities_false_at_that_time(self):
    # update_n_impurities: True at t=0, False at t=5 (step interpolation).
    # Species profile: zero at LCFS only at t=5 (where update is off).
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "update_n_impurities": {0: True, 5: False},
        "diverted": True,
    }
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {
                "Ne": {
                    0: {0: 0.01, 1.0: 0.001},
                    2: {0: 0.01, 1.0: 0.001},
                    5: {0: 0.01, 1.0: 0.0},
                    8: {0: 0.01, 1.0: 0.0},
                },
            },
        }
    }
    # Should not raise.
    config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_lcfs_raises_when_update_n_impurities_true_at_that_time(self):
    # update_n_impurities: False at t=0, True at t=5 (step interpolation).
    # Species profile: zero at LCFS at t=5 (where update is on).
    self.config_dict["edge"] = {
        "model_name": "extended_lengyel",
        "computation_mode": "inverse",
        "T_e_target": 10.0,
        "seed_impurity_weights": {"Ne": 1.0},
        "enrichment_factor": {"Ne": 1.0},
        "update_n_impurities": {0: False, 5: True},
        "diverted": True,
    }
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {
                "Ne": {
                    0: {0: 0.01, 1.0: 0.0},
                    5: {0: 0.01, 1.0: 0.0},  # (BAD, update on)
                },
            },
        }
    }
    with self.assertRaisesRegex(
        ValueError,
        "has a zero or negative n_e_ratio at rho_norm=1.0",
    ):
      config.ToraxConfig.from_dict(self.config_dict)

  def test_zero_n_e_ratio_at_lcfs_passes_when_no_edge_model(self):
    self.config_dict["plasma_composition"] = {
        "impurity": {
            "impurity_mode": "n_e_ratios",
            "species": {"Ne": {0: {0: 0.01, 0.5: 0.005, 1.0: 0.0}}},
        }
    }
    # Should not raise because no edge model is active.
    config.ToraxConfig.from_dict(self.config_dict)

  def test_adaptive_transport_with_internal_boundary_conditions_raises(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict["pedestal"] = {
        "structure_model": {"model_name": "set_T_ped_n_ped"},
        "mode": "adaptive_transport",
        "formation_model": {"model_name": "martin_scaling"},
    }
    config_dict["profile_conditions"]["internal_boundary_conditions"] = {
        "model_name": "prescribed",
        "T_e": {0.0: {(0.8, 1.0): 10.0}},
    }
    with self.assertRaisesRegex(
        ValueError,
        "Internal boundary conditions cannot be configured when pedestal mode"
        " is ADAPTIVE_TRANSPORT",
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_adaptive_transport_without_internal_boundary_conditions_passes(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict["pedestal"] = {
        "structure_model": {"model_name": "set_T_ped_n_ped"},
        "mode": "adaptive_transport",
        "formation_model": {"model_name": "martin_scaling"},
    }
    # Should not raise when internal_boundary_conditions is default/empty
    config.ToraxConfig.from_dict(config_dict)


if __name__ == "__main__":
  absltest.main()
