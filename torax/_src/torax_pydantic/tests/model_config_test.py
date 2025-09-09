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
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import chex
from torax._src import version
from torax._src.config import config_loader
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


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
    config_pydantic = model_config.ToraxConfig.from_dict(config_dict)

    with self.subTest("original_config_dict_unchanged"):
      chex.assert_trees_all_equal(config_dict, config_dict_copy)
      # And the object ids should be unchanged.
      unique_objects_after = get_unique_objects(config_dict, list())
      self.assertListEqual(unique_objects_before, unique_objects_after)

    with self.subTest("has_unique_submodels"):
      self.assertTrue(config_pydantic._has_unique_submodels)

    self.assertEqual(
        config_pydantic.time_step_calculator.calculator_type.value,
        config_dict["time_step_calculator"]["calculator_type"],
    )
    with self.subTest("pedestal_model_set"):
      self.assertEqual(
          config_pydantic.pedestal.model_name,
          config_dict["pedestal"]["model_name"]
          if "model_name" in config_dict["pedestal"]
          else "no_pedestal",
      )
    with self.subTest("transport_model_set"):
      self.assertEqual(
          config_pydantic.transport.model_name,
          config_dict["transport"]["model_name"]
          if "model_name" in config_dict["transport"]
          else "constant",
      )
    # The full model should always be serializable.
    config_json = config_pydantic.model_dump_json()
    with self.subTest("json_roundtrip"):
      config_pydantic_roundtrip = model_config.ToraxConfig.model_validate_json(
          config_json
      )
      chex.assert_trees_all_equal(config_pydantic, config_pydantic_roundtrip)

    with self.subTest("json_has_torax_version"):
      self.assertEqual(
          json.loads(config_json)["torax_version"], version.TORAX_VERSION
      )

    with self.subTest("geometry_grid_set"):
      mesh = config_pydantic.geometry.build_provider.torax_mesh
      time_varying_arrays = [
          m
          for m in config_pydantic.submodels
          if isinstance(m, torax_pydantic.TimeVaryingArray)
      ]
      assert time_varying_arrays  # Should be non-empty.
      for m in time_varying_arrays:
        chex.assert_trees_all_equal(mesh.face_centers, m.grid.face_centers)
        chex.assert_trees_all_equal(mesh.cell_centers, m.grid.cell_centers)

  def test_geometry_direct_nrho_hires_factor_update(self):

    config_dict = config_loader.import_module(
        "tests/test_data/test_iterhybrid_predictor_corrector.py",
    )["CONFIG"]
    config_pydantic = model_config.ToraxConfig.from_dict(config_dict)

    new_n_rho = config_pydantic.geometry.geometry_configs.config.n_rho * 2  # pytype: disable=attribute-error
    new_hires_factor = (
        config_pydantic.geometry.geometry_configs.config.hires_factor * 2  # pytype: disable=attribute-error
    )

    # Check that the caches are invalidated.
    config_pydantic.plasma_composition.Z_eff.get_value(t=0.2, grid_type="cell")
    config_pydantic.plasma_composition.Z_eff.get_value(t=0.2, grid_type="face")

    config_pydantic.update_fields({
        "geometry.geometry_configs.config.n_rho": new_n_rho,
        "geometry.geometry_configs.config.hires_factor": new_hires_factor,
    })

    self.assertEqual(
        config_pydantic.geometry.geometry_configs.config.n_rho,  # pytype: disable=attribute-error
        new_n_rho,
    )
    self.assertEqual(
        config_pydantic.geometry.geometry_configs.config.hires_factor,  # pytype: disable=attribute-error
        new_hires_factor,
    )

    with self.subTest("nrho_updated_reset_mesh_cache"):
      v1_cell = config_pydantic.plasma_composition.Z_eff.get_value(
          t=0.2, grid_type="cell"
      )
      v1_face = config_pydantic.plasma_composition.Z_eff.get_value(
          t=0.2, grid_type="face"
      )
      self.assertLen(v1_cell, new_n_rho)
      self.assertLen(v1_face, new_n_rho + 1)

  def test_n_rho_updated_from_full_geometry_dict_update(self):

    config_dict = config_loader.import_module(
        "tests/test_data/test_iterhybrid_predictor_corrector.py",
    )["CONFIG"]
    config_pydantic = model_config.ToraxConfig.from_dict(config_dict)

    new_n_rho = config_pydantic.geometry.geometry_configs.config.n_rho * 2  # pytype: disable=attribute-error

    config_pydantic.update_fields({
        "geometry": {
            "n_rho": new_n_rho,
            "geometry_type": "circular",
        },
    })

    self.assertEqual(
        config_pydantic.geometry.geometry_configs.config.n_rho,  # pytype: disable=attribute-error
        new_n_rho,
    )

    with self.subTest("nrho_updated_reset_mesh_cache"):
      v1_cell = config_pydantic.plasma_composition.Z_eff.get_value(
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

    config_dict["transport"] = {"model_name": transport_model}
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

  def _assert_warning_logged(
      self, warning_snippet, expect_warning, config_dict
  ):
    # Avoid assertion failure when no warnings are logged at all.
    try:
      with self.assertLogs(level=logging.WARNING) as cm:
        model_config.ToraxConfig.from_dict(config_dict)
        warnings = "\n".join(cm.output)
    except Exception:  # pylint: disable=broad-except
      warnings = ""
    if expect_warning:
      self.assertIn(warning_snippet, warnings)
    else:
      self.assertNotIn(warning_snippet, warnings)


if __name__ == "__main__":
  absltest.main()
