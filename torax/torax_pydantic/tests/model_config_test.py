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
from absl.testing import parameterized
import chex
from torax.config import config_loader
from torax.torax_pydantic import model_config
from torax.torax_pydantic import torax_pydantic


class ConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      "test_implicit",
      "test_bohmgyrobohm_all",
      "test_iterhybrid_predictor_corrector",
      "test_iterhybrid_rampup",
  )
  def test_full_config_construction(self, config_name):
    """Test for basic config construction."""

    config_dict = config_loader.import_module(
        f".tests.test_data.{config_name}",
        config_package="torax",
    ).CONFIG
    config_pydantic = model_config.ToraxConfig.from_dict(config_dict)

    self.assertEqual(
        config_pydantic.time_step_calculator.calculator_type.value,
        config_dict["time_step_calculator"]["calculator_type"],
    )
    self.assertEqual(
        config_pydantic.pedestal.pedestal_config.pedestal_model,
        config_dict["pedestal"]["pedestal_model"]
        if "pedestal_model" in config_dict["pedestal"]
        else "set_tped_nped",
    )
    # The full model should always be serializable.
    with self.subTest("json_serialization"):
      config_json = config_pydantic.model_dump_json()
      config_pydantic_roundtrip = model_config.ToraxConfig.model_validate_json(
          config_json
      )
      chex.assert_trees_all_equal(config_pydantic, config_pydantic_roundtrip)

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

  def test_config_safe_update(self):

    config_dict = config_loader.import_module(
        ".tests.test_data.test_iterhybrid_newton",
        config_package="torax",
    ).CONFIG
    config_pydantic = model_config.ToraxConfig.from_dict(config_dict)

    new_n_rho = config_pydantic.geometry.geometry_configs.config.n_rho * 2  # pytype: disable=attribute-error
    new_hires_fac = (
        config_pydantic.geometry.geometry_configs.config.hires_fac * 2  # pytype: disable=attribute-error
    )

    # Check that the caches are invalidated.
    config_pydantic.runtime_params.plasma_composition.Zeff.get_value(
        t=0.2, grid_type="cell"
    )
    config_pydantic.runtime_params.plasma_composition.Zeff.get_value(
        t=0.2, grid_type="face"
    )

    config_pydantic.update_fields({
        "geometry.geometry_configs.config.n_rho": new_n_rho,
        "geometry.geometry_configs.config.hires_fac": new_hires_fac,
    })

    self.assertEqual(
        config_pydantic.geometry.geometry_configs.config.n_rho,  # pytype: disable=attribute-error
        new_n_rho,
    )
    self.assertEqual(
        config_pydantic.geometry.geometry_configs.config.hires_fac,  # pytype: disable=attribute-error
        new_hires_fac,
    )

    with self.subTest("nrho_updated_reset_mesh_cache"):
      v1_cell = (
          config_pydantic.runtime_params.plasma_composition.Zeff.get_value(
              t=0.2, grid_type="cell"
          )
      )
      v1_face = (
          config_pydantic.runtime_params.plasma_composition.Zeff.get_value(
              t=0.2, grid_type="face"
          )
      )
      self.assertLen(v1_cell, new_n_rho)
      self.assertLen(v1_face, new_n_rho + 1)


if __name__ == "__main__":
  absltest.main()
