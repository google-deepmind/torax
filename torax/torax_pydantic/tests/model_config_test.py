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


class ConfigTest(parameterized.TestCase):

  @parameterized.parameters(
      "test_crank_nicolson",
      "test_implicit",
      "test_qei",
      "test_pedestal",
      "test_cgmheat",
      "test_bohmgyrobohm_all",
      "test_semiimplicit_convection",
      "test_qlknnheat",
      "test_fixed_dt",
      "test_psiequation",
      "test_psi_and_heat",
      "test_absolute_generic_current_source",
      "test_newton_raphson_zeroiter",
      "test_bootstrap",
      "test_psi_heat_dens",
      "test_particle_sources_constant",
      "test_particle_sources_cgm",
      "test_prescribed_generic_current_source",
      "test_fusion_power",
      "test_all_transport_fusion_qlknn",
      "test_chease",
      "test_eqdsk",
      "test_ohmic_power",
      "test_bremsstrahlung",
      "test_bremsstrahlung_time_dependent_Zimp",
      "test_qei_chease_highdens",
      "test_psichease_ip_parameters",
      "test_psichease_ip_chease",
      "test_psichease_prescribed_jtot",
      "test_psichease_prescribed_johm",
      "test_timedependence",
      "test_prescribed_timedependent_ne",
      "test_ne_qlknn_defromchie",
      "test_ne_qlknn_deff_veff",
      "test_all_transport_crank_nicolson",
      "test_pc_method_ne",
      "test_iterbaseline_mockup",
      "test_iterhybrid_mockup",
      "test_iterhybrid_predictor_corrector",
      "test_iterhybrid_predictor_corrector_eqdsk",
      "test_iterhybrid_predictor_corrector_clip_inputs",
      "test_iterhybrid_predictor_corrector_zeffprofile",
      "test_iterhybrid_predictor_corrector_zi2",
      "test_iterhybrid_predictor_corrector_timedependent_isotopes",
      "test_iterhybrid_predictor_corrector_tungsten",
      "test_iterhybrid_predictor_corrector_ec_linliu",
      "test_iterhybrid_predictor_corrector_constant_fraction_impurity_radiation",
      "test_iterhybrid_predictor_corrector_set_pped_tpedratio_nped",
      "test_iterhybrid_predictor_corrector_cyclotron",
      "test_iterhybrid_newton",
      "test_iterhybrid_rampup",
      "test_time_dependent_circular_geo",
      "test_changing_config_before",
      "test_changing_config_after",
      "test_psichease_ip_parameters_vloop",
      "test_psichease_ip_chease_vloop",
      "test_psichease_prescribed_jtot_vloop",
  )
  def test_full_config_construction(self, config_name):
    """Test for basic config construction."""

    module = config_loader.import_module(
        f".tests.test_data.{config_name}",
        config_package="torax",
    )

    # Test only the subset of config fields that are currently supported.
    module_config = {
        key: module.CONFIG[key]
        for key in model_config.ToraxConfig.model_fields.keys()
    }
    config_pydantic = model_config.ToraxConfig.from_dict(module_config)

    self.assertEqual(
        config_pydantic.time_step_calculator.calculator_type.value,
        module_config["time_step_calculator"]["calculator_type"],
    )
    self.assertEqual(
        config_pydantic.pedestal.pedestal_config.pedestal_model,
        module_config["pedestal"]["pedestal_model"]
        if "pedestal_model" in module_config["pedestal"]
        else "set_tped_nped",
    )
    # The full model should always be serializable.
    with self.subTest("json_serialization"):
      config_json = config_pydantic.model_dump_json()
      config_pydantic_roundtrip = model_config.ToraxConfig.model_validate_json(
          config_json
      )
      chex.assert_trees_all_equal(config_pydantic, config_pydantic_roundtrip)


if __name__ == "__main__":
  absltest.main()
