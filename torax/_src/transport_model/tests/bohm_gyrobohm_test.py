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
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config


# pylint: disable=invalid-name
class BohmGyroBohmTest(absltest.TestCase):

  def _build_model_and_params(self, **bgb_params):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        # Set min clipping to 0.0 to avoid values being clipped and hiding
        # results.
        'chi_min': 0.0,
        'chi_max': 1e9,
        'D_e_min': 0.0,
        'V_e_min': 0.0,
        'core_transport_models': {
            'bohm_gyrobohm': {
                'model_name': 'bohm-gyrobohm',
                'D_face_c1': 0.1,
                'D_face_c2': 0.2,
                'V_face_coeff': 0.3,
                **bgb_params,
            },
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    pedestal_outputs = pedestal_model_output_lib.PedestalModelOutput(
        rho_norm_ped_top=1.0,
        T_i_ped=0.0,
        T_e_ped=0.0,
        n_e_ped=0.0,
    )
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    return (
        model,
        runtime_params,
        geo,
        core_profiles,
        pedestal_outputs,
        two_point_mask,
    )

  def test_coeff_multiplier_feature(self):
    """Test that modifying coefficients or multipliers equivalently affects outputs.

    Verifies that if the product of coefficient and multiplier is held constant—
    either by changing the coefficient with multipliers left at 1 or by leaving
    the coefficients at default (1) and scaling the multipliers—the computed
    transport coefficients (chi_face_ion and chi_face_el) remain identical.
    """
    (
        model_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
        pedestal_outputs_A,
        two_point_mask_A,
    ) = self._build_model_and_params(
        chi_e_bohm_coeff=2.0,
        chi_e_gyrobohm_coeff=3.0,
        chi_i_bohm_coeff=4.0,
        chi_i_gyrobohm_coeff=5.0,
        chi_e_bohm_multiplier=1.0,
        chi_e_gyrobohm_multiplier=1.0,
        chi_i_bohm_multiplier=1.0,
        chi_i_gyrobohm_multiplier=1.0,
    )

    (
        model_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
        pedestal_outputs_B,
        two_point_mask_B,
    ) = self._build_model_and_params(
        chi_e_bohm_coeff=1.0,
        chi_e_gyrobohm_coeff=1.0,
        chi_i_bohm_coeff=1.0,
        chi_i_gyrobohm_coeff=1.0,
        chi_e_bohm_multiplier=2.0,
        chi_e_gyrobohm_multiplier=3.0,
        chi_i_bohm_multiplier=4.0,
        chi_i_gyrobohm_multiplier=5.0,
    )

    output_A = model_A(
        runtime_params_A,
        geo_A,
        core_profiles_A,
        pedestal_outputs_A,
        two_point_mask_A,
    )
    output_B = model_B(
        runtime_params_B,
        geo_B,
        core_profiles_B,
        pedestal_outputs_B,
        two_point_mask_B,
    )

    np.testing.assert_allclose(output_A.chi_face_ion, output_B.chi_face_ion)
    np.testing.assert_allclose(output_A.chi_face_el, output_B.chi_face_el)

  def test_raw_bohm_and_gyrobohm_fields(self):
    """Test that the raw Bohm and gyro-Bohm fields are computed consistently."""
    (
        model_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
        pedestal_outputs_A,
        two_point_mask_A,
    ) = self._build_model_and_params(
        chi_e_bohm_coeff=2.0,
        chi_e_gyrobohm_coeff=3.0,
        chi_i_bohm_coeff=4.0,
        chi_i_gyrobohm_coeff=5.0,
        chi_e_bohm_multiplier=1.0,
        chi_e_gyrobohm_multiplier=1.0,
        chi_i_bohm_multiplier=1.0,
        chi_i_gyrobohm_multiplier=1.0,
    )

    (
        model_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
        pedestal_outputs_B,
        two_point_mask_B,
    ) = self._build_model_and_params(
        chi_e_bohm_coeff=1.0,
        chi_e_gyrobohm_coeff=1.0,
        chi_i_bohm_coeff=1.0,
        chi_i_gyrobohm_coeff=1.0,
        chi_e_bohm_multiplier=2.0,
        chi_e_gyrobohm_multiplier=3.0,
        chi_i_bohm_multiplier=4.0,
        chi_i_gyrobohm_multiplier=5.0,
    )

    output_A = model_A(
        runtime_params_A,
        geo_A,
        core_profiles_A,
        pedestal_outputs_A,
        two_point_mask_A,
    )
    output_B = model_B(
        runtime_params_B,
        geo_B,
        core_profiles_B,
        pedestal_outputs_B,
        two_point_mask_B,
    )

    # Verify that the raw fields (which are computed before applying the
    # scaling factors) are identical between the two configurations.
    np.testing.assert_allclose(
        output_A.chi_face_el_bohm, output_B.chi_face_el_bohm
    )
    np.testing.assert_allclose(
        output_A.chi_face_el_gyrobohm, output_B.chi_face_el_gyrobohm
    )
    np.testing.assert_allclose(
        output_A.chi_face_ion_bohm, output_B.chi_face_ion_bohm
    )
    np.testing.assert_allclose(
        output_A.chi_face_ion_gyrobohm, output_B.chi_face_ion_gyrobohm
    )

    # Verify the raw fields add up to the total fields.
    np.testing.assert_allclose(
        output_A.chi_face_ion_bohm + output_A.chi_face_ion_gyrobohm,
        output_A.chi_face_ion,
    )
    np.testing.assert_allclose(
        output_A.chi_face_el_bohm + output_A.chi_face_el_gyrobohm,
        output_A.chi_face_el,
    )
    np.testing.assert_allclose(
        output_B.chi_face_ion_bohm + output_B.chi_face_ion_gyrobohm,
        output_B.chi_face_ion,
    )
    np.testing.assert_allclose(
        output_B.chi_face_el_bohm + output_B.chi_face_el_gyrobohm,
        output_B.chi_face_el,
    )


if __name__ == '__main__':
  absltest.main()

