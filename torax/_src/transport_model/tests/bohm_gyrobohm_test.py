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
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import bohm_gyrobohm
from torax._src.transport_model import pydantic_model


# pylint: disable=invalid-name
class BohmGyroBohmTest(absltest.TestCase):

  def _build_model_and_params(self, **bgb_params):
    bgb_config = pydantic_model.BohmGyroBohmTransportModel(
        D_face_c1=0.1,
        D_face_c2=0.2,
        V_face_coeff=0.3,
        **bgb_params,
    )
    model = bgb_config.build_transport_model()
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    t = torax_config.numerics.t_initial
    bgb_runtime_params = bgb_config.build_runtime_params(t)
    geo = torax_config.geometry.build_provider(t=t)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=t)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    return (
        model,
        bgb_runtime_params,
        runtime_params,
        geo,
        core_profiles,
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
        bgb_params_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
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
        bgb_params_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
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
        bgb_params_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
        two_point_mask_A,
    )
    output_B = model_B(
        bgb_params_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
        two_point_mask_B,
    )

    np.testing.assert_allclose(output_A.chi_face_ion, output_B.chi_face_ion)
    np.testing.assert_allclose(output_A.chi_face_el, output_B.chi_face_el)

  def test_raw_bohm_and_gyrobohm_fields(self):
    """Test that the raw Bohm and gyro-Bohm fields are computed consistently."""
    (
        model_A,
        bgb_params_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
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
        bgb_params_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
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
        bgb_params_A,
        runtime_params_A,
        geo_A,
        core_profiles_A,
        two_point_mask_A,
    )
    output_B = model_B(
        bgb_params_B,
        runtime_params_B,
        geo_B,
        core_profiles_B,
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

  def test_to_output_dict(self):
    n_face = 10
    bgb_output = bohm_gyrobohm.BohmGyroBohmTransportOutput(
        chi_face_ion=jnp.ones((1, n_face)) * 1.0,
        chi_face_el=jnp.ones((1, n_face)) * 2.0,
        d_face_el=jnp.ones((1, n_face)) * 0.5,
        v_face_el=jnp.ones((1, n_face)) * -0.1,
        chi_face_el_bohm=jnp.ones((1, n_face)) * 0.2,
        chi_face_el_gyrobohm=jnp.ones((1, n_face)) * 1.8,
        chi_face_ion_bohm=jnp.ones((1, n_face)) * 0.1,
        chi_face_ion_gyrobohm=jnp.ones((1, n_face)) * 0.9,
    )
    context = output_grid_context.OutputGridContext(
        times=np.array([0.0]),
        rho_face_norm=np.linspace(0, 1, n_face),
        rho_cell_norm=np.linspace(0, 1, n_face - 1),
        rho_cell_plus_boundaries_norm=np.linspace(0, 1, n_face + 1),
    )
    out_dict = bgb_output.to_output_dict(context)
    self.assertIn(output_keys.CHI_TURB_I, out_dict)
    self.assertIn(output_keys.CHI_TURB_E, out_dict)
    self.assertIn(output_keys.D_TURB_E, out_dict)
    self.assertIn(output_keys.V_TURB_E, out_dict)
    self.assertIn(output_keys.CHI_BOHM_E, out_dict)
    self.assertIn(output_keys.CHI_GYROBOHM_E, out_dict)
    self.assertIn(output_keys.CHI_BOHM_I, out_dict)
    self.assertIn(output_keys.CHI_GYROBOHM_I, out_dict)


if __name__ == '__main__':
  absltest.main()

