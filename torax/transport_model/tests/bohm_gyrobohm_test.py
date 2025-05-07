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

from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax.config import build_runtime_params
from torax.config import numerics
from torax.config import plasma_composition
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import default_configs
from torax.torax_pydantic import model_config
from torax.transport_model import bohm_gyrobohm


# pylint: disable=invalid-name
class BohmGyroBohmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    config = default_configs.get_default_config_dict()
    config['transport'] = {'transport_model': 'bohm-gyrobohm'}
    torax_config = model_config.ToraxConfig.from_dict(config)
    self.model = torax_config.transport.build_transport_model()
    self.geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )
    self.core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        self.geo,
        source_models_lib.SourceModels(
            sources=torax_config.sources.source_model_config
        ),
    )
    # pedestal_model_outputs is not used in the transport model; we can mock it.
    self.pedestal_outputs = mock.create_autospec(object)

  def _create_dynamic_params_slice(
      self,
      chi_e_bohm_coeff,
      chi_e_gyrobohm_coeff,
      chi_i_bohm_coeff,
      chi_i_gyrobohm_coeff,
      chi_e_bohm_multiplier,
      chi_e_gyrobohm_multiplier,
      chi_i_bohm_multiplier,
      chi_i_gyrobohm_multiplier,
  ):
    """Creates a mock dynamic runtime params slice for the BohmGyroBohm model."""
    transport_mock = mock.create_autospec(
        bohm_gyrobohm.DynamicRuntimeParams,
        instance=True,
        chi_e_bohm_coeff=chi_e_bohm_coeff,
        chi_e_gyrobohm_coeff=chi_e_gyrobohm_coeff,
        chi_i_bohm_coeff=chi_i_bohm_coeff,
        chi_i_gyrobohm_coeff=chi_i_gyrobohm_coeff,
        d_face_c1=0.1,
        d_face_c2=0.2,
        v_face_coeff=0.3,
        chi_e_bohm_multiplier=chi_e_bohm_multiplier,
        chi_e_gyrobohm_multiplier=chi_e_gyrobohm_multiplier,
        chi_i_bohm_multiplier=chi_i_bohm_multiplier,
        chi_i_gyrobohm_multiplier=chi_i_gyrobohm_multiplier,
    )
    plasma_composition_mock = mock.create_autospec(
        plasma_composition.PlasmaComposition,
        instance=True,
        Zeff_face=jnp.ones_like(self.geo.rho_face),
        main_ion=mock.create_autospec(
            plasma_composition.DynamicIonMixture,
            instance=True,
            avg_A=2.0,
        ),
    )
    numerics_mock = mock.create_autospec(
        numerics.Numerics,
        instance=True,
        density_reference=100,
    )

    # Create the dynamic runtime params slice mock with nested mocks.
    dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        instance=True,
        transport=transport_mock,
        plasma_composition=plasma_composition_mock,
        numerics=numerics_mock,
    )
    return dynamic_params

  def test_coeff_multiplier_feature(self):
    """Test that modifying coefficients or multipliers equivalently affects outputs.

    Verifies that if the product of coefficient and multiplier is held constant—
    either by changing the coefficient with multipliers left at 1 or by leaving
    the coefficients at default (1) and scaling the multipliers—the computed
    transport coefficients (chi_face_ion and chi_face_el) remain identical.
    """
    # Configuration A: Set non-default coefficients with all multipliers = 1.
    dyn_params_A = self._create_dynamic_params_slice(
        chi_e_bohm_coeff=2.0,
        chi_e_gyrobohm_coeff=3.0,
        chi_i_bohm_coeff=4.0,
        chi_i_gyrobohm_coeff=5.0,
        chi_e_bohm_multiplier=1.0,
        chi_e_gyrobohm_multiplier=1.0,
        chi_i_bohm_multiplier=1.0,
        chi_i_gyrobohm_multiplier=1.0,
    )

    # Configuration B: Set coefficients = 1 and adjust multipliers so that the
    # effective products are the same as in configuration A.
    dyn_params_B = self._create_dynamic_params_slice(
        chi_e_bohm_coeff=1.0,
        chi_e_gyrobohm_coeff=1.0,
        chi_i_bohm_coeff=1.0,
        chi_i_gyrobohm_coeff=1.0,
        chi_e_bohm_multiplier=2.0,
        chi_e_gyrobohm_multiplier=3.0,
        chi_i_bohm_multiplier=4.0,
        chi_i_gyrobohm_multiplier=5.0,
    )

    output_A = self.model._call_implementation(
        dyn_params_A, self.geo, self.core_profiles, self.pedestal_outputs
    )
    output_B = self.model._call_implementation(
        dyn_params_B, self.geo, self.core_profiles, self.pedestal_outputs
    )

    np.testing.assert_allclose(output_A.chi_face_ion, output_B.chi_face_ion)
    np.testing.assert_allclose(output_A.chi_face_el, output_B.chi_face_el)

  def test_raw_bohm_and_gyrobohm_fields(self):
    """Test that the raw Bohm and gyro-Bohm fields are computed consistently."""
    # Configuration A: Non-default coefficients with multipliers set to 1.
    dyn_params_A = self._create_dynamic_params_slice(
        chi_e_bohm_coeff=2.0,
        chi_e_gyrobohm_coeff=3.0,
        chi_i_bohm_coeff=4.0,
        chi_i_gyrobohm_coeff=5.0,
        chi_e_bohm_multiplier=1.0,
        chi_e_gyrobohm_multiplier=1.0,
        chi_i_bohm_multiplier=1.0,
        chi_i_gyrobohm_multiplier=1.0,
    )

    # Configuration B: Coefficients set to 1 and multipliers adjusted so that
    # the effective products remain the same as in configuration A.
    dyn_params_B = self._create_dynamic_params_slice(
        chi_e_bohm_coeff=1.0,
        chi_e_gyrobohm_coeff=1.0,
        chi_i_bohm_coeff=1.0,
        chi_i_gyrobohm_coeff=1.0,
        chi_e_bohm_multiplier=2.0,
        chi_e_gyrobohm_multiplier=3.0,
        chi_i_bohm_multiplier=4.0,
        chi_i_gyrobohm_multiplier=5.0,
    )

    output_A = self.model._call_implementation(
        dyn_params_A, self.geo, self.core_profiles, self.pedestal_outputs
    )
    output_B = self.model._call_implementation(
        dyn_params_B, self.geo, self.core_profiles, self.pedestal_outputs
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
