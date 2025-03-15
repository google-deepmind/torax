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
import jax.numpy as jnp
from unittest import mock
from types import SimpleNamespace

from torax.transport_model.bohm_gyrobohm import BohmGyroBohmTransportModel, DynamicRuntimeParams

from torax.config import build_runtime_params, runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model


class BohmGyroBohmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Patch the QLKNN model loader to avoid FileNotFoundError.
    from torax.transport_model import qlknn_transport_model

    self._old_get_model = qlknn_transport_model.get_model
    qlknn_transport_model.get_model = lambda path: type(
        'DummyQLKNN', (), {'version': 'dummy'}
    )

    self.model = BohmGyroBohmTransportModel()
    self.runtime_params = general_runtime_params.GeneralRuntimeParams()
    self.sources = source_pydantic_model.Sources()
    self.source_models = source_models_lib.SourceModels(
        sources=self.sources.source_model_config
    )
    self.geo = geometry_pydantic_model.CircularConfig().build_geometry()
    # Create a dummy pedestal config (even though it is not used by the transport model)
    self.pedestal = pedestal_pydantic_model.Pedestal.from_dict({
        'pedestal_model': 'set_tped_nped',
        'Tiped': 5,
        'Teped': 4,
        'rho_norm_ped_top': {0.0: 0.5, 1.0: 0.7},
        'neped': 0.7,
        'neped_is_fGW': False,
    })
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        self.runtime_params,
        sources=self.sources,
        torax_mesh=self.geo.torax_mesh,
        pedestal=self.pedestal,
    )
    self.dynamic_runtime_params_slice = provider(t=0.0)
    self.static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=self.runtime_params,
            sources=self.sources,
            torax_mesh=self.geo.torax_mesh,
        )
    )
    self.core_profiles = initialization.initial_core_profiles(
        self.static_runtime_params_slice,
        self.dynamic_runtime_params_slice,
        self.geo,
        self.source_models,
    )
    self.numerics = SimpleNamespace(nref=1.0)
    self.plasma_composition = SimpleNamespace(
        main_ion=SimpleNamespace(avg_A=2.0)
    )
    # pedestal_model_outputs is not used in the transport model; we can mock it.
    self.pedestal_outputs = mock.create_autospec(object)

  def tearDown(self):
    # Revert the monkeypatch.
    from torax.transport_model import qlknn_transport_model

    qlknn_transport_model.get_model = self._old_get_model
    super().tearDown()

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
    """Helper function to create a dummy dynamic runtime params slice.

    Note: We supply dummy values for additional required parameters.
    """
    transport_params = DynamicRuntimeParams(
        # Dummy values for the base fields.
        chimin=0.0,
        chimax=1.0,
        Demin=0.0,
        Demax=1.0,
        Vemin=0.0,
        Vemax=1.0,
        apply_inner_patch=False,
        De_inner=0.0,
        Ve_inner=0.0,
        chii_inner=0.0,
        chie_inner=0.0,
        rho_inner=0.5,
        apply_outer_patch=False,
        De_outer=0.0,
        Ve_outer=0.0,
        chii_outer=0.0,
        chie_outer=0.0,
        rho_outer=1.0,
        smoothing_sigma=1.0,
        smooth_everywhere=True,
        # Transport-model specific fields.
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
    return SimpleNamespace(
        transport=transport_params,
        numerics=self.numerics,
        plasma_composition=self.plasma_composition,
    )

  def test_coeff_multiplier_feature(self):
    """Test that modifying coefficients or multipliers equivalently affects outputs.

    Verifies that if the product of coefficient and multiplier is held constant—
    either by changing the coefficient with multipliers left at 1 or by leaving the
    coefficients at default (1) and scaling the multipliers—the computed transport
    coefficients (chi_face_ion and chi_face_el) remain identical.
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

    # Configuration B: Set coefficients = 1 and adjust multipliers so that the effective
    # products are the same as in configuration A.
    dyn_params_B = self._create_dynamic_params_slice(
        chi_e_bohm_coeff=1.0,
        chi_e_gyrobohm_coeff=1.0,
        chi_i_bohm_coeff=1.0,
        chi_i_gyrobohm_coeff=1.0,
        chi_e_bohm_multiplier=2.0,  # 1.0 * 2.0 = 2.0
        chi_e_gyrobohm_multiplier=3.0,  # 1.0 * 3.0 = 3.0
        chi_i_bohm_multiplier=4.0,  # 1.0 * 4.0 = 4.0
        chi_i_gyrobohm_multiplier=5.0,  # 1.0 * 5.0 = 5.0
    )

    output_A = self.model._call_implementation(
        dyn_params_A, self.geo, self.core_profiles, self.pedestal_outputs
    )
    output_B = self.model._call_implementation(
        dyn_params_B, self.geo, self.core_profiles, self.pedestal_outputs
    )

    self.assertTrue(
        jnp.allclose(output_A.chi_face_ion, output_B.chi_face_ion),
        msg=(
            'chi_face_ion values differ between coefficient and multiplier'
            ' configurations'
        ),
    )
    self.assertTrue(
        jnp.allclose(output_A.chi_face_el, output_B.chi_face_el),
        msg=(
            'chi_face_el values differ between coefficient and multiplier'
            ' configurations'
        ),
    )


if __name__ == '__main__':
  absltest.main()
