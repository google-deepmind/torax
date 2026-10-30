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

from typing import Annotated, Literal

from absl.testing import absltest
import numpy as np
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import register_model
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.transport import base as neoclassical_transport_base
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs

# pylint: disable=invalid-name


class NeoclassicalTransportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    register_model.register_neoclassical_transport_model(
        FakeNeoclassicalTransportModelConfig
    )

  def test_clipping(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['neoclassical'] = {
        'transport': {
            'model_name': 'fake',
            'chi_min': 0.75,
            'chi_max': 1.25,
            'D_e_min': 2.25,
            'D_e_max': 2.75,
            'V_e_min': -1.25,
            'V_e_max': -0.75,
        }
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    core_profiles, _, geo = core_profile_helpers.make_initial_inputs(
        torax_config
    )
    neoclassical_transport_model = (
        torax_config.neoclassical.transport.build_model()
    )
    transport_params = (
        torax_config.neoclassical.transport.build_runtime_params()
    )
    neoclassical_intermediates = formulas.compute_neoclassical_intermediates(
        geo, core_profiles, fast_ions=()
    )
    neoclassical_transport_coeffs = neoclassical_transport_model(
        transport_params, geo, core_profiles, neoclassical_intermediates
    )

    assert np.all(
        neoclassical_transport_coeffs.chi_face_ion
        >= torax_config.neoclassical.transport.chi_min
    ), 'chi_min clipping failed on chi_face_ion'
    assert np.all(
        neoclassical_transport_coeffs.chi_face_ion
        <= torax_config.neoclassical.transport.chi_max
    ), 'chi_max clipping failed on chi_face_ion'

    assert np.all(
        neoclassical_transport_coeffs.chi_face_el
        >= torax_config.neoclassical.transport.chi_min
    ), 'chi_min clipping failed on chi_face_el'
    assert np.all(
        neoclassical_transport_coeffs.chi_face_el
        <= torax_config.neoclassical.transport.chi_max
    ), 'chi_max clipping failed on chi_face_el'

    assert np.all(
        neoclassical_transport_coeffs.d_face_el
        >= torax_config.neoclassical.transport.D_e_min
    ), 'D_e_min clipping failed'
    assert np.all(
        neoclassical_transport_coeffs.d_face_el
        <= torax_config.neoclassical.transport.D_e_max
    ), 'D_e_max clipping failed'

    assert np.all(
        neoclassical_transport_coeffs.v_face_el
        >= torax_config.neoclassical.transport.V_e_min
    ), 'V_e_min clipping failed on v_face_el'
    assert np.all(
        neoclassical_transport_coeffs.v_face_el
        <= torax_config.neoclassical.transport.V_e_max
    ), 'V_e_max clipping failed on v_face_el'

    # v_face_el_ware should remain unclipped.
    raw_coeffs = neoclassical_transport_model._call_implementation(
        transport_params, geo, core_profiles, neoclassical_intermediates
    )
    np.testing.assert_allclose(
        neoclassical_transport_coeffs.v_face_el_ware,
        raw_coeffs.v_face_el_ware,
    )


class FakeNeoclassicalTransportModel(
    neoclassical_transport_base.NeoclassicalTransportModel
):
  """Fake NeoclassicalTransportModel for testing purposes."""

  def _call_implementation(
      self,
      runtime_params: transport_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: profiles.CoreProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> transport_coeffs.NeoclassicalTransport:
    del runtime_params, neoclassical_intermediates
    chi_face_ion = np.linspace(0.5, 2, geometry.rho_face_norm.shape[0])
    chi_face_el = np.linspace(0.25, 1, geometry.rho_face_norm.shape[0])
    d_face_el = np.linspace(2, 3, geometry.rho_face_norm.shape[0])
    v_conv = np.linspace(-0.2, -2, geometry.rho_face_norm.shape[0])
    v_face_el_ware = np.linspace(-0.1, -1, geometry.rho_face_norm.shape[0])
    return transport_coeffs.NeoclassicalTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_conv + v_face_el_ware,
        v_face_el_ware=v_face_el_ware,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))


class FakeNeoclassicalTransportModelConfig(
    neoclassical_transport_base.NeoclassicalTransportModelConfig
):
  model_name: Annotated[Literal['fake'], torax_pydantic.JAX_STATIC] = 'fake'

  def build_model(self) -> FakeNeoclassicalTransportModel:
    return FakeNeoclassicalTransportModel()
