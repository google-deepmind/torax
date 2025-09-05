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
from absl.testing import parameterized
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import pydantic_model as neoclassical_pydantic_model
from torax._src.neoclassical.transport import base as neoclassical_transport_base
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import runtime_params as runtime_params_lib


class NeoclassicalTransportTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    neoclassical_pydantic_model.Neoclassical.model_fields[
        'transport'
    ].annotation |= FakeNeoclassicalTransportModelConfig
    neoclassical_pydantic_model.Neoclassical.model_rebuild(force=True)
    model_config.ToraxConfig.model_rebuild(force=True)

  def test_clipping(self):
    config = default_configs.get_default_config_dict()
    config['neoclassical'] = {
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
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(
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
    neoclassical_transport_model = (
        torax_config.neoclassical.transport.build_model()
    )
    neoclassical_transport_coeffs = neoclassical_transport_model(
        runtime_params, geo, core_profiles
    )

    assert np.all(
        neoclassical_transport_coeffs.chi_neo_i
        >= torax_config.neoclassical.transport.chi_min
    ), 'chi_min clipping failed on chi_neo_i'
    assert np.all(
        neoclassical_transport_coeffs.chi_neo_i
        <= torax_config.neoclassical.transport.chi_max
    ), 'chi_max clipping failed on chi_neo_i'

    assert np.all(
        neoclassical_transport_coeffs.chi_neo_e
        >= torax_config.neoclassical.transport.chi_min
    ), 'chi_min clipping failed on chi_neo_e'
    assert np.all(
        neoclassical_transport_coeffs.chi_neo_e
        <= torax_config.neoclassical.transport.chi_max
    ), 'chi_max clipping failed on chi_neo_e'

    assert np.all(
        neoclassical_transport_coeffs.D_neo_e
        >= torax_config.neoclassical.transport.D_e_min
    ), 'D_e_min clipping failed'
    assert np.all(
        neoclassical_transport_coeffs.D_neo_e
        <= torax_config.neoclassical.transport.D_e_max
    ), 'D_e_max clipping failed'

    assert np.all(
        neoclassical_transport_coeffs.V_neo_e
        >= torax_config.neoclassical.transport.V_e_min
    ), 'V_e_min clipping failed on V_neo_e'
    assert np.all(
        neoclassical_transport_coeffs.V_neo_e
        <= torax_config.neoclassical.transport.V_e_max
    ), 'V_e_max clipping failed on V_neo_e'

    assert np.all(
        neoclassical_transport_coeffs.V_neo_ware_e
        >= torax_config.neoclassical.transport.V_e_min
    ), 'V_e_min clipping failed on V_neo_ware_e'
    assert np.all(
        neoclassical_transport_coeffs.V_neo_ware_e
        <= torax_config.neoclassical.transport.V_e_max
    ), 'V_e_max clipping failed on V_neo_ware_e'


class FakeNeoclassicalTransportModel(
    neoclassical_transport_base.NeoclassicalTransportModel
):
  """Fake NeoclassicalTransportModel for testing purposes."""

  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> neoclassical_transport_base.NeoclassicalTransport:
    chi_neo_i = np.linspace(0.5, 2, geometry.rho_face_norm.shape[0])
    chi_neo_e = np.linspace(0.25, 1, geometry.rho_face_norm.shape[0])
    D_neo_e = np.linspace(2, 3, geometry.rho_face_norm.shape[0])
    V_neo_e = np.linspace(-0.2, -2, geometry.rho_face_norm.shape[0])
    V_neo_ware_e = np.linspace(-0.1, -1, geometry.rho_face_norm.shape[0])
    return neoclassical_transport_base.NeoclassicalTransport(
        chi_neo_i=chi_neo_i,
        chi_neo_e=chi_neo_e,
        D_neo_e=D_neo_e,
        V_neo_e=V_neo_e,
        V_neo_ware_e=V_neo_ware_e,
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
