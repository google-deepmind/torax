# Copyright 2025 DeepMind Technologies Limited
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

"""Testing the public API of the transport package."""

import dataclasses
from typing import Annotated, Literal

from absl.testing import absltest
import jax.numpy as jnp
import torax
from torax import transport
from torax._src.test_utils import default_configs


@dataclasses.dataclass(frozen=True, eq=False)
class FakeTransportModel(transport.TransportModel):
  """Fake transport model that always returns zeros."""

  def call_implementation(
      self,
      transport_runtime_params: transport.RuntimeParams,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      pedestal_model_outputs: torax.PedestalModelOutput,
  ) -> transport.TurbulentTransport:
    return transport.TurbulentTransport(
        chi_face_ion=jnp.zeros_like(geo.rho_face_norm),
        chi_face_el=jnp.zeros_like(geo.rho_face_norm),
        d_face_el=jnp.zeros_like(geo.rho_face_norm),
        v_face_el=jnp.zeros_like(geo.rho_face_norm),
    )


class FakeTransportPydantic(transport.TransportBase):
  """Fake transport model pydantic config."""

  model_name: Annotated[Literal['fake_api'], torax.JAX_STATIC] = 'fake_api'

  def build_transport_model(self) -> FakeTransportModel:
    return FakeTransportModel()


transport.register_transport_model(FakeTransportPydantic)


class TransportTest(absltest.TestCase):

  def test_fake_transport_model(self):
    """Tests that the fake transport model returns zeros."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'fake_api',
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)

  def test_fake_transport_model_with_combined(self):
    """Tests that the fake transport model returns zeros."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {'model_name': 'fake_api', 'rho_max': 0.5},
            {'model_name': 'fake_api', 'rho_min': 0.5},
        ],
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)

if __name__ == '__main__':
  absltest.main()
