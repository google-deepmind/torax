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

"""Testing the public API of the pedestal package."""

import dataclasses
from typing import Annotated, Literal

from absl.testing import absltest
import jax.numpy as jnp
import torax
from torax import pedestal
from torax._src.test_utils import default_configs


@dataclasses.dataclass(frozen=True, eq=False)
class FakePedestalModel(pedestal.PedestalStructureModel):
  """Fake pedestal structure model that returns fixed values."""

  def __call__(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      structure_state: pedestal.StructureState,
  ) -> tuple[pedestal.PedestalStructure, pedestal.StructureState]:
    del runtime_params, geo, core_profiles
    return (
        pedestal.PedestalStructure(
            rho_norm_ped_top=jnp.array(0.9),
            T_i_ped=jnp.array(5.0),
            T_e_ped=jnp.array(5.0),
            n_e_ped=jnp.array(0.7e20),
        ),
        structure_state,
    )


class FakePedestalPydantic(pedestal.BaseStructure):
  """Fake pedestal structure model pydantic config."""

  model_name: Annotated[Literal['fake_pedestal'], torax.JAX_STATIC] = (
      'fake_pedestal'
  )

  def build_structure_model(self) -> FakePedestalModel:
    return FakePedestalModel()


pedestal.register_pedestal_structure_model(FakePedestalPydantic)


class PedestalTest(absltest.TestCase):

  def test_fake_pedestal_model(self):
    """Tests that the fake pedestal model can be used in a simulation."""
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'structure_model': {
            'model_name': 'fake_pedestal',
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


if __name__ == '__main__':
  absltest.main()
