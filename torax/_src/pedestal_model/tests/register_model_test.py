# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for the pedestal model registration."""

import dataclasses
from typing import Annotated, Literal

from absl.testing import absltest
import jax.numpy as jnp
import pydantic
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.pedestal_model import config as pedestal_config
from torax._src.pedestal_model import pedestal_state as pedestal_state_lib
from torax._src.pedestal_model import register_model
from torax._src.pedestal_model.structure import base as structure_base
from torax._src.pedestal_model.structure import pedestal_structure
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic


class RegisterPedestalModelTest(absltest.TestCase):

  def test_registered_model_in_config(self):
    register_model.register_pedestal_structure_model(FixedPedestalConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict['pedestal'] = {
        'structure_model': {
            'model_name': 'fixed_pedestal',
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.assertIsInstance(pedestal_model.structure_model, FixedPedestalModel)

  def test_dynamic_registration_updates_discriminator(self):
    register_model.register_pedestal_structure_model(FixedPedestalConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict['pedestal'] = {
        'structure_model': {'model_name': 'invalid_name'}
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_name' found using 'model_name' does not match any"
        r".*'fixed_pedestal'",
    ):
      config.ToraxConfig.from_dict(config_dict)


@dataclasses.dataclass(frozen=True, eq=False)
class FixedPedestalModel(structure_base.PedestalStructureModel):
  """Fixed PedestalStructureModel for testing purposes."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: profiles.CoreProfiles,
      structure_state: pedestal_state_lib.StructureState,
  ) -> tuple[
      pedestal_structure.PedestalStructure,
      pedestal_state_lib.StructureState,
  ]:
    del runtime_params, geo, core_profiles
    return (
        pedestal_structure.PedestalStructure(
            rho_norm_ped_top=jnp.array(0.9),
            T_i_ped=jnp.array(5.0),
            T_e_ped=jnp.array(5.0),
            n_e_ped=jnp.array(0.7e20),
        ),
        structure_state,
    )


class FixedPedestalConfig(pedestal_config.StructureConfigBase):
  """Fixed pedestal structure config for testing."""

  model_name: Annotated[
      Literal['fixed_pedestal'], torax_pydantic.JAX_STATIC
  ] = 'fixed_pedestal'

  def build_structure_model(
      self,
  ) -> FixedPedestalModel:
    return FixedPedestalModel()


if __name__ == '__main__':
  absltest.main()
