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
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax._src.pedestal_model import register_model
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


class RegisterPedestalModelTest(absltest.TestCase):

  def test_registered_model_in_config(self):
    register_model.register_pedestal_model(FixedPedestalConfig)

    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'fixed_pedestal',
        'set_pedestal': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    self.assertIsInstance(pedestal_model, FixedPedestalModel)


@dataclasses.dataclass(frozen=True, eq=False)
class FixedPedestalModel(pedestal_model_lib.PedestalModel):
  """Fixed PedestalModel for testing purposes."""

  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> pedestal_model_output_lib.PedestalModelOutput:
    return pedestal_model_output_lib.PedestalModelOutput(
        rho_norm_ped_top=jnp.array(0.9),
        rho_norm_ped_top_idx=jnp.abs(geo.rho_norm - 0.9).argmin(),
        T_i_ped=jnp.array(5.0),
        T_e_ped=jnp.array(5.0),
        n_e_ped=jnp.array(0.7e20),
    )


class FixedPedestalConfig(pedestal_pydantic_model.BasePedestal):
  """Fixed pedestal config for testing."""

  model_name: Annotated[
      Literal['fixed_pedestal'], torax_pydantic.JAX_STATIC
  ] = 'fixed_pedestal'

  def build_pedestal_model(
      self,
  ) -> FixedPedestalModel:
    return FixedPedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self,
      t,
  ) -> pedestal_runtime_params_lib.RuntimeParams:
    return pedestal_runtime_params_lib.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        mode=self.mode,
        formation=self.formation_model.build_runtime_params(t),
        saturation=self.saturation_model.build_runtime_params(t),
        chi_max=self.chi_max,
        D_e_max=self.D_e_max,
        V_e_max=self.V_e_max,
        V_e_min=self.V_e_min,
    )


if __name__ == '__main__':
  absltest.main()
