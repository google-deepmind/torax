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
class FakePedestalModel(pedestal.PedestalModel):
  """Fake pedestal model that returns fixed values."""

  def _call_implementation(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
  ) -> pedestal.PedestalModelOutput:
    return pedestal.PedestalModelOutput(
        rho_norm_ped_top=jnp.array(0.9),
        rho_norm_ped_top_idx=jnp.abs(geo.rho_norm - 0.9).argmin(),
        T_i_ped=jnp.array(5.0),
        T_e_ped=jnp.array(5.0),
        n_e_ped=jnp.array(0.7e20),
    )


class FakePedestalPydantic(pedestal.BasePedestal):
  """Fake pedestal model pydantic config."""

  model_name: Annotated[Literal['fake_pedestal'], torax.JAX_STATIC] = (
      'fake_pedestal'
  )

  def build_pedestal_model(self) -> FakePedestalModel:
    return FakePedestalModel(
        formation_model=self.formation_model.build_formation_model(),
        saturation_model=self.saturation_model.build_saturation_model(),
    )

  def build_runtime_params(
      self,
      t,
  ) -> pedestal.RuntimeParams:
    return pedestal.RuntimeParams(
        set_pedestal=self.set_pedestal.get_value(t),
        mode=self.mode,
        formation=self.formation_model.build_runtime_params(t),
        saturation=self.saturation_model.build_runtime_params(t),
        chi_max=self.chi_max.get_value(t),
        D_e_max=self.D_e_max.get_value(t),
        V_e_max=self.V_e_max.get_value(t),
        V_e_min=self.V_e_min.get_value(t),
        pedestal_top_smoothing_width=self.pedestal_top_smoothing_width.get_value(
            t
        ),
        use_formation_model_with_adaptive_source=self.use_formation_model_with_adaptive_source,
        transition_time_width=self.transition_time_width.get_value(t),
    )


pedestal.register_pedestal_model(FakePedestalPydantic)


class PedestalTest(absltest.TestCase):

  def test_fake_pedestal_model(self):
    """Tests that the fake pedestal model can be used in a simulation."""
    config = default_configs.get_default_config_dict()
    config['pedestal'] = {
        'model_name': 'fake_pedestal',
        'set_pedestal': True,
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


if __name__ == '__main__':
  absltest.main()
