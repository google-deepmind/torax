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

"""Testing the public API of the edge package."""

import dataclasses
from typing import Annotated, Literal

from absl.testing import absltest
import jax.numpy as jnp
import torax
from torax import edge
from torax._src.test_utils import default_configs


@dataclasses.dataclass(frozen=True, eq=False)
class _CustomEdgeModel(edge.EdgeModel):
  """Custom edge model that returns fixed boundary condition values."""

  def __call__(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      core_sources: torax.SourceProfiles,
      previous_edge_outputs: edge.EdgeModelOutputs | None = None,
  ) -> edge.EdgeModelOutputs:
    del runtime_params, geo, core_profiles, core_sources, previous_edge_outputs
    return edge.EdgeModelOutputs(
        T_i_right_bc=jnp.array(80.0),
        T_e_right_bc=jnp.array(120.0),
        n_e_right_bc=jnp.array(2.0e19),
        impurity_right_bc={},
    )


class _CustomEdgeModelConfig(edge.EdgeModelConfig):
  """Custom edge model pydantic config."""

  model_name: Annotated[Literal['custom_edge'], torax.JAX_STATIC] = (
      'custom_edge'
  )

  def build_edge_model(self) -> _CustomEdgeModel:
    return _CustomEdgeModel()


edge.register_edge_model(_CustomEdgeModelConfig)


class EdgeTest(absltest.TestCase):

  def test_custom_edge_model_runs(self):
    """Tests that the custom edge model can be used in a simulation."""
    config = default_configs.get_default_config_dict()
    config['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config['edge'] = {
        'model_name': 'custom_edge',
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


if __name__ == '__main__':
  absltest.main()
