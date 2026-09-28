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
from torax._src.edge import pydantic_model as edge_pydantic_model
from torax._src.test_utils import default_configs


@dataclasses.dataclass(frozen=True, eq=False)
class FakeEdgeModel(edge.EdgeModel):
  """Fake edge model that returns fixed boundary condition values."""

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


class FakeEdgeConfig(edge.EdgeModelConfig):
  """Fake edge model pydantic config."""

  model_name: Annotated[Literal["fake_edge"], torax.JAX_STATIC] = "fake_edge"

  def build_edge_model(self) -> FakeEdgeModel:
    return FakeEdgeModel()


class EdgePublicApiTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    original_edge_config = edge_pydantic_model.EdgeConfig
    original_annotation = torax.ToraxConfig.model_fields["edge"].annotation

    def _restore_config():
      setattr(edge_pydantic_model, "EdgeConfig", original_edge_config)
      setattr(
          torax.ToraxConfig.model_fields["edge"],
          "annotation",
          original_annotation,
      )
      torax.ToraxConfig.model_rebuild(force=True)

    self.addCleanup(_restore_config)
    edge.register_edge_model(FakeEdgeConfig)

  def test_registered_edge_model_in_torax_config(self):
    config = default_configs.get_default_config_dict()
    config["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config["edge"] = {
        "model_name": "fake_edge",
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    self.assertIsNotNone(torax_config.edge)
    edge_model = torax_config.edge.build_edge_model()
    self.assertIsInstance(edge_model, FakeEdgeModel)
    params = torax_config.edge.build_runtime_params(t=jnp.array(0.0))
    self.assertIsInstance(params, edge.RuntimeParams)

  def test_fake_edge_model_in_simulation(self):
    config = default_configs.get_default_config_dict()
    config["plasma_composition"]["impurity"] = {
        "impurity_mode": "n_e_ratios",
        "species": {
            "N": {0: 0.01, 1: 0.01},
        },
    }
    config["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config["edge"] = {
        "model_name": "fake_edge",
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    torax.run_simulation(torax_config)


if __name__ == "__main__":
  absltest.main()
