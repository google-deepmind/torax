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

"""Tests for the edge model registration."""

import dataclasses
from typing import Annotated, ClassVar, Literal

from absl.testing import absltest
import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import base
from torax._src.edge import register_model
from torax._src.geometry import geometry
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


class RegisterEdgeModelTest(absltest.TestCase):

  def test_registered_model_in_config(self):
    register_model.register_edge_model(DummyEdgeConfig)

    config = default_configs.get_default_config_dict()
    # Non-circular geometry is required because ToraxConfig validates that edge
    # models are not used with CircularGeometry.
    config["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config["edge"] = {"model_name": "dummy_edge"}
    torax_config = model_config.ToraxConfig.from_dict(config)
    self.assertIsNotNone(torax_config.edge)
    edge_model = torax_config.edge.build_edge_model()
    self.assertIsInstance(edge_model, DummyEdgeModel)

  def test_update_flags_default_to_supported_bcs(self):
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = default_configs.get_default_config_dict()
    config["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config["edge"] = {"model_name": "dummy_te"}
    torax_config = model_config.ToraxConfig.from_dict(config)
    assert torax_config.edge is not None
    runtime_params = torax_config.edge.build_runtime_params(t=0.0)
    self.assertTrue(runtime_params.update_T_e)
    self.assertFalse(runtime_params.update_T_i)
    self.assertFalse(runtime_params.update_n_e)
    self.assertFalse(runtime_params.update_impurity)

  def test_error_if_update_flag_set_for_unsupported_bc(self):
    """Test that an error is raised if an unsupported BC is updated."""
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = default_configs.get_default_config_dict()
    config["geometry"] = {
        "geometry_type": "chease",
        "geometry_file": "iterhybrid.mat2cols",
    }
    config["edge"] = {
        "model_name": "dummy_te",
        "update_T_i": True,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"'update_T_i' cannot be True because 'dummy_te' does not"
        r" support the 'T_i' boundary condition\.",
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_dynamic_registration_updates_discriminator(self):
    register_model.register_edge_model(DummyEdgeConfig)

    config = default_configs.get_default_config_dict()
    config["edge"] = {"model_name": "invalid_edge_model"}
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_edge_model' found using 'model_name' does not"
        r" match any.*'dummy_edge'",
    ):
      model_config.ToraxConfig.from_dict(config)


@dataclasses.dataclass(frozen=True, eq=False)
class DummyEdgeModel(base.EdgeModel):
  """Dummy edge model for testing purposes."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      previous_edge_outputs: base.EdgeModelOutputs | None = None,
  ) -> base.EdgeModelOutputs:
    del runtime_params, geo, core_profiles, core_sources, previous_edge_outputs
    return base.EdgeModelOutputs(
        T_i_right_bc=jnp.array(80.0),
        T_e_right_bc=jnp.array(120.0),
        n_e_right_bc=jnp.array(2.0e19),
        impurity_right_bc={},
    )


class DummyEdgeConfig(base.EdgeModelConfig):
  """Dummy edge model configuration for testing."""

  model_name: Annotated[Literal["dummy_edge"], torax_pydantic.JAX_STATIC] = (
      "dummy_edge"
  )

  def build_edge_model(self) -> DummyEdgeModel:
    return DummyEdgeModel()


@dataclasses.dataclass(frozen=True, eq=False)
class DummyElectronTemperatureModel(DummyEdgeModel):
  """Dummy edge model that only computes the electron temperature BC."""


class DummyElectronTemperatureConfig(base.EdgeModelConfig):
  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(T_e=True)
  )
  model_name: Annotated[Literal["dummy_te"], torax_pydantic.JAX_STATIC] = (
      "dummy_te"
  )

  def build_edge_model(self) -> DummyElectronTemperatureModel:
    return DummyElectronTemperatureModel()


if __name__ == "__main__":
  absltest.main()
