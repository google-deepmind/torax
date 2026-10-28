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
from absl.testing import parameterized
import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.edge import base
from torax._src.edge import register_model
from torax._src.edge import runtime_params as edge_runtime_params_lib
from torax._src.geometry import geometry
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import torax_pydantic


def _base_config_dict() -> dict[str, object]:
  config_dict = default_configs.get_default_config_dict()
  # Non-circular geometry and n_e_ratios impurity mode are required when an
  # edge model is configured.
  config_dict["geometry"] = {
      "geometry_type": "chease",
      "geometry_file": "iterhybrid.mat2cols",
  }
  config_dict["plasma_composition"] = {
      "impurity": {
          "impurity_mode": "n_e_ratios",
          "species": {"Ne": 0.01},
      }
  }
  return config_dict


class RegisterEdgeModelTest(parameterized.TestCase):

  def test_registered_model_in_config(self):
    register_model.register_edge_model(DummyEdgeConfig)

    config_dict = _base_config_dict()
    config_dict["edge"] = {"model_name": "dummy_edge"}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    self.assertIsNotNone(torax_config.edge)
    edge_model = torax_config.edge.build_edge_model()
    self.assertIsInstance(edge_model, DummyEdgeModel)

  def test_update_flags_default_to_supported_boundary_conditions(self):
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config_dict = _base_config_dict()
    config_dict["edge"] = {"model_name": "dummy_te"}
    torax_config = config.ToraxConfig.from_dict(config_dict)
    assert torax_config.edge is not None
    runtime_params = torax_config.edge.build_runtime_params(t=0.0)
    self.assertTrue(runtime_params.update_T_e)
    self.assertFalse(runtime_params.update_T_i)
    self.assertFalse(runtime_params.update_n_e)
    self.assertFalse(runtime_params.update_n_impurities)

  def test_error_if_update_flag_set_for_unsupported_bc(self):
    """Test that an error is raised if an unsupported BC is updated."""
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config_dict = _base_config_dict()
    config_dict["edge"] = {
        "model_name": "dummy_te",
        "update_T_i": True,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"'update_T_i' cannot be True because 'dummy_te' does not"
        r" support the 'T_i' boundary condition\.",
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_missing_supported_boundary_conditions_raises_error(self):
    class BadEdgeConfig(base.EdgeModelConfig):
      model_name: Annotated[Literal["bad"], torax_pydantic.JAX_STATIC] = "bad"

      def build_edge_model(self) -> DummyEdgeModel:
        return DummyEdgeModel()

    with self.assertRaisesRegex(
        pydantic.ValidationError, "supported_boundary_conditions must be set."
    ):
      BadEdgeConfig.from_dict({})

  def _combined_sub_models_config_dict(self) -> dict[str, object]:
    register_model.register_edge_model(DummyEdgeConfig)
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config_dict = _base_config_dict()
    config_dict["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "full": {"model_name": "dummy_edge", "update_T_e": False},
            "te": {"model_name": "dummy_te"},
        },
    }
    return config_dict

  def test_combined_edge_builds_sub_models(self):
    config_dict = self._combined_sub_models_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    assert torax_config.edge is not None
    edge_model = torax_config.edge.build_edge_model()
    assert isinstance(edge_model, base.CombinedEdgeModel)
    self.assertIsInstance(edge_model.sub_models["full"], DummyEdgeModel)
    self.assertIsInstance(
        edge_model.sub_models["te"], DummyElectronTemperatureModel
    )
    self.assertEqual(edge_model.T_e_sub_model, "te")
    self.assertEqual(edge_model.T_i_sub_model, "full")
    self.assertEqual(edge_model.n_e_sub_model, "full")
    self.assertEqual(edge_model.n_impurities_sub_model, "full")

  def test_combined_edge_aggregates_sub_model_update_flags(self):
    config_dict = self._combined_sub_models_config_dict()
    torax_config = config.ToraxConfig.from_dict(config_dict)
    assert torax_config.edge is not None
    runtime_params = torax_config.edge.build_runtime_params(t=0.0)
    self.assertTrue(runtime_params.update_T_e)
    self.assertTrue(runtime_params.update_T_i)
    self.assertTrue(runtime_params.update_n_e)
    self.assertTrue(runtime_params.update_n_impurities)

  @parameterized.named_parameters(
      dict(
          testcase_name="constant_T_e_flags",
          flag_name="update_T_e",
          first_flag=True,
          second_flag=True,
      ),
      dict(
          testcase_name="disjoint_time_varying_T_e_flags",
          flag_name="update_T_e",
          first_flag={0.0: True, 1.0: False},
          second_flag={0.0: False, 1.0: True},
      ),
      dict(
          testcase_name="constant_n_impurities_flags",
          flag_name="update_n_impurities",
          first_flag=True,
          second_flag=True,
      ),
      dict(
          testcase_name="disjoint_time_varying_n_impurities_flags",
          flag_name="update_n_impurities",
          first_flag={0.0: True, 1.0: False},
          second_flag={0.0: False, 1.0: True},
      ),
  )
  def test_combined_edge_multiple_bc_providers_raises_error(
      self, flag_name, first_flag, second_flag
  ):
    register_model.register_edge_model(DummyEdgeConfig)

    base_sub_model = {
        "model_name": "dummy_edge",
        "update_T_e": False,
        "update_T_i": False,
        "update_n_e": False,
        "update_n_impurities": False,
    }
    config_dict = _base_config_dict()
    config_dict["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "first": {**base_sub_model, flag_name: first_flag},
            "second": {**base_sub_model, flag_name: second_flag},
        },
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        rf"'{flag_name}' may only be enabled on a single sub-model"
        r" across all time, but is enabled on \['first', 'second'\]\.",
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_combined_edge_top_level_update_flag_raises_error(self):
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config_dict = _base_config_dict()
    config_dict["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "te": {"model_name": "dummy_te"},
        },
        "update_T_e": True,
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"'update_T_e' cannot be set on a combined edge model\.",
    ):
      config.ToraxConfig.from_dict(config_dict)

  def test_combined_edge_empty_sub_models_raises_error(self):
    config_dict = _base_config_dict()
    config_dict["edge"] = {
        "model_name": "combined",
        "sub_models": {},
    }
    with self.assertRaises(pydantic.ValidationError):
      config.ToraxConfig.from_dict(config_dict)

  def test_dynamic_registration_updates_discriminator(self):
    register_model.register_edge_model(DummyEdgeConfig)

    config_dict = default_configs.get_default_config_dict()
    config_dict["edge"] = {"model_name": "invalid_edge_model"}
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"Input tag 'invalid_edge_model' found using 'model_name' does not"
        r" match any.*'dummy_edge'",
    ):
      config.ToraxConfig.from_dict(config_dict)


@dataclasses.dataclass(frozen=True, eq=False)
class DummyEdgeModel(base.EdgeModel):
  """Dummy edge model for testing purposes."""

  def __call__(
      self,
      edge_runtime_params: edge_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
      previous_edge_outputs: base.EdgeModelOutputs | None = None,
  ) -> base.EdgeModelOutputs:
    del edge_runtime_params, runtime_params, geo, core_profiles, core_sources
    del previous_edge_outputs
    return base.EdgeModelOutputs(
        T_i_right_bc=jnp.array(80.0),
        T_e_right_bc=jnp.array(120.0),
        n_e_right_bc=jnp.array(2.0e19),
        impurity_right_bc={},
    )


class DummyEdgeConfig(base.EdgeModelConfig):
  """Dummy edge model configuration for testing."""

  supported_boundary_conditions: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(
          T_e=True,
          T_i=True,
          n_e=True,
          n_impurities=True,
      )
  )
  model_name: Annotated[Literal["dummy_edge"], torax_pydantic.JAX_STATIC] = (
      "dummy_edge"
  )

  def build_edge_model(self) -> DummyEdgeModel:
    return DummyEdgeModel()


@dataclasses.dataclass(frozen=True, eq=False)
class DummyElectronTemperatureModel(DummyEdgeModel):
  """Dummy edge model that only computes the electron temperature BC."""


class DummyElectronTemperatureConfig(base.EdgeModelConfig):
  supported_boundary_conditions: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(T_e=True)
  )
  model_name: Annotated[Literal["dummy_te"], torax_pydantic.JAX_STATIC] = (
      "dummy_te"
  )

  def build_edge_model(self) -> DummyElectronTemperatureModel:
    return DummyElectronTemperatureModel()


if __name__ == "__main__":
  absltest.main()
