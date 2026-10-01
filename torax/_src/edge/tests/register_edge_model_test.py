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
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.geometry import geometry
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic


def _base_config() -> dict[str, object]:
  config = default_configs.get_default_config_dict()
  # Non-circular geometry is required because ToraxConfig validates that edge
  # models are not used with CircularGeometry.
  config["geometry"] = {
      "geometry_type": "chease",
      "geometry_file": "iterhybrid.mat2cols",
  }
  return config


class RegisterEdgeModelTest(parameterized.TestCase):

  def test_registered_model_in_config(self):
    register_model.register_edge_model(DummyEdgeConfig)

    config = _base_config()
    config["edge"] = {"model_name": "dummy_edge"}
    torax_config = model_config.ToraxConfig.from_dict(config)
    self.assertIsNotNone(torax_config.edge)
    edge_model = torax_config.edge.build_edge_model()
    self.assertIsInstance(edge_model, DummyEdgeModel)

  def test_update_flags_default_to_supported_bcs(self):
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = _base_config()
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

    config = _base_config()
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

  def _combined_single_bc_sub_models_config(self) -> dict[str, object]:
    register_model.register_edge_model(DummyElectronTemperatureConfig)
    register_model.register_edge_model(DummyIonTemperatureConfig)
    register_model.register_edge_model(DummyElectronDensityConfig)
    register_model.register_edge_model(DummyImpuritiesConfig)

    config = _base_config()
    config["edge"] = {
        "sub_models": {
            "te": {"model_name": "dummy_te"},
            "ti": {"model_name": "dummy_ti"},
            "ne": {"model_name": "dummy_ne"},
            "imp": {"model_name": "dummy_imp"},
        }
    }
    return config

  def test_combined_edge_builds_sub_models(self):
    config = self._combined_single_bc_sub_models_config()
    torax_config = model_config.ToraxConfig.from_dict(config)
    assert torax_config.edge is not None
    edge_model = torax_config.edge.build_edge_model()
    assert isinstance(edge_model, base.CombinedEdgeModel)
    self.assertIsInstance(
        edge_model.sub_models["te"], DummyElectronTemperatureModel
    )
    self.assertIsInstance(edge_model.sub_models["ti"], DummyIonTemperatureModel)
    self.assertIsInstance(
        edge_model.sub_models["ne"], DummyElectronDensityModel
    )
    self.assertIsInstance(edge_model.sub_models["imp"], DummyImpuritiesModel)

  def test_combined_edge_aggregates_sub_model_update_flags(self):
    config = self._combined_single_bc_sub_models_config()
    torax_config = model_config.ToraxConfig.from_dict(config)
    assert torax_config.edge is not None
    runtime_params = torax_config.edge.build_runtime_params(t=0.0)
    self.assertTrue(runtime_params.update_T_e)
    self.assertTrue(runtime_params.update_T_i)
    self.assertTrue(runtime_params.update_n_e)
    self.assertTrue(runtime_params.update_impurity)

  @parameterized.named_parameters(
      dict(
          testcase_name="constant_flags",
          full_flag=True,
          te_flag=True,
          expected_time="0\\.0",
      ),
      dict(
          testcase_name="time_varying_flags",
          full_flag={0.0: False, 1.0: True},
          te_flag={0.0: True, 2.0: False},
          expected_time="2\\.0",
      ),
  )
  def test_combined_edge_overlapping_update_flags_raises_error(
      self, full_flag, te_flag, expected_time
  ):
    register_model.register_edge_model(DummyEdgeConfig)
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = _base_config()
    config["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "full": {
                "model_name": "dummy_edge",
                "update_T_e": full_flag,
            },
            "te": {
                "model_name": "dummy_te",
                "update_T_e": te_flag,
            },
        },
    }
    with self.assertRaisesRegex(
        pydantic.ValidationError,
        r"'update_T_e' is True for multiple sub-models"
        rf" \['full', 'te'\] at t={expected_time}\.",
    ):
      model_config.ToraxConfig.from_dict(config)

  @parameterized.named_parameters(
      dict(
          testcase_name="constant_flags",
          full_flag=False,
          te_flag=True,
      ),
      dict(
          testcase_name="time_varying_flags",
          full_flag={0.0: False, 1.0: True},
          te_flag={0.0: True, 1.0: False},
      ),
  )
  def test_combined_edge_disjoint_update_flags_succeeds(
      self, full_flag, te_flag
  ):
    register_model.register_edge_model(DummyEdgeConfig)
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = _base_config()
    config["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "full": {
                "model_name": "dummy_edge",
                "update_T_e": full_flag,
            },
            "te": {
                "model_name": "dummy_te",
                "update_T_e": te_flag,
            },
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    assert torax_config.edge is not None
    edge_model = torax_config.edge.build_edge_model()
    self.assertIsInstance(edge_model, base.CombinedEdgeModel)

  def test_combined_edge_bc_provider_switches_between_sub_models_in_time(self):
    register_model.register_edge_model(DummyEdgeConfig)
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = _base_config()
    config["edge"] = {
        "model_name": "combined",
        "sub_models": {
            "full": {
                "model_name": "dummy_edge",
                "update_T_e": {0.0: False, 1.0: True},
            },
            "te": {
                "model_name": "dummy_te",
                "update_T_e": {0.0: True, 1.0: False},
            },
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    assert torax_config.edge is not None
    params_before = torax_config.edge.build_runtime_params(t=0.5)
    params_after = torax_config.edge.build_runtime_params(t=1.5)
    assert isinstance(params_before, edge_runtime_params.CombinedRuntimeParams)
    assert isinstance(params_after, edge_runtime_params.CombinedRuntimeParams)
    self.assertTrue(params_before.update_T_e)
    self.assertTrue(params_after.update_T_e)
    self.assertTrue(params_before.sub_models["te"].update_T_e)
    self.assertFalse(params_before.sub_models["full"].update_T_e)
    self.assertFalse(params_after.sub_models["te"].update_T_e)
    self.assertTrue(params_after.sub_models["full"].update_T_e)

  def test_combined_edge_top_level_update_flag_raises_error(self):
    register_model.register_edge_model(DummyElectronTemperatureConfig)

    config = _base_config()
    config["edge"] = {
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

  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(
          T_e=True,
          T_i=True,
          n_e=True,
          impurity=True,
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
  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(T_e=True)
  )
  model_name: Annotated[Literal["dummy_te"], torax_pydantic.JAX_STATIC] = (
      "dummy_te"
  )

  def build_edge_model(self) -> DummyElectronTemperatureModel:
    return DummyElectronTemperatureModel()


@dataclasses.dataclass(frozen=True, eq=False)
class DummyIonTemperatureModel(DummyEdgeModel):
  """Dummy edge model that only computes the ion temperature BC."""


class DummyIonTemperatureConfig(base.EdgeModelConfig):
  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(T_i=True)
  )
  model_name: Annotated[Literal["dummy_ti"], torax_pydantic.JAX_STATIC] = (
      "dummy_ti"
  )

  def build_edge_model(self) -> DummyIonTemperatureModel:
    return DummyIonTemperatureModel()


@dataclasses.dataclass(frozen=True, eq=False)
class DummyElectronDensityModel(DummyEdgeModel):
  """Dummy edge model that only computes the electron density BC."""


class DummyElectronDensityConfig(base.EdgeModelConfig):
  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(n_e=True)
  )
  model_name: Annotated[Literal["dummy_ne"], torax_pydantic.JAX_STATIC] = (
      "dummy_ne"
  )

  def build_edge_model(self) -> DummyElectronDensityModel:
    return DummyElectronDensityModel()


@dataclasses.dataclass(frozen=True, eq=False)
class DummyImpuritiesModel(DummyEdgeModel):
  """Dummy edge model that only computes the impurity BCs."""


class DummyImpuritiesConfig(base.EdgeModelConfig):
  supported_bcs: ClassVar[base.SupportedBoundaryConditions] = (
      base.SupportedBoundaryConditions(impurity=True)
  )
  model_name: Annotated[Literal["dummy_imp"], torax_pydantic.JAX_STATIC] = (
      "dummy_imp"
  )

  def build_edge_model(self) -> DummyImpuritiesModel:
    return DummyImpuritiesModel()


if __name__ == "__main__":
  absltest.main()
