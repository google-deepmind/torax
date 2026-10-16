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
from typing import Annotated, ClassVar, Literal
from unittest import mock

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torax
from torax import edge
from torax._src.config import build_runtime_params
from torax._src.edge import base as edge_base
from torax._src.test_utils import default_configs

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _CustomEdgeRuntimeParams(edge.RuntimeParams):
  T_e_bc: float = 0.12
  T_i_bc: float = 0.08
  n_e_bc: float = 2.0e19
  ne_impurity_bc: float | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class _CustomEdgeModel(edge.EdgeModel):
  """Custom edge model that returns configured boundary condition values."""

  computes_T_e: bool = False
  computes_T_i: bool = False
  computes_n_e: bool = False
  computes_impurities: bool = False

  @classmethod
  def from_supported_boundary_conditions(
      cls,
      supported_boundary_conditions: edge.SupportedBoundaryConditions,
  ) -> '_CustomEdgeModel':
    return cls(
        computes_T_e=supported_boundary_conditions.T_e,
        computes_T_i=supported_boundary_conditions.T_i,
        computes_n_e=supported_boundary_conditions.n_e,
        computes_impurities=supported_boundary_conditions.n_impurities,
    )

  def __call__(
      self,
      edge_runtime_params: edge.RuntimeParams,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      core_sources: torax.SourceProfiles,
      previous_edge_outputs: edge.EdgeModelOutputs | None = None,
  ) -> edge.EdgeModelOutputs:
    del runtime_params, geo, core_profiles, core_sources, previous_edge_outputs
    assert isinstance(edge_runtime_params, _CustomEdgeRuntimeParams)
    impurity_right_bc = {}
    if (
        self.computes_impurities
        and edge_runtime_params.ne_impurity_bc is not None
    ):
      impurity_right_bc = {
          'Ne': jnp.asarray(edge_runtime_params.ne_impurity_bc)
      }
    return edge.EdgeModelOutputs(
        T_e_right_bc=(
            jnp.asarray(edge_runtime_params.T_e_bc)
            if self.computes_T_e
            else None
        ),
        T_i_right_bc=(
            jnp.asarray(edge_runtime_params.T_i_bc)
            if self.computes_T_i
            else None
        ),
        n_e_right_bc=(
            jnp.asarray(edge_runtime_params.n_e_bc)
            if self.computes_n_e
            else None
        ),
        impurity_right_bc=impurity_right_bc,
    )


class _BaseCustomEdgeConfig(edge.EdgeModelConfig):
  """Base custom edge model pydantic config."""

  T_e_bc: float = 0.12
  T_i_bc: float = 0.08
  n_e_bc: float = 2.0e19
  ne_impurity_bc: float | None = None

  def build_runtime_params(self, t) -> _CustomEdgeRuntimeParams:
    base_params = super().build_runtime_params(t)
    return _CustomEdgeRuntimeParams(
        update_T_e=base_params.update_T_e,
        update_T_i=base_params.update_T_i,
        update_n_e=base_params.update_n_e,
        update_n_impurities=base_params.update_n_impurities,
        T_e_bc=self.T_e_bc,
        T_i_bc=self.T_i_bc,
        n_e_bc=self.n_e_bc,
        ne_impurity_bc=self.ne_impurity_bc,
    )

  def build_edge_model(self) -> _CustomEdgeModel:
    return _CustomEdgeModel.from_supported_boundary_conditions(
        self.supported_boundary_conditions
    )


class _CustomEdgeModelConfig(_BaseCustomEdgeConfig):
  supported_boundary_conditions: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(
          T_e=True,
          T_i=True,
          n_e=True,
          n_impurities=True,
      )
  )
  model_name: Annotated[Literal['custom_edge'], torax.JAX_STATIC] = (
      'custom_edge'
  )


class _CustomTeConfig(_BaseCustomEdgeConfig):
  supported_boundary_conditions: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(T_e=True)
  )
  model_name: Annotated[Literal['custom_te'], torax.JAX_STATIC] = 'custom_te'
  T_e_bc: float = 0.15


edge.register_edge_model(_CustomEdgeModelConfig)
edge.register_edge_model(_CustomTeConfig)


def _evaluate_combined_edge(
    torax_config: torax.ToraxConfig, t: float = 0.0
) -> edge_base.CombinedEdgeOutputs:
  assert torax_config.edge is not None
  edge_model = torax_config.edge.build_edge_model()
  assert isinstance(edge_model, edge_base.CombinedEdgeModel)
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(t=t)
  assert runtime_params.edge is not None
  outputs = edge_model(
      runtime_params.edge,
      runtime_params,
      mock.create_autospec(torax.Geometry, instance=True),
      mock.create_autospec(torax.CoreProfiles, instance=True),
      mock.create_autospec(torax.SourceProfiles, instance=True),
  )
  assert isinstance(outputs, edge_base.CombinedEdgeOutputs)
  return outputs


def _base_config() -> dict[str, object]:
  config = default_configs.get_default_config_dict()
  config['geometry'] = {
      'geometry_type': 'chease',
      'geometry_file': 'iterhybrid.mat2cols',
  }
  config['plasma_composition'] = {
      'impurity': {
          'impurity_mode': 'n_e_ratios',
          'species': {'Ne': 0.01},
      }
  }
  return config


class EdgeTest(absltest.TestCase):

  def test_custom_edge_model_runs(self):
    """Tests that the custom edge model can be used in a simulation."""
    config = _base_config()
    config['edge'] = {
        'model_name': 'custom_edge',
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    state_history = torax.run_simulation(torax_config)
    data_tree = state_history.simulation_output_to_xr()
    self.assertIn('T_e_right_bc', data_tree.edge.dataset.data_vars)

  def test_combined_edge_merges_bcs_from_sub_models(self):
    config = _base_config()
    config['edge'] = {
        'model_name': 'combined',
        'sub_models': {
            'te': {
                'model_name': 'custom_te',
                'T_e_bc': 0.15,
            },
            'other': {
                'model_name': 'custom_edge',
                'T_e_bc': 0.99,
                'T_i_bc': 0.25,
                'n_e_bc': 2.5e19,
                'ne_impurity_bc': 0.03,
                'update_T_e': False,
            },
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    outputs = _evaluate_combined_edge(torax_config)
    assert outputs.T_e_right_bc is not None
    assert outputs.T_i_right_bc is not None
    assert outputs.n_e_right_bc is not None
    np.testing.assert_allclose(outputs.T_e_right_bc, 0.15)
    np.testing.assert_allclose(outputs.T_i_right_bc, 0.25)
    np.testing.assert_allclose(outputs.n_e_right_bc, 2.5e19)
    np.testing.assert_allclose(outputs.impurity_right_bc['Ne'], 0.03)
    self.assertCountEqual(outputs.sub_models.keys(), ['te', 'other'])
    other_T_e = outputs.sub_models['other'].T_e_right_bc
    assert other_T_e is not None
    np.testing.assert_allclose(other_T_e, 0.99)


if __name__ == '__main__':
  absltest.main()
