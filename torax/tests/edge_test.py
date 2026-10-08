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
import functools
from typing import Annotated, ClassVar, Literal

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import torax
from torax import edge
from torax._src.test_utils import default_configs

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _CustomEdgeRuntimeParams(edge.RuntimeParams):
  T_e_bc: float = 0.12
  T_i_bc: float = 0.08
  n_e_bc: float = 2.0e19
  ne_impurity_bc: float | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _FakeRuntimeParams:
  """Minimal stand-in for `torax.RuntimeParams` holding only `edge`."""

  edge: edge.RuntimeParams


@dataclasses.dataclass(frozen=True, eq=False)
class _CustomEdgeModel(edge.EdgeModel):
  """Custom edge model that returns configured boundary condition values."""

  computes_T_e: bool = False
  computes_T_i: bool = False
  computes_n_e: bool = False
  computes_impurities: bool = False

  @classmethod
  def from_supported_bcs(
      cls, supported_bcs: edge.SupportedBoundaryConditions
  ) -> '_CustomEdgeModel':
    return cls(
        computes_T_e=supported_bcs.T_e,
        computes_T_i=supported_bcs.T_i,
        computes_n_e=supported_bcs.n_e,
        computes_impurities=supported_bcs.impurity,
    )

  def __call__(
      self,
      runtime_params: torax.RuntimeParams,
      geo: torax.Geometry,
      core_profiles: torax.CoreProfiles,
      core_sources: torax.SourceProfiles,
      previous_edge_outputs: edge.EdgeModelOutputs | None = None,
  ) -> edge.EdgeModelOutputs:
    del geo, core_profiles, core_sources, previous_edge_outputs
    assert isinstance(runtime_params.edge, _CustomEdgeRuntimeParams)
    edge_runtime_params = runtime_params.edge
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
        update_impurity=base_params.update_impurity,
        T_e_bc=self.T_e_bc,
        T_i_bc=self.T_i_bc,
        n_e_bc=self.n_e_bc,
        ne_impurity_bc=self.ne_impurity_bc,
    )

  def build_edge_model(self) -> _CustomEdgeModel:
    return _CustomEdgeModel.from_supported_bcs(self.supported_bcs)


class _CustomEdgeModelConfig(_BaseCustomEdgeConfig):
  supported_bcs: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(
          T_e=True,
          T_i=True,
          n_e=True,
          impurity=True,
      )
  )
  model_name: Annotated[Literal['custom_edge'], torax.JAX_STATIC] = (
      'custom_edge'
  )


class _CustomTeConfig(_BaseCustomEdgeConfig):
  supported_bcs: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(T_e=True)
  )
  model_name: Annotated[Literal['custom_te'], torax.JAX_STATIC] = 'custom_te'
  T_e_bc: float = 0.15


class _CustomTiConfig(_BaseCustomEdgeConfig):
  supported_bcs: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(T_i=True)
  )
  model_name: Annotated[Literal['custom_ti'], torax.JAX_STATIC] = 'custom_ti'
  T_i_bc: float = 0.25


class _CustomNeConfig(_BaseCustomEdgeConfig):
  supported_bcs: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(n_e=True)
  )
  model_name: Annotated[Literal['custom_ne'], torax.JAX_STATIC] = 'custom_ne'
  n_e_bc: float = 2.5e19


class _CustomImpuritiesConfig(_BaseCustomEdgeConfig):
  supported_bcs: ClassVar[edge.SupportedBoundaryConditions] = (
      edge.SupportedBoundaryConditions(impurity=True)
  )
  model_name: Annotated[Literal['custom_imp'], torax.JAX_STATIC] = 'custom_imp'
  ne_impurity_bc: float | None = 0.03


edge.register_edge_model(_CustomEdgeModelConfig)
edge.register_edge_model(_CustomTeConfig)
edge.register_edge_model(_CustomTiConfig)
edge.register_edge_model(_CustomNeConfig)
edge.register_edge_model(_CustomImpuritiesConfig)


def _make_combined_te_model() -> edge.CombinedEdgeModel:
  return edge.CombinedEdgeModel(
      sub_models={'te': _CustomEdgeModel(computes_T_e=True)}
  )


def _make_combined_te_runtime_params(
    update: bool, T_e_bc: float
) -> _FakeRuntimeParams:
  sub_params = _CustomEdgeRuntimeParams(
      update_T_e=jnp.asarray(update),
      update_T_i=jnp.asarray(False),
      update_n_e=jnp.asarray(False),
      update_impurity=jnp.asarray(False),
      T_e_bc=T_e_bc,
  )
  return _FakeRuntimeParams(
      edge=edge.CombinedRuntimeParams(
          update_T_e=jnp.asarray(update),
          update_T_i=jnp.asarray(False),
          update_n_e=jnp.asarray(False),
          update_impurity=jnp.asarray(False),
          sub_models={'te': sub_params},
      )
  )


def _evaluate_edge_model(
    model: edge.CombinedEdgeModel,
    runtime_params: _FakeRuntimeParams,
    previous_edge_outputs: edge.EdgeModelOutputs | None = None,
) -> edge.CombinedEdgeOutputs:
  return model(
      runtime_params,  # pyrefly: ignore[bad-argument-type]
      None,  # pyrefly: ignore[bad-argument-type]
      None,  # pyrefly: ignore[bad-argument-type]
      None,  # pyrefly: ignore[bad-argument-type]
      previous_edge_outputs=previous_edge_outputs,
  )


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
    state_history = torax.run_simulation(torax_config)
    data_tree = state_history.simulation_output_to_xr()
    self.assertIn('T_e_right_bc', data_tree.edge.dataset.data_vars)

  def test_combined_edge_with_partial_update_flags_runs(self):
    config = default_configs.get_default_config_dict()
    config['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    config['edge'] = {
        'model_name': 'combined',
        'sub_models': {
            'temps': {
                'model_name': 'custom_edge',
                'T_e_bc': 0.18,
                'T_i_bc': 0.22,
                'n_e_bc': 1.0e19,
                'update_n_e': False,
                'update_impurity': False,
            },
            'density': {
                'model_name': 'custom_ne',
                'n_e_bc': 3.2e19,
            },
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    state_history = torax.run_simulation(torax_config)
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)
    final_core_profiles = state_history.core_profiles[-1]
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.T_e.right_face_constraint, 0.18, rtol=1e-5
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.T_i.right_face_constraint, 0.22, rtol=1e-5
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.n_e.right_face_constraint, 3.2e19, rtol=1e-5
    )

  def test_combined_edge_switches_bc_provider_in_time(self):
    config = default_configs.get_default_config_dict()
    config['geometry'] = {
        'geometry_type': 'chease',
        'geometry_file': 'iterhybrid.mat2cols',
    }
    t_final = 5.0
    config['numerics']['t_final'] = t_final
    t_switch = 0.5 * t_final
    config['edge'] = {
        'model_name': 'combined',
        'sub_models': {
            'early': {
                'model_name': 'custom_te',
                'T_e_bc': 0.15,
                'update_T_e': {0.0: True, t_switch: False},
            },
            'late': {
                'model_name': 'custom_edge',
                'T_e_bc': 0.3,
                'update_T_e': {0.0: False, t_switch: True},
                'update_T_i': False,
                'update_n_e': False,
                'update_impurity': False,
            },
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    state_history = torax.run_simulation(torax_config)
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)
    data_tree = state_history.simulation_output_to_xr()
    times = data_tree.edge.T_e_right_bc.time.values
    T_e_right_bc = data_tree.edge.T_e_right_bc.values
    # Edge outputs stored at t + dt are computed with runtime params at t, so
    # compare each step's output against the start time of that step.
    eval_times = np.concatenate([times[:1], times[:-1]])
    np.testing.assert_allclose(T_e_right_bc[eval_times < t_switch], 0.15)
    np.testing.assert_allclose(T_e_right_bc[eval_times > t_switch], 0.3)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        state_history.core_profiles[-1].T_e.right_face_constraint,
        0.3,
        rtol=1e-5,
    )

  def test_combined_edge_evaluates_sub_models_without_previous_outputs(self):
    model = _make_combined_te_model()
    outputs = _evaluate_edge_model(
        model, _make_combined_te_runtime_params(update=False, T_e_bc=0.1)
    )
    T_e_right_bc = outputs.sub_models['te'].T_e_right_bc
    assert T_e_right_bc is not None
    np.testing.assert_allclose(T_e_right_bc, 0.1)

  def test_combined_edge_evaluates_active_sub_models(self):
    model = _make_combined_te_model()
    previous = _evaluate_edge_model(
        model, _make_combined_te_runtime_params(update=True, T_e_bc=0.1)
    )
    step = jax.jit(functools.partial(_evaluate_edge_model, model))
    outputs = step(
        _make_combined_te_runtime_params(update=True, T_e_bc=0.2), previous
    )
    np.testing.assert_allclose(outputs.sub_models['te'].T_e_right_bc, 0.2)
    np.testing.assert_allclose(outputs.T_e_right_bc, 0.2)

  def test_combined_edge_skips_inactive_sub_models_and_carries_outputs(self):
    model = _make_combined_te_model()
    previous = _evaluate_edge_model(
        model, _make_combined_te_runtime_params(update=True, T_e_bc=0.2)
    )
    step = jax.jit(functools.partial(_evaluate_edge_model, model))
    outputs = step(
        _make_combined_te_runtime_params(update=False, T_e_bc=0.3), previous
    )
    np.testing.assert_allclose(outputs.sub_models['te'].T_e_right_bc, 0.2)


class CombinedEdgeSingleBcSubModelsTest(absltest.TestCase):
  """Runs one simulation with a separate edge sub-model for each BC."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
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
    config['edge'] = {
        'sub_models': {
            'te': {
                'model_name': 'custom_te',
                'T_e_bc': 0.15,
            },
            'ti': {
                'model_name': 'custom_ti',
                'T_i_bc': 0.25,
            },
            'ne': {
                'model_name': 'custom_ne',
                'n_e_bc': 2.5e19,
            },
            'imp': {
                'model_name': 'custom_imp',
                'ne_impurity_bc': 0.03,
            },
        },
    }
    torax_config = torax.ToraxConfig.from_dict(config)
    cls.state_history = torax.run_simulation(torax_config)
    cls.data_tree = cls.state_history.simulation_output_to_xr()

  def test_simulation_completes_without_error(self):
    self.assertEqual(self.state_history.sim_error, torax.SimError.NO_ERROR)

  def test_core_boundary_conditions_set_from_sub_models(self):
    final_core_profiles = self.state_history.core_profiles[-1]
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.T_e.right_face_constraint, 0.15, rtol=1e-5
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.T_i.right_face_constraint, 0.25, rtol=1e-5
    )
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        final_core_profiles.n_e.right_face_constraint, 2.5e19, rtol=1e-5
    )

  def test_merged_edge_outputs(self):
    edge_tree = self.data_tree.edge
    np.testing.assert_allclose(edge_tree.T_e_right_bc.values[-1], 0.15)
    np.testing.assert_allclose(edge_tree.T_i_right_bc.values[-1], 0.25)
    np.testing.assert_allclose(edge_tree.n_e_right_bc.values[-1], 2.5e19)
    np.testing.assert_allclose(
        edge_tree.impurity_right_bc.sel(impurity='Ne').values[-1],
        0.03,
    )

  def test_sub_model_outputs_are_child_nodes(self):
    edge_tree = self.data_tree.edge
    self.assertCountEqual(edge_tree.children, ['te', 'ti', 'ne', 'imp'])
    np.testing.assert_allclose(edge_tree['te'].T_e_right_bc.values[-1], 0.15)
    np.testing.assert_allclose(edge_tree['ti'].T_i_right_bc.values[-1], 0.25)


if __name__ == '__main__':
  absltest.main()
