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

"""Unit tests for the unified physics pipeline."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src import model_state as model_state_lib
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.orchestration import explicit_model_outputs as explicit_model_outputs_lib
from torax._src.orchestration import physics_pipeline
from torax._src.orchestration import run_simulation
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class PhysicsPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    (
        self.sim_state,
        self.post_processed_outputs,
        self.step_fn,
    ) = run_simulation.prepare_simulation(self.torax_config)
    self.runtime_params = self.step_fn.runtime_params_provider(
        t=self.sim_state.t
    )
    self.geo = self.step_fn.geometry_provider(self.sim_state.t)
    self.models = self.step_fn.solver.models
    self.solver_state = solver_state_lib.SolverState.from_core_profiles(
        self.sim_state.core_profiles
    )
    source_states = self.models.source_models.initial_state(self.runtime_params)
    explicit_source_outputs, _ = source_profile_builders.build_basic_sources(
        source_models=self.models.source_models,
        source_states=source_states,
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.sim_state.core_profiles,
        explicit=True,
    )
    self.explicit_model_outputs = (
        explicit_model_outputs_lib.ExplicitModelOutputs(
            sources=explicit_source_outputs,
            pedestal=self.sim_state.pedestal_output.structure,
        )
    )

  def test_model_state_updated(self):
    _, model_state = physics_pipeline.evaluate_step_physics(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        models=self.models,
        explicit_model_outputs=self.explicit_model_outputs,
        model_state=self.sim_state.model_state,
    )
    self.assertIsInstance(model_state, model_state_lib.ModelState)

  def test_v_loop_lcfs_propagation(self):
    # Default without explicit v_loop_lcfs: flux v_loop_lcfs is 0.0 and
    # psidot right face is outer cell.
    outputs_no_v_loop, _ = physics_pipeline.evaluate_step_physics(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        models=self.models,
        explicit_model_outputs=self.explicit_model_outputs,
        model_state=self.sim_state.model_state,
        v_loop_lcfs=None,
    )
    np.testing.assert_allclose(
        outputs_no_v_loop.core_profiles.flux.v_loop_lcfs,
        jnp.asarray(0.0, dtype=jax_utils.get_dtype()),
    )
    assert (
        outputs_no_v_loop.core_profiles.psidot.right_face_constraint is not None
    )
    np.testing.assert_allclose(
        outputs_no_v_loop.core_profiles.psidot.right_face_constraint,
        outputs_no_v_loop.core_profiles.psidot.value[-1],
    )

    # With explicit v_loop_lcfs: both flux and psidot reflect the value.
    custom_v_loop = jnp.asarray(3.25, dtype=jax_utils.get_dtype())
    outputs_with_v_loop, _ = physics_pipeline.evaluate_step_physics(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        models=self.models,
        explicit_model_outputs=self.explicit_model_outputs,
        model_state=self.sim_state.model_state,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(
        outputs_with_v_loop.core_profiles.flux.v_loop_lcfs,
        custom_v_loop,
    )
    assert (
        outputs_with_v_loop.core_profiles.psidot.right_face_constraint
        is not None
    )
    np.testing.assert_allclose(
        outputs_with_v_loop.core_profiles.psidot.right_face_constraint,
        custom_v_loop,
    )

  def test_explicit_source_preservation(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['geometry'] = {'geometry_type': 'circular', 'n_rho': 4}
    config_dict['numerics'] = {
        'evolve_current': True,
        'enable_fast_ions': True,
    }
    config_dict['plasma_composition'] = {
        'main_ion': {'D': 0.5, 'T': 0.5},
        'impurity': {
            'impurity_mode': 'n_e_ratios',
            'species': {'He3': 0.03},
        },
    }
    config_dict['sources'] = {
        'icrh': {
            'model_name': 'scaled_profile',
            'minority_species': 'He3',
            'is_explicit': True,
        },
    }
    config_dict['profile_conditions']['fast_ions'] = [{
        'source': 'icrh',
        'species': 'He3',
        'n': {0.0: {0.0: 1e19, 1.0: 1e19}},
        'T': {0.0: {0.0: 100.0, 1.0: 100.0}},
        'n_right_bc': 0.0,
        'T_right_bc': 0.0,
    }]
    torax_config = config.ToraxConfig.from_dict(config_dict)
    (
        sim_state,
        _,
        step_fn,
    ) = run_simulation.prepare_simulation(torax_config)
    runtime_params = step_fn.runtime_params_provider(t=sim_state.t)
    geo = step_fn.geometry_provider(sim_state.t)
    models = step_fn.solver.models
    solver_state = solver_state_lib.SolverState.from_core_profiles(
        sim_state.core_profiles
    )

    source_states = models.source_models.initial_state(runtime_params)
    explicit_source_outputs, _ = source_profile_builders.build_basic_sources(
        source_models=models.source_models,
        source_states=source_states,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=sim_state.core_profiles,
        explicit=True,
    )

    # Create synthetic explicit source output with a distinctive value.
    distinctive_value = 7.77e19
    modified_fast_ion = dataclasses.replace(
        sim_state.core_profiles.fast_ions[0],
        n=dataclasses.replace(
            sim_state.core_profiles.fast_ions[0].n,
            value=jnp.full_like(
                sim_state.core_profiles.fast_ions[0].n.value,
                distinctive_value,
            ),
        ),
    )
    synthetic_sources = dict(explicit_source_outputs)
    synthetic_sources['icrh'] = dataclasses.replace(
        synthetic_sources['icrh'],
        fast_ions=(modified_fast_ion,),
    )
    explicit_outputs = explicit_model_outputs_lib.ExplicitModelOutputs(
        sources=synthetic_sources,
        pedestal=sim_state.pedestal_output.structure,
    )

    result, _ = physics_pipeline.evaluate_step_physics(
        runtime_params=runtime_params,
        geo=geo,
        solver_state=solver_state,
        models=models,
        explicit_model_outputs=explicit_outputs,
        model_state=sim_state.model_state,
    )

    # Because icrh is explicit, its fast ion population must be preserved
    # from explicit_outputs.
    self.assertIn('icrh', result.core_sources.fast_ions)
    np.testing.assert_allclose(
        result.core_sources.fast_ions['icrh'][0].n.value,
        distinctive_value,
    )


if __name__ == '__main__':
  absltest.main()
