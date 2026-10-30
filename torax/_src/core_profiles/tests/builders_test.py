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

"""Tests for core_profiles builders."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import jax_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import builders
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.neoclassical import neoclassical_intermediates as neoclassical_intermediates_lib
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_profile_builders
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config


class BuildersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    self.core_profiles_t0, self.runtime_params, self.geo = (
        core_profile_helpers.make_initial_inputs(self.torax_config)
    )
    self.solver_state = solver_state_lib.SolverState.from_core_profiles(
        self.core_profiles_t0
    )

  def test_build_base_core_profiles_structure_and_values(self):
    result = builders.build_base_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
    )

    self.assertIsInstance(result, state.BaseCoreProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )
    np.testing.assert_allclose(
        result.kinetic.T_e.value,
        self.solver_state.T_e.value,
    )
    np.testing.assert_allclose(
        result.kinetic.n_e.value,
        self.solver_state.n_e.value,
    )
    np.testing.assert_allclose(
        result.flux.psi.value,
        self.solver_state.psi.value,
    )
    np.testing.assert_allclose(
        result.toroidal_angular_velocity.value,
        self.solver_state.toroidal_angular_velocity.value,
    )
    np.testing.assert_allclose(
        result.flux.v_loop_lcfs,
        jnp.asarray(0.0, dtype=jax_utils.get_dtype()),
    )

    # Check that safety factor and total current match direct calculations.
    expected_q_face = psi_calculations.calc_q_face(
        self.geo, self.solver_state.psi
    )
    np.testing.assert_allclose(result.flux.q_face, expected_q_face)
    self.assertTrue(result.quasineutrality_satisfied())

  def test_build_base_core_profiles_custom_v_loop_lcfs(self):
    custom_v_loop = jnp.asarray(1.75, dtype=jax_utils.get_dtype())
    result = builders.build_base_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(result.flux.v_loop_lcfs, custom_v_loop)

  def test_build_base_core_profiles_jittable(self):
    @jax.jit
    def jitted_builder(solver_state):
      return builders.build_base_core_profiles(
          runtime_params=self.runtime_params,
          geo=self.geo,
          solver_state=solver_state,
      )

    result = jitted_builder(self.solver_state)
    self.assertIsInstance(result, state.BaseCoreProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )

  def test_build_core_profiles_structure_and_values(self):
    models = self.torax_config.build_models()
    source_states = models.source_models.initial_state(self.runtime_params)
    explicit_source_outputs, _ = source_profile_builders.build_basic_sources(
        source_models=models.source_models,
        source_states=source_states,
        runtime_params=self.runtime_params,
        geo=self.geo,
        core_profiles=self.core_profiles_t0,
        explicit=True,
    )
    (
        result,
        basic_source_outputs,
        updated_source_states,
        bootstrap_current,
        neoclassical_intermediates,
    ) = builders.build_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        source_models=models.source_models,
        neoclassical_model=models.neoclassical_model,
        source_states=source_states,
        explicit=False,
        explicit_source_outputs=explicit_source_outputs,
    )
    self.assertIsInstance(result, state.CoreProfiles)
    self.assertIsInstance(basic_source_outputs, dict)
    self.assertIsInstance(updated_source_states, dict)
    self.assertIsInstance(
        bootstrap_current, bootstrap_current_base.BootstrapCurrent
    )
    self.assertIsInstance(
        neoclassical_intermediates,
        neoclassical_intermediates_lib.NeoclassicalIntermediates,
    )
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )
    np.testing.assert_allclose(
        result.kinetic.T_e.value,
        self.solver_state.T_e.value,
    )
    np.testing.assert_allclose(
        result.kinetic.n_e.value,
        self.solver_state.n_e.value,
    )
    np.testing.assert_allclose(
        result.flux.psi.value,
        self.solver_state.psi.value,
    )

  def test_build_core_profiles_custom_v_loop_lcfs(self):
    custom_v_loop = jnp.asarray(2.5, dtype=jax_utils.get_dtype())
    models = self.torax_config.build_models()
    source_states = models.source_models.initial_state(self.runtime_params)
    (
        result,
        _,
        _,
        _,
        _,
    ) = builders.build_core_profiles(
        runtime_params=self.runtime_params,
        geo=self.geo,
        solver_state=self.solver_state,
        source_models=models.source_models,
        neoclassical_model=models.neoclassical_model,
        source_states=source_states,
        explicit=True,
        v_loop_lcfs=custom_v_loop,
    )
    np.testing.assert_allclose(result.flux.v_loop_lcfs, custom_v_loop)
    assert result.psidot.right_face_constraint is not None
    np.testing.assert_allclose(
        result.psidot.right_face_constraint,
        custom_v_loop,
    )

  def test_build_core_profiles_updates_fast_ions_before_neoclassical(self):
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
            'is_explicit': False,
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
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    models = torax_config.build_models()
    core_profiles_t0, runtime_params_t0, geo = (
        core_profile_helpers.make_initial_inputs(torax_config)
    )
    runtime_params_t1 = provider(t=1.0)
    source_states = models.source_models.initial_state(runtime_params_t0)
    explicit_source_outputs, _ = source_profile_builders.build_basic_sources(
        source_models=models.source_models,
        source_states=source_states,
        runtime_params=runtime_params_t0,
        geo=geo,
        core_profiles=core_profiles_t0,
        explicit=True,
    )
    solver_state = solver_state_lib.SolverState.from_core_profiles(
        core_profiles_t0
    )

    observed_calls = []
    original_calc = (
        models.neoclassical_model.calculate_conductivity_current_and_rotation
    )

    def spy_calc(runtime_params, geo, core_profiles, **kwargs):
      observed_calls.append((core_profiles, kwargs.get('fast_ions', ())))
      return original_calc(runtime_params, geo, core_profiles, **kwargs)

    with jax.disable_jit():
      with mock.patch.object(
          type(models.neoclassical_model),
          'calculate_conductivity_current_and_rotation',
          side_effect=spy_calc,
      ):
        (
            result,
            _,
            _,
            _,
            _,
        ) = builders.build_core_profiles(
            runtime_params=runtime_params_t1,
            geo=geo,
            solver_state=solver_state,
            source_models=models.source_models,
            neoclassical_model=models.neoclassical_model,
            source_states=source_states,
            explicit=False,
            explicit_source_outputs=explicit_source_outputs,
        )

    self.assertLen(observed_calls, 1)
    passed_core_profiles, passed_fast_ions = observed_calls[0]
    self.assertIsInstance(passed_core_profiles, state.BaseCoreProfiles)
    self.assertNotEmpty(passed_fast_ions)
    passed_he3 = [fi for fi in passed_fast_ions if fi.species == 'He3']
    self.assertLen(passed_he3, 1)
    np.testing.assert_allclose(passed_he3[0].n.value, 1e19)
    self.assertNotEmpty(result.fast_ions)
    final_he3 = [fi for fi in result.fast_ions if fi.species == 'He3']
    self.assertLen(final_he3, 1)
    np.testing.assert_allclose(final_he3[0].n.value, 1e19)

  def test_build_core_profiles_jittable(self):
    models = self.torax_config.build_models()
    source_states = models.source_models.initial_state(self.runtime_params)

    @jax.jit
    def jitted_build_core_profiles(solver_state):
      return builders.build_core_profiles(
          runtime_params=self.runtime_params,
          geo=self.geo,
          solver_state=solver_state,
          source_models=models.source_models,
          neoclassical_model=models.neoclassical_model,
          source_states=source_states,
          explicit=True,
      )

    result, _, _, _, _ = jitted_build_core_profiles(self.solver_state)
    self.assertIsInstance(result, state.CoreProfiles)
    np.testing.assert_allclose(
        result.kinetic.T_i.value,
        self.solver_state.T_i.value,
    )


if __name__ == '__main__':
  absltest.main()
