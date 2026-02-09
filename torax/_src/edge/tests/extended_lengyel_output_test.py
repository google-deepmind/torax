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

import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import sim_state
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.solver import jax_root_finding
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


class ExtendedLengyelOutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'T_i_right_bc': 27.7,
            'T_e_right_bc': {0.0: 42.0, 1.0: 0.0001},
            'n_e_right_bc': ({0.0: 0.1e20, 1.0: 2.0e20}, 'step'),
        },
        'numerics': {},
        'plasma_composition': {
            'impurity': {
                'impurity_mode': 'n_e_ratios',
                'species': {'Ne': 0.01},
            }
        },
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': default_sources.get_default_source_config(),
        'solver': {},
        'transport': {'model_name': 'constant'},
        'pedestal': {},
    })

    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    ones = jnp.ones_like(self.geo.rho)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        self.torax_config
    )(t=0.0)

    self.source_profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        T_i={'fusion': ones, 'generic_heat': 2 * ones},
        T_e={
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
            'fusion': ones,
            'generic_heat': 3 * ones,
        },
        n_e={},
        psi={},
    )

    source_models = self.torax_config.sources.build_models()
    neoclassical_models = self.torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    self.core_transport = state.CoreTransport.zeros(self.geo)

    t = jnp.array(0.0)
    dt = jnp.array(0.1)
    self.sim_state = sim_state.SimState(
        core_profiles=self.core_profiles,
        core_transport=self.core_transport,
        core_sources=self.source_profiles,
        t=t,
        dt=dt,
        solver_numeric_outputs=state.SolverNumericOutputs(
            outer_solver_iterations=1,
            solver_error_state=1,
            inner_solver_iterations=1,
            sawtooth_crash=False,
        ),
        geometry=self.geo,
        edge_outputs=None,
    )

    previous_post_processed_outputs = (
        post_processing.PostProcessedOutputs.zeros(self.geo)
    )
    self._output_state = post_processing.make_post_processed_outputs(
        self.sim_state, runtime_params, previous_post_processed_outputs
    )

  def test_roots_are_saved_correctly(self):
    """Tests that the 'roots' dimension is saved and resized correctly."""

    # Simulate 3 roots found
    num_roots = 3

    # Create batch outputs for roots
    # We populate some fields with batch dimension
    roots_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        q_parallel=jnp.ones((num_roots,)) * 1.5,
        q_perpendicular_target=jnp.ones((num_roots,)) * 2.5,
        T_e_separatrix=jnp.ones((num_roots,)) * 3.5,
        T_e_target=jnp.array([10.0, 50.0, 100.0]),  # distinct roots
        pressure_neutral_divertor=jnp.ones((num_roots,)) * 5.5,
        alpha_t=jnp.ones((num_roots,)) * 0.5,
        kappa_e=jnp.ones((num_roots,)) * 0.1,
        c_z_prefactor=jnp.ones((num_roots,)) * 1.0,
        Z_eff_separatrix=jnp.ones((num_roots,)) * 1.5,
        seed_impurity_concentrations={'Ne': jnp.ones((num_roots,)) * 0.01},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=jnp.array([
                extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
                extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
                extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            ]),
            numerics_outcome=jax_root_finding.RootMetadata(
                iterations=jnp.ones((num_roots,), dtype=jnp.int32) * 5,
                residual=jnp.ones((num_roots, 2))
                * 1e-4,  # residual is vector per root
                error=jnp.zeros((num_roots,)),
                last_tau=jnp.ones((num_roots,)),
            ),  # pytype: disable=wrong-arg-types
        ),  # pytype: disable=wrong-arg-types
        calculated_enrichment={'Ne': jnp.ones((num_roots,)) * 1.0},
    )  # pytype: disable=wrong-arg-types

    extended_lengyel_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        q_parallel=jnp.array(1.0),
        q_perpendicular_target=jnp.array(2.0),
        T_e_separatrix=jnp.array(3.0),
        T_e_target=jnp.array(4.0),
        pressure_neutral_divertor=jnp.array(5.0),
        alpha_t=jnp.array(0.5),
        kappa_e=jnp.array(0.1),
        c_z_prefactor=jnp.array(1.0),
        Z_eff_separatrix=jnp.array(1.5),
        seed_impurity_concentrations={'Ne': jnp.array(0.01)},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
        ),
        calculated_enrichment={'Ne': jnp.array(1.0)},
        roots=roots_outputs,
        multiple_roots_found=jnp.array(True),
    )  # pytype: disable=wrong-arg-types

    sim_state_with_edge = dataclasses.replace(
        self.sim_state,
        edge_outputs=extended_lengyel_outputs,
    )

    history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[sim_state_with_edge],
        post_processed_outputs_history=(self._output_state,),
        torax_config=self.torax_config,
    )

    output_xr = history.simulation_output_to_xr()
    self.assertIsNotNone(output_xr)
    self.assertIn(output.EDGE, output_xr.children)
    edge_node = output_xr.children[output.EDGE]

    # Check if 'roots' child node exists
    self.assertIn('roots', edge_node.children)
    roots_dataset = edge_node.children['roots'].dataset

    # Let's Assert that 'T_e_target' is in data_vars (without prefix)
    self.assertIn('T_e_target', roots_dataset.data_vars)

    roots_Te = roots_dataset['T_e_target']
    self.assertIn('n_roots', roots_Te.dims)

    # Verify values
    root_values = roots_Te.values
    # Shape: (time, roots) => (1, 3)
    self.assertEqual(root_values.shape, (1, 3))

    expected_roots = [10.0, 50.0, 100.0]
    np.testing.assert_allclose(root_values[0], expected_roots)


if __name__ == '__main__':
  absltest.main()
