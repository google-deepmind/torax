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

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from torax._src.edge.extended_lengyel import extended_lengyel_solvers
from torax._src.edge.extended_lengyel import extended_lengyel_standalone
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.solver import jax_root_finding

# pylint: disable=invalid-name


class ExtendedLengyelOutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    rho_cell = np.linspace(0.125, 0.875, 4)
    self.context = output_grid_context.OutputGridContext(
        times=np.array([0.0]),
        rho_face_norm=np.linspace(0.0, 1.0, 5),
        rho_cell_norm=rho_cell,
        rho_cell_plus_boundaries_norm=np.concatenate([[0.0], rho_cell, [1.0]]),
    )

  def _make_outputs(
      self, **overrides: Any
  ) -> extended_lengyel_standalone.ExtendedLengyelOutputs:
    defaults: dict[str, Any] = dict(
        T_e_right_bc=jnp.array(3.0),
        T_i_right_bc=jnp.array(3.0),
        impurity_right_bc={'Ar': jnp.array(0.01)},
        n_e_right_bc=jnp.array(jnp.nan),
        q_parallel=jnp.array(1.0),
        q_perpendicular_target=jnp.array(2.0),
        T_e_separatrix=jnp.array(3.0),
        T_e_target=jnp.array(4.0),
        pressure_neutral_divertor=jnp.array(5.0),
        alpha_t=jnp.array(0.5),
        kappa_e=jnp.array(0.1),
        c_z_prefactor=jnp.array(1.0),
        Z_eff_separatrix=jnp.array(1.5),
        seed_impurity_concentrations={'Ar': jnp.array(0.01)},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
        ),
        calculated_enrichment={'Ar': jnp.array(1.0)},
    )
    defaults.update(overrides)
    single_step = extended_lengyel_standalone.ExtendedLengyelOutputs(**defaults)
    return jax.tree_util.tree_map(lambda x: np.stack([x]), single_step)

  @parameterized.named_parameters(
      (
          'fixed_point',
          lambda: extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
          (extended_lengyel_standalone.FIXED_POINT_OUTCOME,),
      ),
      (
          'newton',
          lambda: jax_root_finding.RootMetadata(
              iterations=jnp.array(10),
              residual=jnp.array([1e-6, 3e-6]),
              error=jnp.array(0),
              last_tau=jnp.array(1.0),
          ),
          (
              extended_lengyel_standalone.SOLVER_ITERATIONS,
              extended_lengyel_standalone.SOLVER_RESIDUAL,
              extended_lengyel_standalone.SOLVER_ERROR,
          ),
      ),
  )
  def test_extended_lengyel_outputs_to_xr(
      self, numerics_outcome_fn, expected_solver_keys
  ):
    """Tests that ExtendedLengyelOutputs serialize to xarray DataTree."""
    numerics_outcome = numerics_outcome_fn()
    outputs = self._make_outputs(
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=numerics_outcome,
        ),
    )
    edge_node = outputs.to_xr_datatree(self.context)
    edge_dataset = edge_node.dataset

    # Check standard fields
    self.assertIn(output_keys.T_E_RIGHT_BC, edge_dataset.data_vars)
    self.assertIn(output_keys.T_I_RIGHT_BC, edge_dataset.data_vars)
    self.assertIn(output_keys.N_E_RIGHT_BC, edge_dataset.data_vars)
    self.assertIn(output_keys.IMPURITY_RIGHT_BC, edge_dataset.data_vars)

    # Check extended fields
    self.assertIn(
        extended_lengyel_standalone.Q_PARALLEL, edge_dataset.data_vars
    )
    self.assertIn(
        extended_lengyel_standalone.T_E_TARGET, edge_dataset.data_vars
    )
    self.assertIn(extended_lengyel_standalone.ALPHA_T, edge_dataset.data_vars)
    self.assertIn(
        extended_lengyel_standalone.Z_EFF_SEPARATRIX, edge_dataset.data_vars
    )
    self.assertIn(
        extended_lengyel_standalone.SEED_IMPURITY_CONCENTRATIONS,
        edge_dataset.data_vars,
    )
    self.assertIn(
        extended_lengyel_standalone.CALCULATED_ENRICHMENT,
        edge_dataset.data_vars,
    )
    self.assertIn(
        extended_lengyel_standalone.SOLVER_PHYSICS_OUTCOME,
        edge_dataset.data_vars,
    )
    for solver_key in expected_solver_keys:
      self.assertIn(solver_key, edge_dataset.data_vars)

    # Verify values match
    np.testing.assert_allclose(
        edge_dataset[extended_lengyel_standalone.ALPHA_T].values,
        np.array([0.5]),
    )
    np.testing.assert_allclose(
        edge_dataset[extended_lengyel_standalone.SEED_IMPURITY_CONCENTRATIONS]
        .sel(seed_impurity='Ar')
        .values,
        np.array([0.01]),
    )
    np.testing.assert_allclose(
        edge_dataset[output_keys.T_E_RIGHT_BC].values,
        np.array([3.0]),
    )
    np.testing.assert_allclose(
        edge_dataset[output_keys.T_I_RIGHT_BC].values,
        np.array([3.0]),
    )
    self.assertTrue(np.isnan(edge_dataset[output_keys.N_E_RIGHT_BC].values[0]))
    np.testing.assert_allclose(
        edge_dataset[output_keys.IMPURITY_RIGHT_BC].sel(impurity='Ar').values,
        np.array([0.01]),
    )
    if isinstance(numerics_outcome, jax_root_finding.RootMetadata):
      np.testing.assert_allclose(
          edge_dataset[extended_lengyel_standalone.SOLVER_ITERATIONS].values,
          np.array([10]),
      )
      self.assertEqual(
          edge_dataset[extended_lengyel_standalone.SOLVER_RESIDUAL].dims,
          (output_keys.TIME,),
      )
      # Mean of abs([1e-6, 3e-6]) is 2e-6
      np.testing.assert_allclose(
          edge_dataset[extended_lengyel_standalone.SOLVER_RESIDUAL].values,
          np.array([2e-6]),
      )

  def test_seed_impurity_concentrations_does_not_align_with_enrichment(self):
    outputs = self._make_outputs(
        impurity_right_bc={'Ar': jnp.array(0.01)},
        seed_impurity_concentrations={'Ar': jnp.array(0.01)},
        calculated_enrichment={'Ar': jnp.array(1.0), 'W': jnp.array(0.5)},
    )
    edge_dataset = outputs.to_xr_datatree(self.context).dataset

    # Verify seed_impurity_concentrations has dimension SEED_IMPURITY ('Ar')
    self.assertIn(extended_lengyel_standalone.SEED_IMPURITY, edge_dataset.dims)
    seed_var = edge_dataset[
        extended_lengyel_standalone.SEED_IMPURITY_CONCENTRATIONS
    ]
    self.assertIn(extended_lengyel_standalone.SEED_IMPURITY, seed_var.dims)
    self.assertLen(
        seed_var.coords[extended_lengyel_standalone.SEED_IMPURITY], 1
    )
    self.assertEqual(
        seed_var.coords[extended_lengyel_standalone.SEED_IMPURITY].values[0],
        'Ar',
    )

    # Verify calculated_enrichment has dimension ENRICHMENT_IMPURITY with
    # 'Ar' and 'W'.
    enrich_var = edge_dataset[extended_lengyel_standalone.CALCULATED_ENRICHMENT]
    self.assertIn(
        extended_lengyel_standalone.ENRICHMENT_IMPURITY, enrich_var.dims
    )
    self.assertLen(
        enrich_var.coords[extended_lengyel_standalone.ENRICHMENT_IMPURITY], 2
    )
    self.assertCountEqual(
        enrich_var.coords[
            extended_lengyel_standalone.ENRICHMENT_IMPURITY
        ].values,
        ['Ar', 'W'],
    )

    # Verify impurity_right_bc has dimension output_keys.IMPURITY with coords
    # ['Ar'] only.
    self.assertIn(output_keys.IMPURITY_RIGHT_BC, edge_dataset.data_vars)
    bc_var = edge_dataset[output_keys.IMPURITY_RIGHT_BC]
    self.assertIn(output_keys.IMPURITY, bc_var.dims)
    self.assertLen(bc_var.coords[output_keys.IMPURITY], 1)
    self.assertEqual(
        bc_var.coords[output_keys.IMPURITY].values[0],
        'Ar',
    )
    np.testing.assert_allclose(
        bc_var.sel(impurity='Ar').values,
        np.array([0.01]),
    )

  def test_roots_are_saved_correctly(self):
    """Tests that the 'roots' dimension is saved and resized correctly."""
    num_roots = 3
    roots_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        T_e_right_bc=jnp.ones((num_roots,)) * 3.5,
        T_i_right_bc=jnp.ones((num_roots,)) * 3.5,
        impurity_right_bc={'Ne': jnp.ones((num_roots,)) * 0.01},
        n_e_right_bc=jnp.full((num_roots,), jnp.nan),
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
                residual=jnp.ones((num_roots, 2)) * 1e-4,
                error=jnp.zeros((num_roots,)),
                last_tau=jnp.ones((num_roots,)),
            ),  # type: ignore[arg-type]
        ),  # type: ignore[arg-type]
        calculated_enrichment={'Ne': jnp.ones((num_roots,)) * 1.0},
    )  # type: ignore[arg-type]

    outputs = self._make_outputs(
        impurity_right_bc={'Ne': jnp.array(0.01)},
        seed_impurity_concentrations={'Ne': jnp.array(0.01)},
        calculated_enrichment={'Ne': jnp.array(1.0)},
        roots=roots_outputs,
        multiple_roots_found=jnp.array(True),
    )

    edge_node = outputs.to_xr_datatree(self.context)
    self.assertIn(extended_lengyel_standalone.ROOTS, edge_node.children)
    roots_dataset = (
        edge_node.children[extended_lengyel_standalone.ROOTS].dataset
    )

    # Assert T_e_target is in data_vars without prefix.
    self.assertIn(
        extended_lengyel_standalone.T_E_TARGET, roots_dataset.data_vars
    )
    roots_Te = roots_dataset[extended_lengyel_standalone.T_E_TARGET]
    self.assertIn(extended_lengyel_standalone.N_ROOTS, roots_Te.dims)

    root_values = roots_Te.values
    self.assertEqual(root_values.shape, (1, 3))
    np.testing.assert_allclose(root_values[0], [10.0, 50.0, 100.0])

    # Verify standalone edge dataset coordinates contain time but not spatial
    # radial grids.
    self.assertIn(output_keys.TIME, edge_node.dataset.coords)
    self.assertNotIn(output_keys.RHO_CELL_NORM, edge_node.dataset.coords)
    self.assertNotIn(output_keys.RHO_FACE_NORM, edge_node.dataset.coords)
    self.assertNotIn(output_keys.RHO_NORM, edge_node.dataset.coords)


if __name__ == '__main__':
  absltest.main()
