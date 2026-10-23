# Copyright 2024 DeepMind Technologies Limited
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

import copy
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import convection_terms
from torax._src.fvm import diffusion_terms
from torax._src.fvm import implicit_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.orchestration import initial_state
from torax._src.orchestration import run_simulation
from torax._src.sources import source_profile_builders
from torax._src.torax_pydantic import model_config


def _make_face_centers(dr: float, num_cells: int) -> np.ndarray:
  """Creates uniform face centers from dr and number of cells."""
  return np.linspace(0.0, num_cells * dr, num=num_cells + 1)


class FVMTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_cells=2, theta_implicit=0, time_steps=29),
      dict(num_cells=3, theta_implicit=0.5, time_steps=21),
      dict(num_cells=4, theta_implicit=1.0, time_steps=34),
  ])
  def test_leftward_convection(self, num_cells, theta_implicit, time_steps):
    """Tests that leftward convection spreads the right boundary value."""
    num_faces = num_cells + 1
    right_boundary = jnp.array((1.0, -2.0))
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        face_centers=_make_face_centers(1.0, num_cells),
        right_face_grad_constraint=None,
        right_face_constraint=right_boundary[0],
    )
    x_1 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        face_centers=_make_face_centers(1.0, num_cells),
        right_face_grad_constraint=None,
        right_face_constraint=right_boundary[1],
    )
    x = (x_0, x_1)
    # Not deeply investigated, but dt = 1. seems unstable for explicit method.
    dt = jnp.array(1.0 - 0.5 * (theta_implicit == 0))
    transient_cell_i = jnp.ones(num_cells)
    transient_cell = (transient_cell_i, transient_cell_i)
    # Use convection leftward everywhere so the right boundary condition will
    # flow across the whole mesh
    v_face_i = -jnp.ones(num_faces)
    v_face = (v_face_i, v_face_i)
    coeffs = block_1d_coeffs.Block1DCoeffs(
        transient_out_cell=transient_cell,
        transient_in_cell=transient_cell,
        v_face=v_face,
    )
    for _ in range(time_steps):
      x = implicit_solve_block.implicit_solve_block(
          dt=dt,
          x_old=x,
          x_new_guess=x,
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          theta_implicit=theta_implicit,
      )

    np.testing.assert_allclose(x[0].value, right_boundary[0])
    np.testing.assert_allclose(x[1].value, right_boundary[1])

  @parameterized.parameters([
      dict(theta_implicit=0.0),
      dict(theta_implicit=0.5),
      dict(theta_implicit=1.0),
  ])
  def test_implicit_source_cross(self, theta_implicit):
    """Tests that implicit source cross terms act on sub-timestep scale."""

    # We model the evolution of two scalars, x and y:
    # x(0) = 0, y(0) = 0
    # dx / dt = 1, dy /dt = x.
    #
    # The analytical solution to this is
    # x(t) = t
    # y(t) = t^2 / 2.
    #
    # Now consider using a differential equation solver to step from t=0 to
    # t=delta_t.
    # An explicit solver, or an implicit solver using an explicit source term
    # to model the dependence of y on x, will have y(delta_t) = 0 because
    # x(0)=0. This approach will need a second time step before y becomes
    # nonzero.
    # An implicit solver using an implicit source term will correctly have
    # y > 0 on the first step.
    #
    # Mapping this onto the terminology of `solver.implicit_solve_block`, we
    # use a grid with only one cell per channel, with one channel representing
    # x and the other representing y.

    # We have to use 2 cells to avoid the unsupported corner case where the
    # mesh consists of only one cell, with the same cell affected by both
    # boundary conditions.
    # For the purposes of this test, both cells model the same scalar, so
    # it's OK to look at either cell in isolation. Since there is 0 diffusion
    # and 0 convection the two cells don't interact.
    num_cells = 2

    num_faces = num_cells + 1
    dt = jnp.array(1.0)
    transient_cell_i = jnp.ones(num_cells)
    transient_cell = (transient_cell_i, transient_cell_i)
    d_face_i = jnp.zeros(num_cells + 1)
    d_face = (d_face_i, d_face_i)
    v_face_i = jnp.zeros(num_faces)
    v_face = (v_face_i, v_face_i)
    source_mat_i = jnp.zeros(num_cells)
    right_boundary = jnp.array(0.0)

    kwargs = {
        'dt': dt,
        'theta_implicit': theta_implicit,
    }

    # Make x start to increase in channel `start` and drive an increase in the
    # other channel.
    # Exercise both directions to make sure we test both off-diagonal blocks of
    # the solver.
    for start in [0, 1]:
      # Make both x_0 and x_1 start at 0
      x_0 = cell_variable.CellVariable(
          value=jnp.zeros(num_cells),
          face_centers=_make_face_centers(1.0, num_cells),
          right_face_grad_constraint=None,
          right_face_constraint=right_boundary,
      )
      x_1 = cell_variable.CellVariable(
          value=jnp.zeros(num_cells),
          face_centers=_make_face_centers(1.0, num_cells),
          right_face_grad_constraint=None,
          right_face_constraint=right_boundary,
      )
      x = (x_0, x_1)

      # Mark the starting channel drive the destination channel
      source_mat_01 = jnp.ones(num_cells) * start
      source_mat_10 = jnp.ones(num_cells) * (1 - start)
      source_mat_cell = (
          (source_mat_i, source_mat_01),
          (source_mat_10, source_mat_i),
      )
      # Make the starting channel increase during the time step
      source_0 = jnp.ones(num_cells) * (1 - start)
      source_1 = jnp.ones(num_cells) * start
      source_cell = (source_0, source_1)
      coeffs = block_1d_coeffs.Block1DCoeffs(
          transient_out_cell=transient_cell,
          transient_in_cell=transient_cell,
          d_face=d_face,
          v_face=v_face,
          source_mat_cell=source_mat_cell,
          source_cell=source_cell,
      )

      x = implicit_solve_block.implicit_solve_block(
          x_old=x,
          x_new_guess=x,
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          **kwargs,
      )

      if theta_implicit == 0.0:
        # For explicit method, the source terms are applied at t=0, when
        # u[start] == 0. So they should have no effect
        np.testing.assert_allclose(x[1 - start].value, 0.0)
      else:
        # By t=1, u[start] is greater than 0, and the implicit source terms
        # should also drive u[1 - start] to be greater than 0
        self.assertGreater(x[1 - start].value.min(), 0.0)

  def _build_fvm_test_inputs(
      self, num_cells: int = 4, theta_implicit: float = 1.0
  ):
    """Builds FVM test inputs for diffusive transport with no sources."""
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(),
            sources=dict(),
            solver=dict(
                use_predictor_corrector=False, theta_implicit=theta_implicit
            ),
            transport=dict(
                core_transport_models=dict(
                    prescribed=dict(model_name='prescribed', chi_i=1)
                ),
                chi_min=0,
            ),
            time_step_calculator=dict(),
        )
    )
    step_fn = run_simulation.make_step_fn(torax_config)
    runtime_params = step_fn.runtime_params_provider(
        t=torax_config.numerics.t_initial
    )
    geo = step_fn.geometry_provider(torax_config.numerics.t_initial)
    sim_state = initial_state._get_initial_state(
        runtime_params=runtime_params,
        geo=geo,
        step_fn=step_fn,
    )
    evolving_names = ('T_i',)
    explicit_source_profiles = source_profile_builders.build_all_zero_profiles(
        geo
    )
    coeffs = calc_coeffs.calc_coeffs(
        runtime_params=runtime_params,
        geo=sim_state.geometry,
        core_profiles=sim_state.core_profiles,
        models=step_fn.solver.models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
        pedestal_transition_state=sim_state.pedestal_transition_state,
        model_state=sim_state.model_state,
    )
    return (
        step_fn,
        runtime_params,
        sim_state,
        evolving_names,
        explicit_source_profiles,
        coeffs,
    )

  @parameterized.parameters([
      dict(num_cells=4, theta_implicit=0, time_steps=29),
      dict(num_cells=4, theta_implicit=0.5, time_steps=21),
      dict(num_cells=5, theta_implicit=1.0, time_steps=34),
  ])
  def test_linear_solution_yields_zero_nonlinear_residual_and_loss(
      self, num_cells, theta_implicit, time_steps
  ):
    (
        step_fn,
        runtime_params,
        sim_state,
        evolving_names,
        explicit_source_profiles,
        coeffs,
    ) = self._build_fvm_test_inputs(
        num_cells=num_cells, theta_implicit=theta_implicit
    )
    # dt well under the explicit stability limit for dx=1 and chi=1
    dt = jnp.array(0.2)
    # initialize x_new for timestepping
    x_new = (sim_state.core_profiles.T_i,)
    for _ in range(time_steps):
      x_old = copy.deepcopy(x_new)
      x_new = implicit_solve_block.implicit_solve_block(
          dt=dt,
          x_old=x_old,
          x_new_guess=x_new,
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          theta_implicit=theta_implicit,
      )

      # When the coefficients are kept constant, the loss
      # should just be a quadratic bowl with the linear
      # solution as the minimum with approximately zero residual
      # core_profiles_t_plus_dt is not updated since coeffs stay constant here
      loss = residual_and_loss.theta_method_block_loss(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params,
          geo_t_plus_dt=sim_state.geometry,
          x_old=x_old,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          core_profiles_t_plus_dt=sim_state.core_profiles,
          models=step_fn.solver.models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
          pedestal_transition_state=sim_state.pedestal_transition_state,
          model_state=sim_state.model_state,
      )

      residual = residual_and_loss.theta_method_block_residual(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params,
          geo_t_plus_dt=sim_state.geometry,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          x_old=x_old,
          core_profiles_t_plus_dt=sim_state.core_profiles,
          models=step_fn.solver.models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
          pedestal_transition_state=sim_state.pedestal_transition_state,
          model_state=sim_state.model_state,
      )

      np.testing.assert_allclose(loss, 0.0, atol=1e-7)
      np.testing.assert_allclose(residual, 0.0, atol=1e-7)

  def test_implicit_solve_block_uses_updated_boundary_conditions(self):
    num_cells = 4
    _, _, _, _, _, coeffs = self._build_fvm_test_inputs(
        num_cells=num_cells, theta_implicit=1.0
    )
    dt = jnp.array(1.0)
    initial_right_boundary = jnp.array(0.0)
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        face_centers=_make_face_centers(1.0, num_cells),
        right_face_grad_constraint=None,
        right_face_constraint=initial_right_boundary,
    )
    # Run with different theta_implicit values.
    for theta_implicit in [0.0, 0.5, 1.0]:
      x_new = implicit_solve_block.implicit_solve_block(
          dt=dt,
          x_old=(x_0,),
          x_new_guess=(x_0,),
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          theta_implicit=theta_implicit,
      )
      # No matter what theta_implicit is used, the x_new will be all 0s because
      # there is no source and the boundaries are set to 0.
      np.testing.assert_allclose(x_new[0].value, 0.0)

    # If we run with an updated boundary condition applied at time t=dt, then
    # we should get non-zero values from the implicit terms.
    final_right_boundary = jnp.array(1.0)
    x_1 = dataclasses.replace(x_0, right_face_constraint=final_right_boundary)
    # However, the explicit terms (when theta_implicit = 0), should still be
    # all 0.
    x_new = implicit_solve_block.implicit_solve_block(
        dt=dt,
        x_old=(x_0,),
        x_new_guess=(x_1,),
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_new=coeffs,
        theta_implicit=0.0,
    )
    np.testing.assert_allclose(x_new[0].value, 0.0)
    # x_new should still have the updated boundary conditions though.
    np.testing.assert_allclose(
        x_new[0].right_face_constraint, final_right_boundary
    )
    # And when theta_implicit is > 0, the values should be > 0.
    x_new = implicit_solve_block.implicit_solve_block(
        dt=dt,
        x_old=(x_0,),
        x_new_guess=(x_1,),
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_new=coeffs,
        theta_implicit=0.5,
    )
    self.assertGreater(x_new[0].value.min(), 0.0)

  def test_theta_residual_with_static_boundary_conditions(self):
    num_cells = 4
    (
        step_fn,
        runtime_params,
        sim_state,
        evolving_names,
        explicit_source_profiles,
        coeffs_old,
    ) = self._build_fvm_test_inputs(num_cells=num_cells, theta_implicit=0.5)
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        face_centers=_make_face_centers(1.0, num_cells),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(0.0),
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        sim_state.core_profiles,
        T_i=x_0,
    )
    # When the boundary conditions are not time-dependent and stay at 0,
    # with diffusive transport and zero transport, then the state will stay
    # at all 0, and the residual should be 0.
    residual = residual_and_loss.theta_method_block_residual(
        dt=jnp.array(1.0),
        runtime_params_t_plus_dt=runtime_params,
        geo_t_plus_dt=sim_state.geometry,
        x_old=(x_0,),
        x_new_guess_vec=x_0.value,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        models=step_fn.solver.models,
        explicit_source_profiles=explicit_source_profiles,
        coeffs_old=coeffs_old,
        evolving_names=evolving_names,
        pedestal_transition_state=sim_state.pedestal_transition_state,
        model_state=sim_state.model_state,
    )
    np.testing.assert_allclose(residual, 0.0)

  def test_theta_residual_with_updated_boundary_conditions(self):
    num_cells = 4
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        face_centers=_make_face_centers(1.0, num_cells),
        right_face_grad_constraint=None,
        right_face_constraint=jnp.array(0.0),
    )
    x_1 = dataclasses.replace(x_0, right_face_constraint=jnp.array(1.0))

    # When the boundary condition updates at time t+dt, then the implicit part
    # of the update would generate a residual. When theta_implicit is 0, the
    # residual would still be 0.
    (
        step_fn,
        runtime_params_theta0,
        sim_state,
        evolving_names,
        explicit_source_profiles,
        coeffs_old,
    ) = self._build_fvm_test_inputs(num_cells=num_cells, theta_implicit=0.0)
    core_profiles_t_plus_dt = dataclasses.replace(
        sim_state.core_profiles,
        T_i=x_1,
    )
    residual_theta0 = residual_and_loss.theta_method_block_residual(
        dt=jnp.array(1.0),
        runtime_params_t_plus_dt=runtime_params_theta0,
        geo_t_plus_dt=sim_state.geometry,
        x_old=(x_0,),
        x_new_guess_vec=x_0.value,
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        evolving_names=evolving_names,
        models=step_fn.solver.models,
        explicit_source_profiles=explicit_source_profiles,
        coeffs_old=coeffs_old,
        pedestal_transition_state=sim_state.pedestal_transition_state,
        model_state=sim_state.model_state,
    )
    np.testing.assert_allclose(residual_theta0, 0.0)

    # But when theta_implicit > 0, the residual should be non-zero.
    _, runtime_params_theta05, _, _, _, _ = self._build_fvm_test_inputs(
        num_cells=num_cells, theta_implicit=0.5
    )
    residual_theta05 = residual_and_loss.theta_method_block_residual(
        dt=jnp.array(1.0),
        runtime_params_t_plus_dt=runtime_params_theta05,
        geo_t_plus_dt=sim_state.geometry,
        x_old=(x_0,),
        core_profiles_t_plus_dt=core_profiles_t_plus_dt,
        x_new_guess_vec=x_0.value,
        models=step_fn.solver.models,
        explicit_source_profiles=explicit_source_profiles,
        coeffs_old=coeffs_old,
        evolving_names=evolving_names,
        pedestal_transition_state=sim_state.pedestal_transition_state,
        model_state=sim_state.model_state,
    )
    self.assertGreater(jnp.abs(jnp.sum(residual_theta05)), 0.0)


def _ghost_extended_values(
    u: jax.Array | np.ndarray,
    cell_widths: np.ndarray,
    bc_kwargs: dict[str, float | None],
) -> jax.Array:
  """Returns `u` padded with mirrored ghost-cell values at both edges."""
  left_value = bc_kwargs['left_face_constraint']
  left_grad = bc_kwargs['left_face_grad_constraint']
  right_value = bc_kwargs['right_face_constraint']
  right_grad = bc_kwargs['right_face_grad_constraint']
  if left_value is not None:
    u_left = 2.0 * left_value - u[0]
  else:
    assert left_grad is not None
    u_left = u[0] - cell_widths[0] * left_grad
  if right_value is not None:
    u_right = 2.0 * right_value - u[-1]
  else:
    assert right_grad is not None
    u_right = u[-1] + cell_widths[-1] * right_grad
  return jnp.concatenate([jnp.array([u_left]), u, jnp.array([u_right])])


# Boundary conditions consistent with the linear profile u(x) = 1 + 3x.
_NON_UNIFORM_GRID_BCS = (
    dict(
        testcase_name='dirichlet',
        bc_kwargs=dict(
            left_face_constraint=1.0,
            left_face_grad_constraint=None,
            right_face_constraint=5.2,
            right_face_grad_constraint=None,
        ),
    ),
    dict(
        testcase_name='neumann',
        bc_kwargs=dict(
            left_face_constraint=None,
            left_face_grad_constraint=3.0,
            right_face_constraint=None,
            right_face_grad_constraint=3.0,
        ),
    ),
)


class NonUniformGridTermsTest(parameterized.TestCase):
  """Tests diffusion and convection terms on an asymmetric non-uniform mesh."""

  # Cell widths (0.2, 0.4, 0.8) and center spacings (0.3, 0.6) are distinct.
  # NumPy (not JAX) arrays: class attributes are evaluated at import time.
  _FACE_CENTERS = np.array([0.0, 0.2, 0.6, 1.4])
  # Linear profile u(x) = 1 + 3x evaluated at the cell centers (0.1, 0.4, 1.0).
  _U_LINEAR = np.array([1.3, 2.2, 4.0])
  # Arbitrary non-linear cell values.
  _U = np.array([1.2, -0.5, 2.3])

  def setUp(self):
    super().setUp()
    self.cell_widths = np.diff(self._FACE_CENTERS)
    cell_centers = 0.5 * (self._FACE_CENTERS[1:] + self._FACE_CENTERS[:-1])
    self.center_spacings = np.diff(cell_centers)

  def _make_var(
      self, value: np.ndarray, bc_kwargs: dict[str, float | None]
  ) -> cell_variable.CellVariable:
    return cell_variable.CellVariable(
        value=jnp.asarray(value),
        face_centers=self._FACE_CENTERS,
        **{
            k: None if v is None else jnp.asarray(v)
            for k, v in bc_kwargs.items()
        },
    )

  @parameterized.named_parameters(*_NON_UNIFORM_GRID_BCS)
  def test_diffusion_terms(self, bc_kwargs: dict[str, float | None]):
    with self.subTest('linear_profile_zero_divergence'):
      linear_var = self._make_var(self._U_LINEAR, bc_kwargs)
      mat, vec = diffusion_terms.make_diffusion_terms(
          d_face=jnp.full(4, 2.5), var=linear_var
      )
      np.testing.assert_allclose(
          mat.matvec(linear_var.value) + vec, 0.0, atol=1e-13
      )

    with self.subTest('flux_balance'):
      var = self._make_var(self._U, bc_kwargs)
      d_face = jnp.array([1.5, 2.0, 0.8, 3.0])
      mat, vec = diffusion_terms.make_diffusion_terms(d_face=d_face, var=var)
      u_ext = _ghost_extended_values(var.value, self.cell_widths, bc_kwargs)
      # Ghost cells mirror the boundary cells, so the distance between a ghost
      # cell center and its neighbor is the boundary cell width.
      face_spacings = np.concatenate([
          self.cell_widths[:1],
          self.center_spacings,
          self.cell_widths[-1:],
      ])
      face_flux = d_face * jnp.diff(u_ext) / face_spacings
      np.testing.assert_allclose(
          mat.matvec(var.value) + vec,
          jnp.diff(face_flux) / self.cell_widths,
          atol=1e-13,
      )

  @parameterized.named_parameters(*_NON_UNIFORM_GRID_BCS)
  def test_convection_terms(self, bc_kwargs: dict[str, float | None]):
    var = self._make_var(self._U, bc_kwargs)
    u = var.value
    v_face = jnp.array([0.7, -1.3, 0.9, -0.4])
    d_face = jnp.array([1.0, 1.1, 0.3, 2.0])
    # Péclet numbers use center-to-center spacings in the interior and the
    # cell-center-to-face half-width at the boundaries.
    peclet_spacings = np.concatenate([
        0.5 * self.cell_widths[:1],
        self.center_spacings,
        0.5 * self.cell_widths[-1:],
    ])
    peclet = peclet_spacings * v_face / d_face
    # The face Péclet numbers (0.07, -0.35, 1.8, -0.08) straddle the series
    # threshold, so both branches of `_peclet_to_alpha` are exercised.
    abs_peclet = np.abs(np.asarray(peclet))
    self.assertTrue(np.any(abs_peclet < convection_terms._SMALL_PECLET))
    self.assertTrue(np.any(abs_peclet >= convection_terms._SMALL_PECLET))

    with self.subTest('flux_assembly'):
      mat, vec = convection_terms.make_convection_terms(v_face, d_face, var)
      u_ext = _ghost_extended_values(u, self.cell_widths, bc_kwargs)
      alpha = convection_terms._peclet_to_alpha(peclet)
      face_flux = v_face * (alpha * u_ext[:-1] + (1.0 - alpha) * u_ext[1:])
      np.testing.assert_allclose(
          mat.matvec(u) + vec,
          -jnp.diff(face_flux) / self.cell_widths,
          atol=1e-13,
      )

    with self.subTest('conservation'):
      # With zero velocity on the boundary faces, the interior face fluxes
      # must cancel in the volume-weighted sum regardless of face weighting.
      v_face_closed = v_face.at[0].set(0.0).at[-1].set(0.0)
      mat, vec = convection_terms.make_convection_terms(
          v_face_closed, d_face, var
      )
      np.testing.assert_allclose(
          jnp.sum(self.cell_widths * (mat.matvec(u) + vec)), 0.0, atol=1e-13
      )

    with self.subTest('zero_peclet_arithmetic_mean_face_values'):
      # Pereverzev counter-convection relies on `d_face=inf` giving exactly
      # the arithmetic-mean interior face values and the CellVariable boundary
      # face values (see `pereverzev._midpoint_face_value`).
      mat, vec = convection_terms.make_convection_terms(
          v_face, jnp.full_like(v_face, jnp.inf), var
      )
      u_face = jnp.concatenate([
          var.left_face_value,
          0.5 * (u[:-1] + u[1:]),
          var.right_face_value,
      ])
      np.testing.assert_allclose(
          mat.matvec(u) + vec,
          -jnp.diff(v_face * u_face) / self.cell_widths,
          atol=1e-13,
      )


class Block1DCoeffsTest(parameterized.TestCase):

  def test_post_init_both_none(self):
    """Both internal_boundary_condition fields None is valid."""
    coeffs = block_1d_coeffs.Block1DCoeffs(
        transient_in_cell=(jnp.ones(4),),
    )
    self.assertIsNone(coeffs.internal_boundary_condition_mask)
    self.assertIsNone(coeffs.internal_boundary_condition_target_vec)

  def test_post_init_both_set(self):
    """Both internal_boundary_condition fields set is valid."""
    mask = jnp.array([[True, False], [False, True]])
    target = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    coeffs = block_1d_coeffs.Block1DCoeffs(
        transient_in_cell=(jnp.ones(2),),
        internal_boundary_condition_mask=mask,
        internal_boundary_condition_target_vec=target,
    )
    self.assertIsNotNone(coeffs.internal_boundary_condition_mask)
    self.assertIsNotNone(coeffs.internal_boundary_condition_target_vec)

  def test_post_init_mask_only_raises(self):
    """Setting only mask without target raises ValueError."""
    with self.assertRaises(ValueError):
      block_1d_coeffs.Block1DCoeffs(
          transient_in_cell=(jnp.ones(2),),
          internal_boundary_condition_mask=jnp.array([[True, False]]),
      )

  def test_post_init_target_only_raises(self):
    """Setting only target without mask raises ValueError."""
    with self.assertRaises(ValueError):
      block_1d_coeffs.Block1DCoeffs(
          transient_in_cell=(jnp.ones(2),),
          internal_boundary_condition_target_vec=jnp.array([[1.0, 0.0]]),
      )

  def test_has_internal_boundary_conditions_false(self):
    """has_internal_boundary_conditions is False when fields are None."""
    coeffs = block_1d_coeffs.Block1DCoeffs(
        transient_in_cell=(jnp.ones(4),),
    )
    self.assertFalse(coeffs.has_internal_boundary_conditions)

  def test_has_internal_boundary_conditions_true(self):
    """has_internal_boundary_conditions is True when fields are set."""
    coeffs = block_1d_coeffs.Block1DCoeffs(
        transient_in_cell=(jnp.ones(2),),
        internal_boundary_condition_mask=jnp.array([[True, False]]),
        internal_boundary_condition_target_vec=jnp.array([[1.0, 0.0]]),
    )
    self.assertTrue(coeffs.has_internal_boundary_conditions)


if __name__ == '__main__':
  absltest.main()
