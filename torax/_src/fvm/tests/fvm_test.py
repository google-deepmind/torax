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
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.fvm import block_1d_coeffs
from torax._src.fvm import calc_coeffs
from torax._src.fvm import cell_variable
from torax._src.fvm import implicit_solve_block
from torax._src.fvm import residual_and_loss
from torax._src.sources import runtime_params as source_runtime_params
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_sources
from torax._src.torax_pydantic import model_config


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
    dr = jnp.array(1.0)
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        dr=dr,
        right_face_grad_constraint=None,
        right_face_constraint=right_boundary[0],
    )
    x_1 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        dr=dr,
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
    dx = jnp.array(1.0)
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
          dr=dx,
          right_face_grad_constraint=None,
          right_face_constraint=right_boundary,
      )
      x_1 = cell_variable.CellVariable(
          value=jnp.zeros(num_cells),
          dr=dx,
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

  @parameterized.parameters([
      dict(num_cells=4, theta_implicit=0, time_steps=29),
      dict(num_cells=4, theta_implicit=0.5, time_steps=21),
      dict(num_cells=5, theta_implicit=1.0, time_steps=34),
  ])
  def test_nonlinear_solve_block_loss_minimum(
      self, num_cells, theta_implicit, time_steps
  ):
    """Tests that the linear solution for a linear problem yields zero residual and loss."""
    source_config = default_sources.get_default_source_config()
    source_config['ei_exchange']['Qei_multiplier'] = 0.0
    source_config['generic_heat']['P_total'] = 0.0
    source_config['fusion']['mode'] = source_runtime_params.Mode.ZERO
    source_config['ohmic']['mode'] = source_runtime_params.Mode.ZERO
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(),
            sources=source_config,
            solver=dict(
                use_predictor_corrector=False, theta_implicit=theta_implicit
            ),
            transport=dict(model_name='constant', chi_min=0, chi_i=1),
            time_step_calculator=dict(),
        )
    )
    physics_models = torax_config.build_physics_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    evolving_names = tuple(['T_i'])
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        explicit=True,
    )
    coeffs = calc_coeffs.calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        physics_models=physics_models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )
    # dt well under the explicit stability limit for dx=1 and chi=1
    dt = jnp.array(0.2)
    # initialize x_new for timestepping
    x_new = (core_profiles.T_i,)
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
          geo_t_plus_dt=geo,
          x_old=x_old,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          core_profiles_t_plus_dt=core_profiles,
          physics_models=physics_models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
      )

      residual = residual_and_loss.theta_method_block_residual(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params,
          geo_t_plus_dt=geo,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          x_old=x_old,
          core_profiles_t_plus_dt=core_profiles,
          physics_models=physics_models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
      )

      np.testing.assert_allclose(loss, 0.0, atol=1e-7)
      np.testing.assert_allclose(residual, 0.0, atol=1e-7)

  def test_implicit_solve_block_uses_updated_boundary_conditions(self):
    """Tests that updated boundary conditions affect x_new."""
    # Create a system with diffusive transport and no sources. When initialized
    # flat, x_new should remain zero unless boundary conditions change.
    num_cells = 4
    source_config = default_sources.get_default_source_config()
    source_config['ei_exchange']['Qei_multiplier'] = 0.0
    source_config['generic_heat']['P_total'] = 0.0
    source_config['fusion']['mode'] = source_runtime_params.Mode.ZERO
    source_config['ohmic']['mode'] = source_runtime_params.Mode.ZERO
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            plasma_composition=dict(),
            profile_conditions=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(),
            sources=source_config,
            solver=dict(use_predictor_corrector=False, theta_implicit=1.0),
            transport=dict(model_name='constant', chi_min=0, chi_i=1),
            time_step_calculator=dict(),
        )
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    physics_models = torax_config.build_physics_models()
    initial_core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        explicit=True,
    )

    dt = jnp.array(1.0)
    evolving_names = tuple(['T_i'])

    coeffs = calc_coeffs.calc_coeffs(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=initial_core_profiles,
        physics_models=physics_models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )
    initial_right_boundary = jnp.array(0.0)
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        dr=jnp.array(1.0),
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

  def test_theta_residual_uses_updated_boundary_conditions(self):
    # Create a system with diffusive transport and no sources. When initialized
    # flat, residual should remain zero unless boundary conditions change.
    num_cells = 4
    source_config = default_sources.get_default_source_config()
    source_config['ei_exchange']['Qei_multiplier'] = 0.0
    source_config['generic_heat']['P_total'] = 0.0
    source_config['fusion']['mode'] = source_runtime_params.Mode.ZERO
    source_config['ohmic']['mode'] = source_runtime_params.Mode.ZERO
    torax_config = model_config.ToraxConfig.from_dict(
        dict(
            numerics=dict(evolve_electron_heat=False),
            profile_conditions=dict(),
            plasma_composition=dict(),
            geometry=dict(geometry_type='circular', n_rho=num_cells),
            pedestal=dict(),
            sources=source_config,
            solver=dict(use_predictor_corrector=False, theta_implicit=0.0),
            transport=dict(model_name='constant', chi_min=0, chi_i=1),
            time_step_calculator=dict(),
        )
    )
    runtime_params_theta0 = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)(
            t=torax_config.numerics.t_initial,
        )
    )
    runtime_params_theta05 = dataclasses.replace(
        runtime_params_theta0,
        solver=dataclasses.replace(
            runtime_params_theta0.solver, theta_implicit=0.5
        ),
    )

    physics_models = torax_config.build_physics_models()
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    initial_core_profiles = initialization.initial_core_profiles(
        runtime_params_theta0,
        geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params_theta0,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
        explicit=True,
    )

    dt = jnp.array(1.0)
    evolving_names = tuple(['T_i'])

    coeffs_old = calc_coeffs.calc_coeffs(
        runtime_params=runtime_params_theta05,
        geo=geo,
        core_profiles=initial_core_profiles,
        physics_models=physics_models,
        explicit_source_profiles=explicit_source_profiles,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )

    initial_right_boundary = jnp.array(0.0)
    x_0 = cell_variable.CellVariable(
        value=jnp.zeros(num_cells),
        dr=jnp.array(1.0),
        right_face_grad_constraint=None,
        right_face_constraint=initial_right_boundary,
    )
    core_profiles_t_plus_dt = initialization.initial_core_profiles(
        runtime_params=runtime_params_theta05,
        geo=geo,
        source_models=physics_models.source_models,
        neoclassical_models=physics_models.neoclassical_models,
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        core_profiles_t_plus_dt,
        T_i=x_0,
    )

    with self.subTest('static_boundary_conditions'):
      # When the boundary conditions are not time-dependent and stay at 0,
      # with diffusive transport and zero transport, then the state will stay
      # at all 0, and the residual should be 0.
      residual = residual_and_loss.theta_method_block_residual(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_theta05,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          x_new_guess_vec=x_0.value,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          physics_models=physics_models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs_old,
          evolving_names=evolving_names,
      )
      np.testing.assert_allclose(residual, 0.0)
    with self.subTest('updated_boundary_conditions'):
      # When the boundary condition updates at time t+dt, then the implicit part
      # of the update would generate a residual. When theta_implicit is 0, the
      # residual would still be 0.
      final_right_boundary = jnp.array(1.0)
      residual = residual_and_loss.theta_method_block_residual(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_theta0,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          x_new_guess_vec=x_0.value,
          core_profiles_t_plus_dt=dataclasses.replace(
              core_profiles_t_plus_dt,
              T_i=dataclasses.replace(
                  x_0, right_face_constraint=final_right_boundary
              ),
          ),
          evolving_names=evolving_names,
          physics_models=physics_models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs_old,
      )
      np.testing.assert_allclose(residual, 0.0)
      # But when theta_implicit > 0, the residual should be non-zero.
      residual = residual_and_loss.theta_method_block_residual(
          dt=dt,
          runtime_params_t_plus_dt=runtime_params_theta05,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          core_profiles_t_plus_dt=dataclasses.replace(
              core_profiles_t_plus_dt,
              T_i=dataclasses.replace(
                  x_0, right_face_constraint=final_right_boundary
              ),
          ),
          x_new_guess_vec=x_0.value,
          physics_models=physics_models,
          explicit_source_profiles=explicit_source_profiles,
          coeffs_old=coeffs_old,
          evolving_names=evolving_names,
      )
      self.assertGreater(jnp.abs(jnp.sum(residual)), 0.0)


if __name__ == '__main__':
  absltest.main()
