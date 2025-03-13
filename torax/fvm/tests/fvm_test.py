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
from torax.config import build_runtime_params
from torax.config import numerics as numerics_lib
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.fvm import block_1d_coeffs
from torax.fvm import calc_coeffs
from torax.fvm import cell_variable
from torax.fvm import implicit_solve_block
from torax.fvm import residual_and_loss
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.pedestal_model import set_tped_nped
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import runtime_params as source_runtime_params
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import torax_refs
from torax.transport_model import constant as constant_transport_model


class FVMTest(torax_refs.ReferenceValueTest):

  @parameterized.parameters([
      dict(num_cells=2, theta_imp=0, time_steps=29),
      dict(num_cells=3, theta_imp=0.5, time_steps=21),
      dict(num_cells=4, theta_imp=1.0, time_steps=34),
  ])
  def test_leftward_convection(self, num_cells, theta_imp, time_steps):
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
    dt = jnp.array(1.0 - 0.5 * (theta_imp == 0))
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
          theta_imp=theta_imp,
      )

    np.testing.assert_allclose(x[0].value, right_boundary[0])
    np.testing.assert_allclose(x[1].value, right_boundary[1])

  @parameterized.parameters([
      dict(theta_imp=0.0),
      dict(theta_imp=0.5),
      dict(theta_imp=1.0),
  ])
  def test_implicit_source_cross(self, theta_imp):
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
    # Mapping this onto the terminology of `stepper.implicit_solve_block`, we
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
        'theta_imp': theta_imp,
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

      if theta_imp == 0.0:
        # For explicit method, the source terms are applied at t=0, when
        # u[start] == 0. So they should have no effect
        np.testing.assert_allclose(x[1 - start].value, 0.0)
      else:
        # By t=1, u[start] is greater than 0, and the implicit source terms
        # should also drive u[1 - start] to be greater than 0
        self.assertGreater(x[1 - start].value.min(), 0.0)

  @parameterized.parameters([
      dict(num_cells=3, theta_imp=0, time_steps=29),
      dict(num_cells=4, theta_imp=0.5, time_steps=21),
      dict(num_cells=5, theta_imp=1.0, time_steps=34),
  ])
  def test_nonlinear_solve_block_loss_minimum(
      self, num_cells, theta_imp, time_steps
  ):
    """Tests that the linear solution for a linear problem yields zero residual and loss."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=False,
        ),
        numerics=numerics_lib.Numerics(
            el_heat_eq=False,
        ),
    )
    stepper_params = stepper_pydantic_model.Stepper.from_dict(
        dict(
            predictor_corrector=False,
            theta_imp=theta_imp,
        )
    )
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cells
    ).build_geometry()
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                chimin=0,
                chii_const=1,
            ),
        )
    )
    pedestal = pedestal_pydantic_model.Pedestal()
    pedestal_model = pedestal.build_pedestal_model()
    transport_model = transport_model_builder()
    sources = default_sources.get_default_sources()
    sources_dict = sources.to_dict()
    sources_dict = sources_dict['source_model_config']
    sources_dict['qei_source']['Qei_mult'] = 0.0
    sources_dict['generic_ion_el_heat_source']['Ptot'] = 0.0
    sources_dict['fusion_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources_dict['ohmic_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources = sources_pydantic_model.Sources.from_dict(sources_dict)
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            stepper=stepper_params,
            pedestal=pedestal,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            torax_mesh=geo.torax_mesh,
            sources=sources,
            stepper=stepper_params,
        )
    )
    core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    evolving_names = tuple(['temp_ion'])
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        source_models=source_models,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        explicit=True,
    )
    coeffs = calc_coeffs.calc_coeffs(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=core_profiles,
        transport_model=transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=source_models,
        pedestal_model=pedestal_model,
        evolving_names=evolving_names,
        use_pereverzev=False,
    )
    # dt well under the explicit stability limit for dx=1 and chi=1
    dt = jnp.array(0.2)
    # initialize x_new for timestepping
    x_new = (core_profiles.temp_ion,)
    for _ in range(time_steps):
      x_old = copy.deepcopy(x_new)
      x_new = implicit_solve_block.implicit_solve_block(
          dt=dt,
          x_old=x_old,
          x_new_guess=x_new,
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          theta_imp=theta_imp,
      )

      # When the coefficients are kept constant, the loss
      # should just be a quadratic bowl with the linear
      # solution as the minimum with approximately zero residual
      # core_profiles_t_plus_dt is not updated since coeffs stay constant here
      loss, _ = residual_and_loss.theta_method_block_loss(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
          geo_t_plus_dt=geo,
          x_old=x_old,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          core_profiles_t_plus_dt=core_profiles,
          transport_model=transport_model,
          explicit_source_profiles=explicit_source_profiles,
          source_models=source_models,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
          pedestal_model=pedestal_model,
      )

      residual, _ = residual_and_loss.theta_method_block_residual(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
          geo_t_plus_dt=geo,
          x_new_guess_vec=jnp.concatenate([var.value for var in x_new]),
          x_old=x_old,
          core_profiles_t_plus_dt=core_profiles,
          transport_model=transport_model,
          explicit_source_profiles=explicit_source_profiles,
          source_models=source_models,
          coeffs_old=coeffs,
          evolving_names=evolving_names,
          pedestal_model=pedestal_model,
      )

      np.testing.assert_allclose(loss, 0.0, atol=1e-7)
      np.testing.assert_allclose(residual, 0.0, atol=1e-7)

  def test_implicit_solve_block_uses_updated_boundary_conditions(self):
    """Tests that updated boundary conditions affect x_new."""
    # Create a system with diffusive transport and no sources. When initialized
    # flat, x_new should remain zero unless boundary conditions change.
    num_cells = 3
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=False,
        ),
        numerics=numerics_lib.Numerics(
            el_heat_eq=False,
        ),
    )
    stepper_params = stepper_pydantic_model.Stepper.from_dict(
        dict(
            predictor_corrector=False,
            theta_imp=1.0,
        )
    )
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                chimin=0,
                chii_const=1,
            ),
        )
    )
    transport_model = transport_model_builder()
    sources = default_sources.get_default_sources()
    sources_dict = sources.to_dict()
    sources_dict = sources_dict['source_model_config']
    sources_dict['qei_source']['Qei_mult'] = 0.0
    sources_dict['generic_ion_el_heat_source']['Ptot'] = 0.0
    sources_dict['fusion_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources_dict['ohmic_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources = sources_pydantic_model.Sources.from_dict(sources_dict)
    pedestal = pedestal_pydantic_model.Pedestal()
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cells
    ).build_geometry()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            stepper=stepper_params,
            pedestal=pedestal,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            torax_mesh=geo.torax_mesh,
            sources=sources,
            stepper=stepper_params,
        )
    )
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cells
    ).build_geometry()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    initial_core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=source_models,
        explicit=True,
    )

    dt = jnp.array(1.0)
    evolving_names = tuple(['temp_ion'])
    pedestal_model = pedestal.build_pedestal_model()

    coeffs = calc_coeffs.calc_coeffs(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=initial_core_profiles,
        transport_model=transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=source_models,
        pedestal_model=pedestal_model,
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
    # Run with different theta_imp values.
    for theta_imp in [0.0, 0.5, 1.0]:
      x_new = implicit_solve_block.implicit_solve_block(
          dt=dt,
          x_old=(x_0,),
          x_new_guess=(x_0,),
          coeffs_old=coeffs,
          # Assume no time-dependent params.
          coeffs_new=coeffs,
          theta_imp=theta_imp,
      )
      # No matter what theta_imp is used, the x_new will be all 0s because there
      # is no source and the boundaries are set to 0.
      np.testing.assert_allclose(x_new[0].value, 0.0)

    # If we run with an updated boundary condition applied at time t=dt, then
    # we should get non-zero values from the implicit terms.
    final_right_boundary = jnp.array(1.0)
    x_1 = dataclasses.replace(x_0, right_face_constraint=final_right_boundary)
    # However, the explicit terms (when theta_imp = 0), should still be all 0.
    x_new = implicit_solve_block.implicit_solve_block(
        dt=dt,
        x_old=(x_0,),
        x_new_guess=(x_1,),
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_new=coeffs,
        theta_imp=0.0,
    )
    np.testing.assert_allclose(x_new[0].value, 0.0)
    # x_new should still have the updated boundary conditions though.
    np.testing.assert_allclose(
        x_new[0].right_face_constraint, final_right_boundary
    )
    # And when theta_imp is > 0, the values should be > 0.
    x_new = implicit_solve_block.implicit_solve_block(
        dt=dt,
        x_old=(x_0,),
        x_new_guess=(x_1,),
        coeffs_old=coeffs,
        # Assume no time-dependent params.
        coeffs_new=coeffs,
        theta_imp=0.5,
    )
    self.assertGreater(x_new[0].value.min(), 0.0)

  def test_theta_residual_uses_updated_boundary_conditions(self):
    # Create a system with diffusive transport and no sources. When initialized
    # flat, residual should remain zero unless boundary conditions change.
    num_cells = 3
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=False,
        ),
        numerics=numerics_lib.Numerics(
            el_heat_eq=False,
        ),
    )
    stepper_params = stepper_pydantic_model.Stepper.from_dict(
        dict(
            predictor_corrector=False,
            theta_imp=0.0,
        )
    )
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=num_cells
    ).build_geometry()
    transport_model_builder = (
        constant_transport_model.ConstantTransportModelBuilder(
            runtime_params=constant_transport_model.RuntimeParams(
                chimin=0,
                chii_const=1,
            ),
        )
    )
    transport_model = transport_model_builder()
    pedestal = pedestal_pydantic_model.Pedestal()
    sources = default_sources.get_default_sources()
    sources_dict = sources.to_dict()
    sources_dict = sources_dict['source_model_config']
    sources_dict['qei_source']['Qei_mult'] = 0.0
    sources_dict['generic_ion_el_heat_source']['Ptot'] = 0.0
    sources_dict['fusion_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources_dict['ohmic_heat_source']['mode'] = source_runtime_params.Mode.ZERO
    sources = sources_pydantic_model.Sources.from_dict(sources_dict)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            stepper=stepper_params,
            pedestal=pedestal,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )
    static_runtime_params_slice_theta0 = (
        build_runtime_params.build_static_runtime_params_slice(
            runtime_params=runtime_params,
            torax_mesh=geo.torax_mesh,
            sources=sources,
            stepper=stepper_params,
        )
    )
    static_runtime_params_slice_theta05 = dataclasses.replace(
        static_runtime_params_slice_theta0,
        stepper=dataclasses.replace(
            static_runtime_params_slice_theta0.stepper, theta_imp=0.5
        ),
    )

    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    pedestal_model = set_tped_nped.SetTemperatureDensityPedestalModel()
    initial_core_profiles = initialization.initial_core_profiles(
        static_runtime_params_slice_theta0,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    explicit_source_profiles = source_profile_builders.build_source_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice_theta0,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=source_models,
        explicit=True,
    )

    dt = jnp.array(1.0)
    evolving_names = tuple(['temp_ion'])

    coeffs_old = calc_coeffs.calc_coeffs(
        static_runtime_params_slice=static_runtime_params_slice_theta05,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=initial_core_profiles,
        transport_model=transport_model,
        explicit_source_profiles=explicit_source_profiles,
        source_models=source_models,
        pedestal_model=pedestal_model,
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
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_runtime_params_slice_theta05,
        geo=geo,
        source_models=source_models,
    )
    core_profiles_t_plus_dt = dataclasses.replace(
        core_profiles_t_plus_dt,
        temp_ion=x_0,
    )

    with self.subTest('static_boundary_conditions'):
      # When the boundary conditions are not time-dependent and stay at 0,
      # with diffusive transport and zero transport, then the state will stay
      # at all 0, and the residual should be 0.
      residual, _ = residual_and_loss.theta_method_block_residual(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice_theta05,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          x_new_guess_vec=x_0.value,
          core_profiles_t_plus_dt=core_profiles_t_plus_dt,
          transport_model=transport_model,
          explicit_source_profiles=explicit_source_profiles,
          source_models=source_models,
          coeffs_old=coeffs_old,
          evolving_names=evolving_names,
          pedestal_model=pedestal_model,
      )
      np.testing.assert_allclose(residual, 0.0)
    with self.subTest('updated_boundary_conditions'):
      # When the boundary condition updates at time t+dt, then the implicit part
      # of the update would generate a residual. When theta_imp is 0, the
      # residual would still be 0.
      final_right_boundary = jnp.array(1.0)
      residual, _ = residual_and_loss.theta_method_block_residual(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice_theta0,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          x_new_guess_vec=x_0.value,
          core_profiles_t_plus_dt=dataclasses.replace(
              core_profiles_t_plus_dt,
              temp_ion=dataclasses.replace(
                  x_0, right_face_constraint=final_right_boundary
              ),
          ),
          evolving_names=evolving_names,
          transport_model=transport_model,
          explicit_source_profiles=explicit_source_profiles,
          source_models=source_models,
          coeffs_old=coeffs_old,
          pedestal_model=pedestal_model,
      )
      np.testing.assert_allclose(residual, 0.0)
      # But when theta_imp > 0, the residual should be non-zero.
      residual, _ = residual_and_loss.theta_method_block_residual(
          dt=dt,
          static_runtime_params_slice=static_runtime_params_slice_theta05,
          dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice,
          geo_t_plus_dt=geo,
          x_old=(x_0,),
          core_profiles_t_plus_dt=dataclasses.replace(
              core_profiles_t_plus_dt,
              temp_ion=dataclasses.replace(
                  x_0, right_face_constraint=final_right_boundary
              ),
          ),
          x_new_guess_vec=x_0.value,
          transport_model=transport_model,
          explicit_source_profiles=explicit_source_profiles,
          source_models=source_models,
          coeffs_old=coeffs_old,
          evolving_names=evolving_names,
          pedestal_model=pedestal_model,
      )
      self.assertGreater(jnp.abs(jnp.sum(residual)), 0.0)


if __name__ == '__main__':
  absltest.main()
