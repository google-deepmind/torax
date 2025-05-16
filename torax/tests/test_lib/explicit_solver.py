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

"""The ExplicitSolver class.

The explicit solver is not intended to perform well; it is included only for
testing purposes. The implementation is intentionally flat with relatively
few configuration options, etc., to ensure reliability for testing purposes.
"""

import dataclasses
from typing import Literal

import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import updaters
from torax._src.fvm import diffusion_terms
from torax._src.geometry import geometry
from torax._src.neoclassical.conductivity import base as conductivity_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles
from torax.stepper import linear_theta_method
from torax.stepper import pydantic_model as solver_pydantic_model
from torax.transport_model import constant as constant_transport_model


class ExplicitSolver(linear_theta_method.LinearThetaMethod):
  """Explicit time step update.

  Coefficients of the various terms are computed at each timestep even though it
  is not strictly necessary (may be in the future if these are time varying).

  Current implementation has no ion-electron heat exchange, only constant chi,
  and simulates the evolution of T_i only.

  Ghost cell formulation for RHS BC.
  """

  def __call__(
      self,
      dt: jax.Array,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      dynamic_runtime_params_slice_t: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_runtime_params_slice_t_plus_dt: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo_t: geometry.Geometry,
      geo_t_plus_dt: geometry.Geometry,
      core_profiles_t: state.CoreProfiles,
      core_profiles_t_plus_dt: state.CoreProfiles,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[
      state.CoreProfiles,
      source_profiles.SourceProfiles,
      conductivity_base.Conductivity,
      state.CoreTransport,
      state.SolverNumericOutputs,
  ]:
    """Applies a time step update. See Solver.__call__ docstring."""

    # pylint: disable=invalid-name

    # The explicit method is for testing purposes and
    # only implemented for ion heat.
    # Ensure that this is what the user requested.
    assert static_runtime_params_slice.evolve_ion_heat
    assert not static_runtime_params_slice.evolve_electron_heat
    assert not static_runtime_params_slice.evolve_density
    assert not static_runtime_params_slice.evolve_current

    consts = constants.CONSTANTS

    density_reference = (
        dynamic_runtime_params_slice_t.numerics.density_reference
    )
    true_n_i = core_profiles_t.n_i.value * density_reference
    true_n_i_face = core_profiles_t.n_i.face_value() * density_reference

    # Transient term coefficient vectors for ion heat equation
    # (has radial dependence through r, n)
    cti = 1.5 * geo_t.vpr * true_n_i * consts.keV2J

    # Diffusion term coefficient
    assert isinstance(
        dynamic_runtime_params_slice_t.transport,
        constant_transport_model.DynamicRuntimeParams,
    )
    d_face_ion = (
        geo_t.g1_over_vpr_face
        * true_n_i_face
        * consts.keV2J
        * dynamic_runtime_params_slice_t.transport.chi_i
    )

    c_mat, c = diffusion_terms.make_diffusion_terms(
        d_face_ion, core_profiles_t.T_i
    )

    # Source term
    c += explicit_source_profiles.total_sources('T_i', geo_t)

    T_i_new = (
        core_profiles_t.T_i.value
        + dt * (jnp.dot(c_mat, core_profiles_t.T_i.value) + c) / cti
    )
    # Update the potentially time-dependent boundary conditions as well.
    updated_boundary_conditions = updaters.compute_boundary_conditions_for_t_plus_dt(
        dt=dynamic_runtime_params_slice_t_plus_dt.numerics.fixed_dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t_plus_dt,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )
    T_i_new = dataclasses.replace(
        core_profiles_t.T_i,
        value=T_i_new,
        **updated_boundary_conditions['T_i'],
    )

    q_face = psi_calculations.calc_q_face(geo_t, core_profiles_t.psi)
    s_face = psi_calculations.calc_s_face(geo_t, core_profiles_t.psi)

    # error isn't used for timestep adaptation for this method.
    # However, too large a timestep will lead to numerical instabilities.
    # // TODO(b/312454528) - add timestep check such that error=0 is appropriate
    solver_numeric_outputs = state.SolverNumericOutputs(
        outer_solver_iterations=1,
        solver_error_state=0,
        inner_solver_iterations=1,
    )
    conductivity = self.source_models.conductivity.calculate_conductivity(
        dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt,
        core_profiles_t_plus_dt,
    )

    return (
        dataclasses.replace(
            core_profiles_t,
            T_i=T_i_new,
            q_face=q_face,
            s_face=s_face,
        ),
        source_profile_builders.build_source_profiles(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
            static_runtime_params_slice=static_runtime_params_slice,
            geo=geo_t,
            core_profiles=core_profiles_t,
            source_models=self.source_models,
            explicit=False,
            explicit_source_profiles=explicit_source_profiles,
            conductivity=conductivity,
        ),
        conductivity,
        state.CoreTransport.zeros(geo_t),
        solver_numeric_outputs,
    )


class ExplicitSolverConfig(solver_pydantic_model.LinearThetaMethod):
  """Fake solver config that allows us to hook into the error logic."""

  solver_type: Literal['explicit'] = 'explicit'

  def build_solver(
      self,
      static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
      transport_model,
      source_models,
      pedestal_model,
  ) -> 'ExplicitSolver':
    return ExplicitSolver(
        static_runtime_params_slice=static_runtime_params_slice,
        transport_model=transport_model,
        source_models=source_models,
        pedestal_model=pedestal_model,
    )
