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

"""The ExplicitStepper class.

The explicit stepper is not intended to perform well; it is included only for
testing purposes. The implementation is intentionally flat with relatively
few configuration options, etc., to ensure reliability for testing purposes.
"""
import dataclasses

import jax
from jax import numpy as jnp
from torax import constants
from torax import core_profile_setters
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import diffusion_terms
from torax.geometry import geometry
from torax.sources import source_operations
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.stepper import stepper as stepper_lib
from torax.transport_model import constant as constant_transport_model


class ExplicitStepper(stepper_lib.Stepper):
  """Explicit time step update.

  Coefficients of the various terms are computed at each timestep even though it
  is not strictly necessary (may be in the future if these are time varying).

  Current implementation has no ion-electron heat exchange, only constant chi,
  and simulates the evolution of temp_ion only.

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
      state.CoreTransport,
      state.StepperNumericOutputs,
  ]:
    """Applies a time step update. See Stepper.__call__ docstring."""

    # Many variables throughout this function are capitalized based on physics
    # notational conventions rather than on Google Python style
    # pylint: disable=invalid-name

    # The explicit method is for testing purposes and
    # only implemented for ion heat.
    # Ensure that this is what the user requested.
    assert static_runtime_params_slice.ion_heat_eq
    assert not static_runtime_params_slice.el_heat_eq
    assert not static_runtime_params_slice.dens_eq
    assert not static_runtime_params_slice.current_eq

    consts = constants.CONSTANTS

    nref = dynamic_runtime_params_slice_t.numerics.nref
    true_ni = core_profiles_t.ni.value * nref
    true_ni_face = core_profiles_t.ni.face_value() * nref

    # Transient term coefficient vectors for ion heat equation
    # (has radial dependence through r, n)
    cti = 1.5 * geo_t.vpr * true_ni * consts.keV2J

    # Diffusion term coefficient
    assert isinstance(
        dynamic_runtime_params_slice_t.transport,
        constant_transport_model.DynamicRuntimeParams,
    )
    d_face_ion = (
        geo_t.g1_over_vpr_face
        * true_ni_face
        * consts.keV2J
        * dynamic_runtime_params_slice_t.transport.chii_const
    )

    c_mat, c = diffusion_terms.make_diffusion_terms(
        d_face_ion, core_profiles_t.temp_ion
    )

    # Source term
    c += source_operations.sum_sources_temp_ion(
        geo_t,
        explicit_source_profiles,
    )

    temp_ion_new = (
        core_profiles_t.temp_ion.value
        + dt * (jnp.dot(c_mat, core_profiles_t.temp_ion.value) + c) / cti
    )
    # Update the potentially time-dependent boundary conditions as well.
    updated_boundary_conditions = core_profile_setters.compute_boundary_conditions_for_t_plus_dt(
        dt=dynamic_runtime_params_slice_t_plus_dt.numerics.fixed_dt,
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice_t=dynamic_runtime_params_slice_t_plus_dt,
        dynamic_runtime_params_slice_t_plus_dt=dynamic_runtime_params_slice_t_plus_dt,
        geo_t_plus_dt=geo_t_plus_dt,
        core_profiles_t=core_profiles_t,
    )
    temp_ion_new = dataclasses.replace(
        core_profiles_t.temp_ion,
        value=temp_ion_new,
        **updated_boundary_conditions['temp_ion'],
    )

    q_face, _ = physics.calc_q_from_psi(
        geo=geo_t,
        psi=core_profiles_t.psi,
        q_correction_factor=dynamic_runtime_params_slice_t.numerics.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo_t, core_profiles_t.psi)

    # error isn't used for timestep adaptation for this method.
    # However, too large a timestep will lead to numerical instabilities.
    # // TODO(b/312454528) - add timestep check such that error=0 is appropriate
    stepper_numeric_outputs = state.StepperNumericOutputs(
        outer_stepper_iterations=1,
        stepper_error_state=0,
        inner_solver_iterations=1,
    )

    return (
        dataclasses.replace(
            core_profiles_t,
            temp_ion=temp_ion_new,
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
        ),
        state.CoreTransport.zeros(geo_t),
        stepper_numeric_outputs,
    )


@dataclasses.dataclass(kw_only=True)
class ExplicitStepperBuilder(stepper_lib.StepperBuilder):
  """Builds an ExplicitStepper."""

  def __call__(
      self, transport_model, sources, pedestal_model
  ) -> ExplicitStepper:
    return ExplicitStepper(transport_model, sources, pedestal_model)
