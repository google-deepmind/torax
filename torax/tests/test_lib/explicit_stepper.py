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
from torax import boundary_conditions
from torax import calc_coeffs
from torax import config_slice
from torax import constants
from torax import fvm
from torax import geometry
from torax import physics
from torax import state as state_module
from torax.sources import source_profiles
from torax.stepper import stepper as stepper_lib


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
      state: state_module.State,
      geo: geometry.Geometry,
      dynamic_config_slice_t: config_slice.DynamicConfigSlice,
      dynamic_config_slice_t_plus_dt: config_slice.DynamicConfigSlice,
      static_config_slice: config_slice.StaticConfigSlice,
      dt: jax.Array,
      explicit_source_profiles: source_profiles.SourceProfiles,
  ) -> tuple[state_module.State, int, calc_coeffs.AuxOutput]:
    """Applies a time step update. See Stepper.__call__ docstring."""

    # Many variables throughout this function are capitalized based on physics
    # notational conventions rather than on Google Python style
    # pylint: disable=invalid-name

    # The explicit method is for testing purposes and
    # only implemented for ion heat.
    # Ensure that this is what the user requested.
    assert static_config_slice.ion_heat_eq
    assert not static_config_slice.el_heat_eq
    assert not static_config_slice.dens_eq
    assert not static_config_slice.current_eq

    consts = constants.CONSTANTS

    true_ni = state.ni.value * dynamic_config_slice_t.nref
    true_ni_face = state.ni.face_value() * dynamic_config_slice_t.nref

    # Transient term coefficient vectors for ion heat equation
    # (has radial dependence through r, n)
    cti = 1.5 * geo.vpr * true_ni * consts.keV2J

    # Diffusion term coefficient
    d_face_ion = (
        geo.g1_over_vpr_face
        * true_ni_face
        * consts.keV2J
        * dynamic_config_slice_t.transport.chii_const
        / geo.rmax**2
    )

    c_mat, c = fvm.diffusion_terms.make_diffusion_terms(
        d_face_ion, state.temp_ion
    )

    # Source term
    c += source_profiles.sum_sources_temp_ion(
        self.sources, explicit_source_profiles, geo
    )

    temp_ion_new = (
        state.temp_ion.value
        + dt * (jnp.dot(c_mat, state.temp_ion.value) + c) / cti
    )
    # Update the potentially time-dependent boundary conditions as well.
    updated_boundary_conditions = (
        boundary_conditions.compute_boundary_conditions(
            dynamic_config_slice_t_plus_dt,
            geo,
        )
    )
    temp_ion_new = dataclasses.replace(
        state.temp_ion,
        value=temp_ion_new,
        **updated_boundary_conditions['temp_ion'],
    )

    q_face, _ = physics.calc_q_from_jtot_psi(
        geo=geo,
        jtot_face=state.currents.jtot,
        psi=state.psi,
        Rmaj=dynamic_config_slice_t_plus_dt.Rmaj,
        q_correction_factor=dynamic_config_slice_t_plus_dt.q_correction_factor,
    )
    s_face = physics.calc_s_from_psi(geo, state.psi)

    # error isn't used for timestep adaptation for this method.
    # However, too large a timestep will lead to numerical instabilities.
    # // TODO(b/312454528) - add timestep check such that error=0 is appropriate
    error = 0

    return (
        dataclasses.replace(
            state,
            temp_ion=temp_ion_new,
            q_face=q_face,
            s_face=s_face,
        ),
        error,
        calc_coeffs.AuxOutput.build_from_geo(geo),
    )
