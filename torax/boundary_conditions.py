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

"""Utilities for computing and updating the boundary conditions for State."""

import jax
from jax import numpy as jnp
from torax import config_slice
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics


# Type-alias for brevity.
BoundaryConditionsMap = dict[str, dict[str, jax.Array | None]]


def compute_boundary_conditions(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
) -> BoundaryConditionsMap:
  """Computes boundary conditions for time t and returns updates to State.

  Args:
    dynamic_config_slice: Runtime configuration at time t.
    geo: Geometry object

  Returns:
    Mapping from State attribute names to dictionaries updating attributes of
    each CellVariable in the state. This dict can in theory recursively replace
    values in a State object.
  """
  Ip = dynamic_config_slice.Ip  # pylint: disable=invalid-name
  Ti_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_config_slice.Ti_bound_right, 'Ti_bound_right'
  )
  Te_bound_right = jax_utils.error_if_not_positive(  # pylint: disable=invalid-name
      dynamic_config_slice.Te_bound_right, 'Te_bound_right'
  )
  ne_bound_right = dynamic_config_slice.ne_bound_right
  # define ion profile based on (flat) Zeff and single assumed impurity
  # with Zimp. main ion limited to hydrogenic species for now.
  # Assume isotopic balance for DT fusion power. Solve for ni based on:
  # Zeff = (ni + Zimp**2 * nimp)/ne  ;  nimp*Zimp + ni = ne

  dilution_factor = physics.get_main_ion_dilution_factor(
      dynamic_config_slice.Zimp, dynamic_config_slice.Zeff
  )
  return {
      'temp_ion': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(Ti_bound_right),
      ),
      'temp_el': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(Te_bound_right),
      ),
      'ne': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(ne_bound_right),
      ),
      'ni': dict(
          left_face_grad_constraint=jnp.zeros(()),
          right_face_grad_constraint=None,
          right_face_constraint=jnp.array(ne_bound_right * dilution_factor),
      ),
      'psi': dict(
          right_face_grad_constraint=Ip
          * 1e6
          * constants.CONSTANTS.mu0
          / geo.G2_face[-1]
          * geo.rmax,
          right_face_constraint=None,
      ),
  }
