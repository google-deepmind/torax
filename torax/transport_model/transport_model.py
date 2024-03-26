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

"""The TransportModel abstract base class.

The transport model calculates various coefficients related to particle
transport.
"""
import abc
from typing import Optional
import chex
import jax
from jax import numpy as jnp
from torax import config_slice
from torax import constants
from torax import geometry
from torax import state


@chex.dataclass
class TransportCoeffs:
  """Transport coefficients.

  Attributes:
    chi_face_ion: chi for ion temperature, on faces.
    chi_face_el: chi for electron temperature, on faces.
    d_face_el: Diffusivity of electron density, on faces.
    v_face_el: Convection strength of electron density, on faces.
  """

  chi_face_ion: Optional[jax.Array]
  chi_face_el: Optional[jax.Array]
  d_face_el: Optional[jax.Array]
  v_face_el: Optional[jax.Array]

  def chi_max(
      self,
      geo: geometry.Geometry,
  ) -> jnp.ndarray:
    """Calculates the maximum value of chi.

    Args:
      geo: Geometry of the torus.

    Returns:
      chi_max: Maximum value of chi.
    """

    return jnp.maximum(
        jnp.max(self.chi_face_ion * geo.g1_over_vpr2_face),
        jnp.max(self.chi_face_el * geo.g1_over_vpr2_face),
    )


class TransportModel(abc.ABC):
  """Calculates various coefficients related to heat and particle transport."""

  def __call__(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> TransportCoeffs:
    return self.smooth_coeffs(
        geo,
        dynamic_config_slice,
        self._call_implementation(dynamic_config_slice, geo, core_profiles),
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> TransportCoeffs:
    pass

  def smooth_coeffs(
      self,
      geo: geometry.Geometry,
      dynamic_config_slice: config_slice.DynamicConfigSlice,
      transport_coeffs: TransportCoeffs,
  ) -> TransportCoeffs:
    """Gaussian smoothing of transport coefficients."""
    smoothing_matrix = build_smoothing_matrix(geo, dynamic_config_slice)
    for coeff in transport_coeffs:
      smoothed_coeff = jnp.dot(smoothing_matrix, transport_coeffs[coeff])
      setattr(transport_coeffs, coeff, smoothed_coeff)
    return transport_coeffs


def build_smoothing_matrix(
    geo: geometry.Geometry,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
) -> jax.Array:
  """Builds a smoothing matrix for the transport model.

  Uses a Gaussian kernel of HWHM defined in the transport config.

  Args:
    geo: Geometry of the torus.
    dynamic_config_slice: Input config parameters that can change without
      triggering a JAX recompilation.

  Returns:
    kernel: A smoothing matrix for convolution with the transport outputs.
  """

  # To reduce the range of the convolution, weights under lower_cutoff are
  # clipped to zero
  lower_cutoff = 0.01

  # used for eps, small number to avoid divisions by zero for sigma = 0
  consts = constants.CONSTANTS

  # 1. Kernel matrix
  kernel = jnp.exp(
      -jnp.log(2)
      * (geo.r_face_norm[:, jnp.newaxis] - geo.r_face_norm) ** 2
      / (dynamic_config_slice.transport.smoothing_sigma**2 + consts.eps)
  )

  # 2. Masking: we do not want transport coefficients calculated in pedestal
  # region or in inner and outer transport patch regions, to impact
  # transport_model calculated coefficients
  mask_outer_edge = jax.lax.cond(
      dynamic_config_slice.set_pedestal,
      lambda: dynamic_config_slice.Ped_top - consts.eps,
      lambda: 1.0,
  )

  mask_outer_edge = jax.lax.cond(
      jnp.logical_and(
          jnp.logical_not(dynamic_config_slice.set_pedestal),
          dynamic_config_slice.transport.apply_outer_patch,
      ),
      lambda: dynamic_config_slice.transport.rho_outer - consts.eps,
      lambda: mask_outer_edge,
  )

  mask_inner_edge = jax.lax.cond(
      dynamic_config_slice.transport.apply_inner_patch,
      lambda: dynamic_config_slice.transport.rho_inner + consts.eps,
      lambda: 0.0,
  )

  mask = jnp.where(
      jnp.logical_and(
          geo.r_face_norm > mask_inner_edge, geo.r_face_norm < mask_outer_edge
      ),
      1.0,
      0.0,
  )

  # remove impact of smoothing on inner and outer patch, or pedestal zone

  # first zero out all rows corresponding to grid points not to be impacted
  diag_mask = jnp.diag(mask)
  kernel = jnp.dot(diag_mask, kernel)
  # now zero out all columns corresponding to grid points not to be impacted,
  # such that they don't impact the smoothing of the other grid points
  num_rows = len(mask)
  mask_mat = jnp.tile(mask, (num_rows, 1))
  kernel *= mask_mat
  # now restore identity to the zero rows, such that smoothing is a no-op for
  # on the grid points where it shouldn't impact
  zero_row_mask = jnp.all(kernel == 0, axis=1)
  kernel = jnp.where(
      zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]), kernel
  )

  # 3. Normalization
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]

  # 4. Remove small numbers
  kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)

  # 5. Final Normalization following removal of small numbers
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]
  return kernel
