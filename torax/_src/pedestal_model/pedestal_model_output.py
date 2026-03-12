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

"""Output of the pedestal model."""

import dataclasses
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportMultipliers:
  """Transport multipliers for the pedestal."""

  chi_e_multiplier: array_typing.FloatScalar
  chi_i_multiplier: array_typing.FloatScalar
  D_e_multiplier: array_typing.FloatScalar
  v_e_multiplier: array_typing.FloatScalar

  @classmethod
  def default(cls):
    return cls(
        chi_e_multiplier=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        chi_i_multiplier=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        D_e_multiplier=jnp.array(1.0, dtype=jax_utils.get_dtype()),
        v_e_multiplier=jnp.array(1.0, dtype=jax_utils.get_dtype()),
    )


def _build_smoothing_matrix(
    rho_face_norm: array_typing.FloatVectorFace,
    rho_norm_ped_top: array_typing.FloatScalar,
    smoothing_width: array_typing.FloatScalar,
    n_sigma: float = 2.0,
) -> jax.Array:
  """Builds a smoothing matrix for the pedestal top."""
  # Gaussian kernel with sigma = smoothing_width.
  kernel = jnp.exp(
      -jnp.log(2)
      * (rho_face_norm[:, jnp.newaxis] - rho_face_norm) ** 2
      / (smoothing_width**2 + constants.CONSTANTS.eps)
  )
  # Smoothing matrix is only non-identity within n_sigma of the pedestal top.
  mask = jnp.abs(rho_face_norm - rho_norm_ped_top) < (n_sigma * smoothing_width)
  # Zero out restricted columns so active points don't read from them
  masked_kernel = jnp.where(mask, kernel, 0.0)
  # Replace restricted rows with identity so they are unmodified (pass-through)
  smoothing_matrix = jnp.where(
      mask[:, jnp.newaxis], masked_kernel, jnp.eye(kernel.shape[0])
  )
  # Normalize the smoothing matrix
  smoothing_matrix /= jnp.sum(smoothing_matrix, axis=1, keepdims=True)
  # Remove small values
  smoothing_matrix = jnp.where(smoothing_matrix < 1e-3, 0.0, smoothing_matrix)
  # Re-normalize
  smoothing_matrix /= jnp.sum(smoothing_matrix, axis=1, keepdims=True)
  return smoothing_matrix


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput:
  """Output of a PedestalModel.

  Attributes:
    rho_norm_ped_top: The requested location of the pedestal top in rho_norm,
      not quantized to either the cell or face grid.
    rho_norm_ped_top_idx: The nearest cell index of the pedestal top.
    T_i_ped: The ion temperature at the pedestal top in keV.
    T_e_ped: The electron temperature at the pedestal top in keV.
    n_e_ped: The electron density at the pedestal top in m^-3.
    transport_multipliers: Multipliers for the transport coefficients in the
      pedestal region. Only used if the pedestal is in ADAPTIVE_TRANSPORT mode.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  # TODO(b/434175938): Can we remove rho_norm_ped_top_idx?
  rho_norm_ped_top_idx: array_typing.IntScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  transport_multipliers: TransportMultipliers = dataclasses.field(
      default_factory=TransportMultipliers.default
  )

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions."""
    # In this case, the mask is only the pedestal top, not the whole pedestal
    # region. This is because we are adding a source/sink term only at the
    # pedestal top.
    # We are using the cell grid here, since internal boundary conditions are
    # applied using an adaptive source (which acts on the cell grid).
    pedestal_mask = (
        jnp.zeros_like(geo.rho, dtype=bool)
        .at[self.rho_norm_ped_top_idx]
        .set(True)
    )
    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
        T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
        n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
    )

  def modify_core_transport(
      self,
      core_transport: state.CoreTransport,
      geo: geometry.Geometry,
      pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
  ) -> state.CoreTransport:
    """Modify transport coefficients in the entire pedestal region.

    Scales the turbulent and Pereverzev transport coefficients in the pedestal
    region by the multipliers in the pedestal model output. This will also scale
    any components of the transport coefficients that are inherited from the
    turbulent model, such as ITG, ETG, TEM, Bohm, GyroBohm, etc. Transport
    coefficients from neoclassical and pedestal transport models are not
    affected.

    Args:
      core_transport: The core transport coefficients to modify.
      geo: The geometry of the torus.
      pedestal_runtime_params: The runtime parameters of the pedestal model.

    Returns:
      The modified core transport coefficients.
    """
    # We are using the face grid here, since transport coefficients are
    # applied on the face grid.

    # TODO(b/485147781):  In the case where we have a CombinedTransportModel
    # with a pedestal transport model specified, we are currently scaling
    # all the coefficients in the pedestal region, whereas we should be only
    # scaling the turbulent coeffs and leaving the pedestal coeffs alone.
    pedestal_active_mask_face = geo.rho_face_norm > self.rho_norm_ped_top

    smoothing_matrix = _build_smoothing_matrix(
        geo.rho_face_norm,
        self.rho_norm_ped_top,
        pedestal_runtime_params.pedestal_top_smoothing_width,
    )

    def multiply_coeff(
        path: jax.tree_util.KeyPath, coeff: array_typing.FloatVectorFace
    ) -> array_typing.FloatVectorFace:
      """Scale turbulent+Pereverzev transport coefficients in the pedestal."""
      # Get the variable name of the leaf
      key = str(path[-1])

      # Apply the correct multiplier based on the variable name
      # TODO(b/488314338): Improve robustness of applying multipliers to
      # transport coefficients, ideally avoiding string matching.
      if "neo" in key:
        # Neoclassical transport should not be affected by scaling from an
        # ADAPTIVE_TRANSPORT pedestal model.
        return coeff
      elif "chi_face_ion" in key:
        # If transport suppression is not in effect, perform no scaling
        # (L-mode). If transport suppression is in effect (i.e. H-mode,
        # chi_i_multiplier != 1.0), then we clip the chi before scaling, to
        # avoid unrealistic values.
        modified_coeff = jnp.where(
            jnp.isclose(self.transport_multipliers.chi_i_multiplier, 1.0),
            coeff,
            jnp.clip(coeff, max=pedestal_runtime_params.chi_max)
            * self.transport_multipliers.chi_i_multiplier,
        )
      elif "chi_face_el" in key:
        modified_coeff = jnp.where(
            jnp.isclose(self.transport_multipliers.chi_e_multiplier, 1.0),
            coeff,
            jnp.clip(coeff, max=pedestal_runtime_params.chi_max)
            * self.transport_multipliers.chi_e_multiplier,
        )
      elif "d_face_el" in key:
        modified_coeff = jnp.where(
            jnp.isclose(self.transport_multipliers.D_e_multiplier, 1.0),
            coeff,
            jnp.clip(coeff, max=pedestal_runtime_params.D_e_max)
            * self.transport_multipliers.D_e_multiplier,
        )
      elif "v_face_el" in key:
        modified_coeff = jnp.where(
            jnp.isclose(self.transport_multipliers.v_e_multiplier, 1.0),
            coeff,
            jnp.clip(
                coeff,
                min=pedestal_runtime_params.V_e_min,
                max=pedestal_runtime_params.V_e_max,
            )
            * self.transport_multipliers.v_e_multiplier,
        )
      else:
        return coeff

      # Only modify the coefficients in the pedestal region.
      modified_coeff = jnp.where(
          pedestal_active_mask_face, modified_coeff, coeff
      )

      # Apply smoothing to the pedestal top
      modified_coeff = jnp.dot(smoothing_matrix, modified_coeff)

      return modified_coeff

    return jax.tree_util.tree_map_with_path(multiply_coeff, core_transport)
