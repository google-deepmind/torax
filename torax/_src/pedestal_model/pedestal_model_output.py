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
import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import jax_utils
from torax._src import state
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

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
    T_i_ped: The ion temperature at the pedestal top in keV.
    T_e_ped: The electron temperature at the pedestal top in keV.
    n_e_ped: The electron density at the pedestal top in m^-3.
    transport_multipliers: Multipliers for the transport coefficients in the
      pedestal region. Only used if the pedestal is in ADAPTIVE_TRANSPORT mode.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar
  transport_multipliers: TransportMultipliers = dataclasses.field(
      default_factory=TransportMultipliers.default
  )

  def get_two_point_face_mask(
      self,
      geo: geometry.Geometry,
      set_pedestal: chex.Numeric = True,
  ) -> array_typing.BoolVectorFace:
    """Returns boolean face mask with True at face immediately left of pedestal top."""
    ped_cell_idx = jnp.argmin(jnp.abs(geo.rho_norm - self.rho_norm_ped_top))
    mask = jnp.zeros_like(geo.rho_face_norm, dtype=bool)
    return mask.at[ped_cell_idx].set(set_pedestal)

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles | None = None,
      pedestal_profile_form: pedestal_runtime_params_lib.PedestalProfileForm = (
          pedestal_runtime_params_lib.PedestalProfileForm.SET_AT_PED_TOP
      ),
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions.

    When pedestal_profile_form is MTANH and core_profiles is provided, generates
    an mtanh-shaped profile across the pedestal region using the formula:
      a(ψ) = a_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]
    where a denotes either T_i, T_e, or n_e, and Δ is derived from
    rho_norm_ped_top via the ψ_N(ρ) mapping:
      ψ_top = ψ_N(rho_norm_ped_top), Δ = (1 - ψ_top) / 1.5

    When pedestal_profile_form is SET_AT_PED_TOP, falls back to a single-point
    mask at the nearest cell to rho_norm_ped_top.

    Args:
      geo: Geometry object for the grid.
      core_profiles: Core profiles, needed for ψ_N mapping and separatrix values
        when using mtanh profiles.
      pedestal_profile_form: Controls the shape of the pedestal profile.

    Returns:
      Internal boundary conditions for T_i, T_e, n_e.
    """
    match pedestal_profile_form:
      case pedestal_runtime_params_lib.PedestalProfileForm.MTANH:
        if core_profiles is None:
          raise ValueError(
              "core_profiles must be provided when pedestal_profile_form"
              " is MTANH."
          )
        return self._tanh_internal_boundary_conditions(geo, core_profiles)
      case pedestal_runtime_params_lib.PedestalProfileForm.SET_AT_PED_TOP:
        # Single-point mask: pin values at the nearest cell to ped top.
        rho_norm_ped_top_idx = jnp.argmin(
            jnp.abs(geo.rho_norm - self.rho_norm_ped_top)
        )
        pedestal_mask = (
            jnp.zeros_like(geo.rho, dtype=bool)
            .at[rho_norm_ped_top_idx]
            .set(True)
        )
        return internal_boundary_conditions_lib.InternalBoundaryConditions(
            T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
            T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
            n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
        )

  def _tanh_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Compute mtanh-shaped internal boundary conditions.

    The mtanh width Δ is derived from rho_norm_ped_top using the ψ_N(ρ)
    mapping. In the mtanh geometry, the pedestal top is at ψ_top = 1 - 1.5Δ,
    so Δ = (1 - ψ_top) / 1.5.

    The profile is then:
      q(ψ) = q_sep + a₀·[tanh(1) - tanh(2(ψ - ψ_mid)/Δ)]
    where:
      ψ_mid = 1 - Δ/2 (center of the tanh)
      a₀ = (q_top - q_sep) / (tanh(1) + tanh(2))

    Values are only applied for cells at or beyond ρ_ped_top (the pedestal
    region). Core cells get 0.0 (no IBC contribution).

    Args:
      geo: Geometry object for the grid.
      core_profiles: Core profiles for ψ_N mapping and separatrix values.

    Returns:
      Internal boundary conditions with mtanh-shaped profiles for T_i, T_e, n_e.
    """
    # Get ψ_N at each cell grid point.
    psi_face = core_profiles.psi.face_value()
    psi_norm_cell = (core_profiles.psi.value - psi_face[0]) / (  # pyrefly: ignore[bad-index]
        psi_face[-1] - psi_face[0]  # pyrefly: ignore[bad-index]
    )

    # Derive Δ from rho_norm_ped_top via ψ_N mapping.
    # Use psi at the nearest cell to rho_ped_top (not interpolated) so that
    # the mtanh formula evaluates to exactly q_top at that cell.
    rho_norm_ped_top_idx = jnp.argmin(
        jnp.abs(geo.rho_norm - self.rho_norm_ped_top)
    )
    psi_top = psi_norm_cell[rho_norm_ped_top_idx]
    delta = (1.0 - psi_top) / 1.5
    psi_mid = 1.0 - delta / 2.0

    # Separatrix values from the rightmost face of core_profiles.
    T_i_sep = core_profiles.T_i.right_face_value
    T_e_sep = core_profiles.T_e.right_face_value
    n_e_sep = core_profiles.n_e.right_face_value

    # Pedestal region mask: cells at or beyond rho_norm_ped_top.
    ped_mask = geo.rho_norm >= self.rho_norm_ped_top

    def _mtanh_profile(val_top, val_sep):
      """Evaluate mtanh for one quantity."""
      tanh1 = jnp.tanh(1.0)
      tanh2 = jnp.tanh(2.0)
      a0 = (val_top - val_sep) / (tanh1 + tanh2)
      profile = val_sep + a0 * (
          tanh1 - jnp.tanh(2.0 * (psi_norm_cell - psi_mid) / delta)
      )
      return jnp.where(ped_mask, profile, 0.0)

    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=_mtanh_profile(self.T_i_ped, T_i_sep),
        T_e=_mtanh_profile(self.T_e_ped, T_e_sep),
        n_e=_mtanh_profile(self.n_e_ped, n_e_sep),
    )

  def modify_core_transport(
      self,
      core_transport: state.CoreTransport,
      geo: geometry.Geometry,
      pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
  ) -> state.CoreTransport:
    """Modify transport coefficients in the entire pedestal region.

    Scales the turbulent total and Pereverzev transport coefficients in the
    pedestal region by the multipliers in the pedestal model output. Transport
    coefficients from neoclassical, core, and pedestal transport
    models are not affected.

    Args:
      core_transport: The core transport coefficients to modify.
      geo: The geometry of the torus.
      pedestal_runtime_params: The runtime parameters of the pedestal model.

    Returns:
      The modified core transport coefficients.
    """
    # We are using the face grid here, since transport coefficients are
    # applied on the face grid.

    # TODO(b/485147781):  In the case where we have a TransportModel
    # with a pedestal transport model specified, we are currently scaling
    # all the coefficients in the pedestal region, whereas we should be only
    # scaling the turbulent coeffs and leaving the pedestal coeffs alone.
    pedestal_active_mask_face = geo.rho_face_norm > self.rho_norm_ped_top

    smoothing_matrix = _build_smoothing_matrix(
        geo.rho_face_norm,
        self.rho_norm_ped_top,
        pedestal_runtime_params.pedestal_top_smoothing_width,
    )

    def _scale_channel(
        coeff: array_typing.FloatVectorFace,
        multiplier: array_typing.FloatScalar,
        clip_min: array_typing.FloatScalar | None = None,
        clip_max: array_typing.FloatScalar | None = None,
    ) -> array_typing.FloatVectorFace:
      """Scales, clips, and smooths a single transport coefficient channel."""
      # If transport suppression is not in effect, perform no scaling (L-mode).
      # If transport suppression is in effect (i.e. H-mode, multiplier != 1.0),
      # then clip before scaling to avoid unrealistic values.
      modified = jnp.where(
          jnp.isclose(multiplier, 1.0),
          coeff,
          jnp.clip(coeff, min=clip_min, max=clip_max) * multiplier,
      )
      # Only modify the coefficients in the pedestal region.
      modified = jnp.where(pedestal_active_mask_face, modified, coeff)
      # Apply smoothing to the pedestal top.
      return jnp.dot(smoothing_matrix, modified)

    def _scale_coeffs(coeffs):
      """Scales standard transport channels using pedestal multipliers."""
      return dataclasses.replace(
          coeffs,
          chi_face_ion=_scale_channel(
              coeffs.chi_face_ion,
              self.transport_multipliers.chi_i_multiplier,
              clip_max=pedestal_runtime_params.chi_max,
          ),
          chi_face_el=_scale_channel(
              coeffs.chi_face_el,
              self.transport_multipliers.chi_e_multiplier,
              clip_max=pedestal_runtime_params.chi_max,
          ),
          d_face_el=_scale_channel(
              coeffs.d_face_el,
              self.transport_multipliers.D_e_multiplier,
              clip_max=pedestal_runtime_params.D_e_max,
          ),
          v_face_el=_scale_channel(
              coeffs.v_face_el,
              self.transport_multipliers.v_e_multiplier,
              clip_min=pedestal_runtime_params.V_e_min,
              clip_max=pedestal_runtime_params.V_e_max,
          ),
      )

    # Scale turbulent total. Core and pedestal transport
    # coefficients are preserved unscaled so raw model outputs remain
    # accessible in output trees and diagnostics.
    modified_turbulent = transport_coeffs_lib.TurbulentTransport(
        total=_scale_coeffs(core_transport.turbulent.total),
        core_coefficients=core_transport.turbulent.core_coefficients,
        pedestal_coefficients=core_transport.turbulent.pedestal_coefficients,
    )

    # Scale Pereverzev transport if present.
    if core_transport.pereverzev is not None:
      modified_pereverzev = _scale_coeffs(core_transport.pereverzev)
    else:
      modified_pereverzev = None

    # Neoclassical transport is not affected by scaling from an
    # ADAPTIVE_TRANSPORT pedestal model.
    coeffs_to_sum = [modified_turbulent.total, core_transport.neoclassical]
    if modified_pereverzev is not None:
      coeffs_to_sum.append(modified_pereverzev)
    total = transport_coeffs_lib.sum_transport_coeffs(*coeffs_to_sum)
    return state.CoreTransport(
        total=total,
        turbulent=modified_turbulent,
        neoclassical=core_transport.neoclassical,
        pereverzev=modified_pereverzev,
    )
