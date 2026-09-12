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

"""L-mode edge kinetic profile model parameterized by beta_poloidal_prime."""

import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import internal_boundary_conditions
from torax._src.internal_boundary_conditions import runtime_params as ibc_runtime_params
from torax._src.physics import psi_calculations

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(ibc_runtime_params.RuntimeParams):
  """Runtime parameters for the beta_poloidal_prime IBC model.

  Attributes:
    rho_norm_edge: Normalized radial coordinate (rho_norm) where the edge model
      starts.
    n_e_edge: Target electron density at rho_norm_edge. In units of m^-3 if
      n_e_is_fGW is False, or in Greenwald fraction if n_e_is_fGW is True.
    beta_poloidal_prime: Critical poloidal beta gradient with respect to
      normalized poloidal flux, -d(beta_pol) / d(psi_norm), in the edge region.
    Ti_Te_ratio: Ratio of ion to electron temperature (T_i / T_e) in the edge.
    n_e_is_fGW: Whether n_e_edge is provided in units of Greenwald fraction.
  """

  rho_norm_edge: array_typing.FloatScalar
  n_e_edge: array_typing.FloatScalar
  beta_poloidal_prime: array_typing.FloatScalar
  Ti_Te_ratio: array_typing.FloatScalar
  n_e_is_fGW: array_typing.BoolScalar = False


@dataclasses.dataclass(frozen=True, eq=False)
class BetaPoloidalPrimeIBCModel(base_model.InternalBoundaryConditionModel):
  """L-mode edge profile model parameterized by critical beta_poloidal_prime."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> internal_boundary_conditions.InternalBoundaryConditions:
    """Evaluates edge kinetic profiles (T_e, T_i, n_e) from beta_poloidal_prime.

    The model parameterizes L-mode edge kinetic profiles by assuming a
    critical poloidal beta gradient with respect to normalized poloidal flux:
      beta_pol' = -d(beta_pol) / d(psi_norm)

    Integrating inward from the separatrix (where 1 - psi_norm = 0):
      beta_pol(psi_norm) = beta_pol_sep + beta_pol' * (1 - psi_norm)

    The local total pressure is reconstructed from beta_pol and local B_pol^2:
      p_total = beta_pol * B_pol^2 / (2 * mu_0)

    After subtracting the fast ion pressure to isolate the thermal pressure,
    the electron and ion temperature profiles are derived using the prescribed
    density profile and Ti/Te ratio.

    Args:
      runtime_params: Runtime parameters containing the beta_poloidal_prime IBC
        configuration.
      geo: Magnetic geometry of the torus.
      core_profiles: Core plasma state profiles.

    Returns:
      Active InternalBoundaryConditions with T_e, T_i, and n_e profiles masked
      to the edge region (geo.rho_norm >= params.rho_norm_edge).
    """
    params = runtime_params.profile_conditions.internal_boundary_conditions
    assert isinstance(params, RuntimeParams), (
        'Expected params to be beta_poloidal_prime.RuntimeParams, got'
        f' {type(params)}.'
    )

    # 1. Poloidal flux coordinates: compute normalized flux psi_norm in [0, 1].
    psi_face = core_profiles.psi.face_value()
    psi_axis = psi_face[0]
    psi_sep = psi_face[-1]
    delta_psi = psi_sep - psi_axis
    psi_norm_cell = (core_profiles.psi.value - psi_axis) / delta_psi

    # Map the edge model boundary location rho_norm_edge to psi_norm_edge.
    psi_edge = jnp.interp(params.rho_norm_edge, geo.rho_face_norm, psi_face)
    psi_norm_edge = (psi_edge - psi_axis) / delta_psi

    # Mask defining where the edge IBC model is active.
    edge_mask = geo.rho_norm >= params.rho_norm_edge

    # 2. Electron density profile:
    # Convert n_e_edge from Greenwald fraction to m^-3 if needed.
    # Ip in MA, a_minor in m, nGW in m^-3.
    nGW = (
        runtime_params.profile_conditions.Ip
        / 1e6  # Convert to MA.
        / (jnp.pi * geo.a_minor**2)
        * 1e20
    )
    n_e_edge = jnp.where(
        params.n_e_is_fGW,
        params.n_e_edge * nGW,
        params.n_e_edge,
    )

    # Linearly interpolate n_e in normalized poloidal flux between the edge
    # boundary n_e_edge (at psi_norm_edge) and the separatrix value n_e_sep
    # (at psi_norm = 1.0).
    n_e_sep = core_profiles.n_e.right_face_value
    edge_flux_frac = jnp.clip(
        (psi_norm_cell - psi_norm_edge)
        / (1.0 - psi_norm_edge + constants.CONSTANTS.eps),
        0.0,
        1.0,
    )
    n_e_edge_profile = n_e_edge + (n_e_sep - n_e_edge) * edge_flux_frac
    n_e_target = jnp.where(edge_mask, n_e_edge_profile, 0.0)

    # 3. Critical poloidal beta gradient and total pressure:
    bpol2_face = psi_calculations.calc_bpol_squared(geo, core_profiles.psi)
    bpol2_cell = geometry.face_to_cell(bpol2_face)
    bpol2_sep = bpol2_face[-1]
    p_total_sep = core_profiles.pressure_total.right_face_value
    beta_pol_sep = p_total_sep / (
        bpol2_sep / (2.0 * constants.CONSTANTS.mu_0) + constants.CONSTANTS.eps
    )

    beta_pol_local_cell = beta_pol_sep + params.beta_poloidal_prime * (
        1.0 - psi_norm_cell
    )
    # Reconstruct total pressure: p_total = beta_pol * B_pol^2 / (2 * mu_0).
    p_total_cell = beta_pol_local_cell * (
        bpol2_cell / (2.0 * constants.CONSTANTS.mu_0) + constants.CONSTANTS.eps
    )

    # 4. Thermal pressure:
    # Subtract fast ion pressure to isolate the thermal component
    p_fast_cell = core_profiles.pressure_fast_i.value
    p_thermal_cell = p_total_cell - p_fast_cell

    # 5. Temperature profiles (T_e, T_i):
    # Total thermal pressure from electrons, ions, and impurities:
    #   p_thermal = n_e * T_e + (n_i + n_imp) * T_i
    # Assuming T_i = T_imp and defining n_i_n_e_ratio = (n_i + n_imp) / n_e:
    #   p_thermal = n_e * T_e * (1 + (T_i / T_e) * n_i_n_e_ratio)
    n_i_n_e_ratio = (
        core_profiles.n_i.value + core_profiles.n_impurity_thermal.value
    ) / (core_profiles.n_e.value + constants.CONSTANTS.eps)

    T_e_edge_profile = p_thermal_cell / (
        constants.CONSTANTS.keV_to_J
        * n_e_edge_profile
        * (1.0 + params.Ti_Te_ratio * n_i_n_e_ratio)
        + constants.CONSTANTS.eps
    )
    T_i_edge_profile = params.Ti_Te_ratio * T_e_edge_profile

    # Apply edge mask so only the edge region is constrained by the IBC.
    T_e_target = jnp.where(edge_mask, T_e_edge_profile, 0.0)
    T_i_target = jnp.where(edge_mask, T_i_edge_profile, 0.0)

    return internal_boundary_conditions.InternalBoundaryConditions(
        T_i=T_i_target,
        T_e=T_e_target,
        n_e=n_e_target,
    )
