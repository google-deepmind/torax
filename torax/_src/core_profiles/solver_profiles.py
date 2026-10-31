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

"""Base core plasma profiles derived directly from SolverState."""

import dataclasses
import functools
from typing import Self
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import flux_profiles
from torax._src.core_profiles import ion_composition as ion_composition_lib
from torax._src.core_profiles import kinetic_profiles
from torax._src.core_profiles import solver_state as solver_state_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class SolverProfiles:
  """Core plasma profiles derived directly from SolverState.

  Contains the profiles computed algebraically from `SolverState` (along with
  runtime parameters and geometry), without depending on other TORAX physics
  models or profiles (such as sources, neoclassical, pedestal, or transport).

  Attributes:
    kinetic: Kinetic profiles (temperatures and densities).
    ions: Ion composition and charge state profiles.
    flux: Magnetic flux, safety factor, and current profiles.
    toroidal_angular_velocity: Toroidal angular velocity [rad/s].
  """

  kinetic: kinetic_profiles.KineticProfiles
  ions: ion_composition_lib.IonComposition
  flux: flux_profiles.FluxProfiles
  toroidal_angular_velocity: cell_variable.CellVariable

  @classmethod
  def from_solver_state(
      cls,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      solver_state: solver_state_lib.SolverState,
      v_loop_lcfs: array_typing.FloatScalar | None = None,
  ) -> Self:
    """Constructs purely algebraic SolverProfiles from SolverState.

    Used by the unified physics pipeline and initial state convergence.

    Args:
      runtime_params: Runtime parameters.
      geo: Geometry of the tokamak.
      solver_state: The current set of primary profiles.
      v_loop_lcfs: Loop voltage at the LCFS [V]. If None, defaults to 0.0.

    Returns:
      SolverProfiles with kinetic, ions, flux, and toroidal angular velocity.
    """
    ion_composition, n_i, n_impurity = (
        ion_composition_lib.IonComposition.from_profiles(
            runtime_params.plasma_composition,
            geo,
            solver_state.n_e,
            solver_state.T_e,
        )
    )
    flux = flux_profiles.FluxProfiles.from_psi(
        geo=geo,
        psi=solver_state.psi,
        min_rho_norm=runtime_params.numerics.min_rho_norm,
        v_loop_lcfs=v_loop_lcfs,
    )
    kinetic = kinetic_profiles.KineticProfiles(
        T_i=solver_state.T_i,
        T_e=solver_state.T_e,
        n_e=solver_state.n_e,
        n_i=n_i,
        n_impurity=n_impurity,
    )
    return cls(
        kinetic=kinetic,
        ions=ion_composition,
        flux=flux,
        toroidal_angular_velocity=solver_state.toroidal_angular_velocity,
    )

  @functools.cached_property
  def impurity_density_scaling(self) -> array_typing.FloatVectorCell:
    """Scaling factor for impurity density: n_imp_true / n_imp_eff."""
    return self.ions.Z_impurity / self.ions.charge_state_info.Z_avg

  @functools.cached_property
  def impurity_density_scaling_face(self) -> array_typing.FloatVectorFace:
    """Scaling factor for impurity density on the face grid."""
    return self.ions.Z_impurity_face / self.ions.charge_state_info_face.Z_avg

  def get_species_concentration(
      self,
      species: str,
  ) -> array_typing.FloatVectorCell:
    """Extracts species fractional concentration relative to electron density.

    Args:
      species: Symbol of the ion species (e.g., 'He3', 'D', 'T', 'Ne').

    Returns:
      Species fractional concentration relative to electron density (n_s / n_e).

    Raises:
      ValueError: If species is not found in main_ion_fractions or
        impurity_fractions.
    """
    if species in self.ions.main_ion_fractions:
      fraction = self.ions.main_ion_fractions[species]
      return jnp.divide(
          self.kinetic.n_i.value * fraction, self.kinetic.n_e.value
      )

    if species in self.ions.impurity_fractions:
      fraction = self.ions.impurity_fractions[species]
      n_imp_species = (
          fraction
          * self.kinetic.n_impurity.value
          * self.impurity_density_scaling
      )
      return jnp.divide(n_imp_species, self.kinetic.n_e.value)

    raise ValueError(
        f'Species {species} not found.'
        f' Available main ions: {tuple(self.ions.main_ion_fractions.keys())},'
        f' impurities: {tuple(self.ions.impurity_fractions.keys())}'
    )

  @functools.cached_property
  def psi_norm(self) -> array_typing.FloatVectorCell:
    """Normalized poloidal flux on the cell grid."""
    psi_face = self.flux.psi.face_value()
    return (self.flux.psi.value - psi_face[0]) / (psi_face[-1] - psi_face[0])

  @functools.cached_property
  def psi_norm_face(self) -> array_typing.FloatVectorFace:
    """Normalized poloidal flux on the face grid."""
    psi_face = self.flux.psi.face_value()
    return (psi_face - psi_face[0]) / (psi_face[-1] - psi_face[0])

  @functools.cached_property
  def pressure_thermal_e(self) -> cell_variable.CellVariable:
    """Electron thermal pressure [Pa]."""
    right_face_constraint = (
        self.kinetic.n_e.get_right_bc()
        * self.kinetic.T_e.get_right_bc()
        * constants.CONSTANTS.keV_to_J
    )
    return cell_variable.CellVariable(
        value=(
            self.kinetic.n_e.value
            * self.kinetic.T_e.value
            * constants.CONSTANTS.keV_to_J
        ),
        face_centers=self.kinetic.n_e.face_centers,
        right_face_constraint=right_face_constraint,
        right_face_grad_constraint=None,
    )

  @jax.jit
  def quasineutrality_satisfied(self) -> jax.Array:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.kinetic.n_i.value * self.ions.Z_i
        + self.kinetic.n_impurity.value * self.ions.Z_impurity,
        self.kinetic.n_e.value,
    )

  @jax.jit
  def negative_temperature_or_density(self) -> jax.Array:
    """Checks if any temperature or density is negative."""
    profiles_to_check = (
        self.kinetic.T_i,
        self.kinetic.T_e,
        self.kinetic.n_e,
        self.kinetic.n_i,
        self.kinetic.n_impurity,
        self.ions.impurity_fractions,
    )
    # Check if any profile is less than -eps
    # (allowing for numerical precision errors).
    leaves: list[jax.Array] = jax.tree.leaves(profiles_to_check)
    return jnp.any(
        jnp.stack([
            jnp.any(jnp.less(x, -float(constants.CONSTANTS.eps)))
            for x in leaves
        ])
    )

  @jax.jit
  def below_minimum_temperature(self, T_min_eV: float) -> jax.Array:
    """Return True if T_e or T_i is below the minimum temperature threshold."""
    # Convert eV -> keV since internal storage is keV.
    T_min_keV = T_min_eV / 1000.0

    is_low_te = jnp.any(self.kinetic.T_e.value < T_min_keV)
    is_low_ti = jnp.any(self.kinetic.T_i.value < T_min_keV)

    return is_low_te | is_low_ti


def calculate_n_i_total_thermal_face(
    solver_profiles: SolverProfiles,
    fast_ions: tuple[fast_ion_lib.FastIon, ...],
) -> array_typing.FloatVectorFace:
  """Total thermal ion particle density (main ions + impurities) [m^-3].

  On the face grid. The bundled impurity density (charge Z_impurity =
  <Z^2>/<Z>) is rescaled to the true impurity particle density with
  `impurity_density_scaling_face` before being summed with the main ions.

  Args:
    solver_profiles: Base core profiles.
    fast_ions: Tuple of fast ion populations.

  Returns:
    Total thermal ion particle density on face grid.
  """
  n_impurity_thermal = fast_ion_lib.calculate_n_impurity_thermal(
      solver_profiles.kinetic.n_impurity,
      solver_profiles.ions.impurity_fractions,
      fast_ions,
  )
  return (
      solver_profiles.kinetic.n_i.face_value()
      + n_impurity_thermal.face_value()
      * solver_profiles.impurity_density_scaling_face
  )


def calculate_pressure_thermal_i(
    solver_profiles: SolverProfiles,
    fast_ions: tuple[fast_ion_lib.FastIon, ...],
) -> cell_variable.CellVariable:
  """Ion thermal pressure [Pa].

  Args:
    solver_profiles: Base core profiles.
    fast_ions: Tuple of fast ion populations.

  Returns:
    Ion thermal pressure CellVariable.
  """
  n_impurity_thermal = fast_ion_lib.calculate_n_impurity_thermal(
      solver_profiles.kinetic.n_impurity,
      solver_profiles.ions.impurity_fractions,
      fast_ions,
  )
  right_face_constraint = (
      solver_profiles.kinetic.T_i.get_right_bc()
      * constants.CONSTANTS.keV_to_J
      * (
          solver_profiles.kinetic.n_i.get_right_bc()
          + n_impurity_thermal.get_right_bc()
      )
  )
  return cell_variable.CellVariable(
      value=(
          solver_profiles.kinetic.T_i.value
          * constants.CONSTANTS.keV_to_J
          * (solver_profiles.kinetic.n_i.value + n_impurity_thermal.value)
      ),
      face_centers=solver_profiles.kinetic.n_i.face_centers,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=None,
  )


def calculate_pressure_thermal_total(
    solver_profiles: SolverProfiles,
    fast_ions: tuple[fast_ion_lib.FastIon, ...],
) -> cell_variable.CellVariable:
  """Total thermal pressure [Pa].

  Args:
    solver_profiles: Base core profiles.
    fast_ions: Tuple of fast ion populations.

  Returns:
    Total thermal pressure CellVariable.
  """
  p_th_i = calculate_pressure_thermal_i(solver_profiles, fast_ions)
  right_face_constraint = (
      solver_profiles.pressure_thermal_e.get_right_bc() + p_th_i.get_right_bc()
  )
  return cell_variable.CellVariable(
      value=solver_profiles.pressure_thermal_e.value + p_th_i.value,
      face_centers=solver_profiles.pressure_thermal_e.face_centers,
      right_face_constraint=right_face_constraint,
      right_face_grad_constraint=None,
  )
