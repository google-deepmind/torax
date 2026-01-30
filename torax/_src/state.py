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

"""Classes defining the TORAX state that evolves over time."""

import dataclasses
import enum
import functools
from typing import Mapping

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import charge_states
import typing_extensions


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CoreProfiles:
  """Dataclass for holding the evolving core plasma profiles.

  This dataclass is inspired by the IMAS `core_profiles` IDS.

  Profiles are stored as `CellVariable` or JAX arrays. Array-based profiles
  are on either the cell or face grid; those on the face grid are denoted with a
  `_face` suffix.

  Attributes:
      T_i: Ion temperature [keV].
      T_e: Electron temperature [keV].
      psi: Poloidal flux [Wb].
      psidot: Time derivative of poloidal flux (loop voltage) [V].
      n_e: Electron density [m^-3].
      n_i: Main ion density [m^-3].
      n_impurity: Impurity density of bundled impurity [m^-3].
      impurity_fractions: Fractional abundances of individual impurity species.
      main_ion_fractions: Fractional abundances of individual main ion species.
      q_face: Safety factor.
      s_face: Magnetic shear.
      v_loop_lcfs: Loop voltage at LCFS (V).
      Z_i: Main ion charge on cell grid [dimensionless].
      Z_i_face: Main ion charge on face grid [dimensionless].
      A_i: Main ion mass [amu].
      # TODO(b/434175938): Remove in V2. Duplication with new charge_state_info.
      Z_impurity: Impurity charge of bundled impurity on cell grid
        [dimensionless].
      Z_impurity_face: Impurity charge of bundled impurity on face grid
        [dimensionless].
      Z_eff: Effective charge on cell grid [dimensionless].
      Z_eff_face: Effective charge on face grid [dimensionless].
      A_impurity: Impurity mass on cell grid [amu].
      A_impurity_face: Impurity mass on face grid [amu].
      sigma: Conductivity on cell grid [S/m].
      sigma_face: Conductivity on face grid [S/m].
      j_total: Total current density on the cell grid [A/m^2].
      j_total_face: Total current density on face grid [A/m^2].
      Ip_profile_face: Plasma current profile on the face grid [A].
      toroidal_angular_velocity: Toroidal angular velocity [rad/s].
      charge_state_info: Container with averaged and per-species ion charge
        state information. See `charge_states.ChargeStateInfo`. Cell grid.
      charge_state_info_face: Container with averaged and per-species ion charge
        state information. See `charge_states.ChargeStateInfo`. Face grid.
  """

  T_i: cell_variable.CellVariable
  T_e: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  psidot: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
  impurity_fractions: Mapping[str, array_typing.FloatVector]
  main_ion_fractions: Mapping[str, array_typing.FloatScalar]
  q_face: array_typing.FloatVectorFace
  s_face: array_typing.FloatVectorFace
  v_loop_lcfs: array_typing.FloatScalar
  Z_i: array_typing.FloatVectorCell
  Z_i_face: array_typing.FloatVectorFace
  A_i: array_typing.FloatScalar
  Z_impurity: array_typing.FloatVectorCell
  Z_impurity_face: array_typing.FloatVectorFace
  A_impurity: array_typing.FloatVectorCell
  A_impurity_face: array_typing.FloatVectorFace
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace
  sigma: array_typing.FloatVectorCell
  sigma_face: array_typing.FloatVectorFace
  j_total: array_typing.FloatVectorCell
  j_total_face: array_typing.FloatVectorFace
  Ip_profile_face: array_typing.FloatVectorFace
  toroidal_angular_velocity: cell_variable.CellVariable
  charge_state_info: charge_states.ChargeStateInfo
  charge_state_info_face: charge_states.ChargeStateInfo

  @functools.cached_property
  def impurity_density_scaling(self) -> jax.Array:
    """Scaling factor for impurity density: n_imp_true / n_imp_eff."""
    return self.Z_impurity / self.charge_state_info.Z_avg

  @functools.cached_property
  def pressure_thermal_e(self) -> cell_variable.CellVariable:
    """Electron thermal pressure [Pa]."""
    return cell_variable.CellVariable(
        value=self.n_e.value * self.T_e.value * constants.CONSTANTS.keV_to_J,
        face_centers=self.n_e.face_centers,
        right_face_constraint=self.n_e.right_face_constraint
        * self.T_e.right_face_constraint
        * constants.CONSTANTS.keV_to_J,
        right_face_grad_constraint=None,
    )

  @functools.cached_property
  def pressure_thermal_i(self) -> cell_variable.CellVariable:
    """Ion thermal pressure [Pa]."""
    return cell_variable.CellVariable(
        value=self.T_i.value
        * constants.CONSTANTS.keV_to_J
        * (self.n_i.value + self.n_impurity.value),
        face_centers=self.n_i.face_centers,
        right_face_constraint=self.T_i.right_face_constraint
        * constants.CONSTANTS.keV_to_J
        * (
            self.n_i.right_face_constraint
            + self.n_impurity.right_face_constraint
        ),
        right_face_grad_constraint=None,
    )

  @functools.cached_property
  def pressure_thermal_total(self) -> cell_variable.CellVariable:
    """Total thermal pressure [Pa]."""
    return cell_variable.CellVariable(
        value=self.pressure_thermal_e.value + self.pressure_thermal_i.value,
        face_centers=self.pressure_thermal_e.face_centers,
        right_face_constraint=self.pressure_thermal_e.right_face_constraint
        + self.pressure_thermal_i.right_face_constraint,
        right_face_grad_constraint=None,
    )

  def quasineutrality_satisfied(self) -> bool:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.n_i.value * self.Z_i + self.n_impurity.value * self.Z_impurity,
        self.n_e.value,
    ).item()

  def negative_temperature_or_density(self) -> jax.Array:
    """Checks if any temperature or density is negative."""
    profiles_to_check = (
        self.T_i,
        self.T_e,
        self.n_e,
        self.n_i,
        self.n_impurity,
        self.impurity_fractions,
    )
    # Check if any profile is less than -eps
    # (allowing for numerical precision errors)
    return np.any(
        np.array([
            np.any(np.less(x, -constants.CONSTANTS.eps))
            for x in jax.tree.leaves(profiles_to_check)
        ])
    )

  def below_minimum_temperature(self, T_minimum_eV: float) -> bool:
    """Return True if T_e or T_i is below the minimum temperature threshold."""
    # Convert eV -> keV since internal storage is keV
    T_minimum_keV = T_minimum_eV / 1000.0

    is_low_te = jnp.any(self.T_e.value < T_minimum_keV)
    is_low_ti = jnp.any(self.T_i.value < T_minimum_keV)

    # Use .item() to return a concrete Python boolean
    return (is_low_te | is_low_ti).item()

  def __str__(self) -> str:
    return f"""
      CoreProfiles(
        T_i={self.T_i},
        T_e={self.T_e},
        psi={self.psi},
        n_e={self.n_e},
        n_i={self.n_i},
        n_impurity={self.n_impurity},
        impurity_fractions={self.impurity_fractions},
      )
    """


# TODO(b/426132633): restructure and rename attributes for V2. Choices were made
# when refactoring to avoid breaking public API.
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class CoreTransport:
  """Coefficients for the plasma transport.

  See docstrings of `neoclassical/transport/base.py` and
  `transport_model/transport_model.py` for more details.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
  chi_face_el_itg: jax.Array | None = None
  chi_face_el_tem: jax.Array | None = None
  chi_face_el_etg: jax.Array | None = None
  chi_face_ion_itg: jax.Array | None = None
  chi_face_ion_tem: jax.Array | None = None
  d_face_el_itg: jax.Array | None = None
  d_face_el_tem: jax.Array | None = None
  v_face_el_itg: jax.Array | None = None
  v_face_el_tem: jax.Array | None = None
  chi_neo_i: jax.Array | None = None
  chi_neo_e: jax.Array | None = None
  D_neo_e: jax.Array | None = None
  V_neo_e: jax.Array | None = None
  V_neo_ware_e: jax.Array | None = None

  def __post_init__(self):
    # Use the array size of chi_face_el as a template.
    template = self.chi_face_el
    if self.chi_neo_i is None:
      self.chi_neo_i = jnp.zeros_like(template)
    if self.chi_neo_e is None:
      self.chi_neo_e = jnp.zeros_like(template)
    if self.D_neo_e is None:
      self.D_neo_e = jnp.zeros_like(template)
    if self.V_neo_e is None:
      self.V_neo_e = jnp.zeros_like(template)
    if self.V_neo_ware_e is None:
      self.V_neo_ware_e = jnp.zeros_like(template)

  def chi_max(
      self,
      geo: geometry.Geometry,
  ) -> jax.Array:
    """Calculates the maximum value of chi.

    Args:
      geo: Geometry of the torus.

    Returns:
      chi_max: Maximum value of chi.
    """
    return jnp.maximum(
        jnp.max((self.chi_face_ion + self.chi_neo_i) * geo.g1_over_vpr2_face),
        jnp.max((self.chi_face_el + self.chi_neo_e) * geo.g1_over_vpr2_face),
    )

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a CoreTransport with all zeros. Useful for initializing."""
    shape = geo.rho_face.shape
    return cls(
        chi_face_ion=jnp.zeros(shape),
        chi_face_el=jnp.zeros(shape),
        d_face_el=jnp.zeros(shape),
        v_face_el=jnp.zeros(shape),
        chi_face_el_bohm=jnp.zeros(shape),
        chi_face_el_gyrobohm=jnp.zeros(shape),
        chi_face_ion_bohm=jnp.zeros(shape),
        chi_face_ion_gyrobohm=jnp.zeros(shape),
        chi_neo_i=jnp.zeros(shape),
        chi_neo_e=jnp.zeros(shape),
        D_neo_e=jnp.zeros(shape),
        V_neo_e=jnp.zeros(shape),
        V_neo_ware_e=jnp.zeros(shape),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class SolverNumericOutputs:
  """Numerical quantities related to the solver.

  Attributes:
    outer_solver_iterations: Number of iterations performed in the outer loop of
      the solver.
    solver_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the solver.
    sawtooth_crash: True if a sawtooth model is active and the solver step
      corresponds to a sawtooth crash step.
  """

  outer_solver_iterations: array_typing.IntScalar
  solver_error_state: array_typing.IntScalar
  inner_solver_iterations: array_typing.IntScalar
  sawtooth_crash: array_typing.BoolScalar


# TODO(b/434175938): change to StrEnum
@enum.unique
class SimError(enum.Enum):
  """Integer enum for sim error handling."""

  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2
  NEGATIVE_CORE_PROFILES = 3
  REACHED_MIN_DT = 4
  LOW_TEMPERATURE_COLLAPSE = 5

  def log_error(self):
    match self:
      case SimError.NEGATIVE_CORE_PROFILES:
        logging.error("""
            Simulation stopped due to negative values in core profiles.
            """)
      case SimError.NAN_DETECTED:
        logging.error("""
            Simulation stopped due to NaNs in state and/or post processed outputs.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.QUASINEUTRALITY_BROKEN:
        logging.error("""
            Simulation stopped due to quasineutrality being violated.
            Possible cause is bad handling of impurity species.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.REACHED_MIN_DT:
        logging.error("""
            Simulation stopped because the adaptive time step became too small.
            A common cause of vanishing timesteps is due to the nonlinear solver
            tending to negative densities or temperatures. This often arises
            through physical reasons like radiation collapse, or unphysical
            configuration such as impurity densities incompatible with physical
            quasineutrality. Check the output file for near-zero temperatures or
            densities at the last valid step.
            """)
      case SimError.LOW_TEMPERATURE_COLLAPSE:
        logging.error("""
          Simulation stopped because ion or electron temperature fell below the
          configured minimum threshold. This is usually caused by radiative
          collapse. Output file contains all profiles up to the last valid step.
          """)
      case SimError.NO_ERROR:
        pass
      case _:
        raise ValueError(f"Unknown SimError: {self}")


class SimStatus(enum.StrEnum):
  """String enum for simulation output file status.

  This indicates the state of the simulation when the output file was written.
  """

  COMPLETED = "completed"  # Simulation completed successfully to t_final
  CHECKPOINT = "checkpoint"  # Intermediate checkpoint (not yet used)
  ERROR = "error"  # Simulation stopped due to an error condition
