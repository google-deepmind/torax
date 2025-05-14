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
import enum
from typing import Optional

from absl import logging
import chex
import jax
from jax import numpy as jnp
from torax import array_typing
from torax.fvm import cell_variable
from torax.geometry import geometry
import typing_extensions

# pylint: disable=invalid-name


@chex.dataclass(frozen=True, eq=False)
class CoreProfiles:
  """Dataclass for holding the evolving core plasma profiles.

  This dataclass is inspired by the IMAS `core_profiles` IDS.

  Many of the profiles in this class are evolved by the PDE system in TORAX, and
  therefore are stored as CellVariables. Other profiles are computed outside the
  internal PDE system, and are simple JAX arrays.

  Attributes:
      T_i: Ion temperature [keV].
      T_e: Electron temperature [keV].
      psi: Poloidal flux [Wb].
      psidot: Time derivative of poloidal flux (loop voltage) [V].
      n_e: Electron density [density_reference m^-3].
      n_i: Main ion density [density_reference m^-3].
      n_impurity: Impurity density [density_reference m^-3].
      q_face: Safety factor.
      s_face: Magnetic shear.
      density_reference: Reference density [m^-3].
      vloop_lcfs: Loop voltage at LCFS (V).
      Z_i: Main ion charge on cell grid [dimensionless].
      Z_i_face: Main ion charge on face grid [dimensionless].
      A_i: Main ion mass [amu].
      Z_impurity: Impurity charge on cell grid [dimensionless].
      Z_impurity_face: Impurity charge on face grid [dimensionless].
      A_impurity: Impurity mass [amu].
      sigma: Conductivity on cell grid [S/m].
      sigma_face: Conductivity on face grid [S/m].
      j_total: Total current density on the cell grid [A/m^2].
      j_total_face: Total current density on face grid [A/m^2].
      Ip_profile_face: Plasma current profile on the face grid [A].
  """

  T_i: cell_variable.CellVariable
  T_e: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  psidot: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  n_i: cell_variable.CellVariable
  n_impurity: cell_variable.CellVariable
  q_face: array_typing.ArrayFloat
  s_face: array_typing.ArrayFloat
  density_reference: array_typing.ScalarFloat
  vloop_lcfs: array_typing.ScalarFloat
  Z_i: array_typing.ArrayFloat
  Z_i_face: array_typing.ArrayFloat
  A_i: array_typing.ScalarFloat
  Z_impurity: array_typing.ArrayFloat
  Z_impurity_face: array_typing.ArrayFloat
  A_impurity: array_typing.ScalarFloat
  sigma: array_typing.ArrayFloat
  sigma_face: array_typing.ArrayFloat
  j_total: array_typing.ArrayFloat
  j_total_face: array_typing.ArrayFloat
  Ip_profile_face: array_typing.ArrayFloat

  def quasineutrality_satisfied(self) -> bool:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.n_i.value * self.Z_i + self.n_impurity.value * self.Z_impurity,
        self.n_e.value,
    ).item()

  def negative_temperature_or_density(self) -> bool:
    """Checks if any temperature or density is negative."""
    profiles_to_check = (
        self.T_i,
        self.T_e,
        self.n_e,
        self.n_i,
        self.n_impurity,
    )
    return any(
        [jnp.any(jnp.less(x, 0.0)) for x in jax.tree.leaves(profiles_to_check)]
    )

  def index(self, i: int) -> typing_extensions.Self:
    """If the CoreProfiles is a history, returns the i-th CoreProfiles."""
    idx = lambda x: x[i]
    state = jax.tree_util.tree_map(idx, self)
    return state

  def sanity_check(self):
    for field in CoreProfiles.__dataclass_fields__:
      value = getattr(self, field)
      if hasattr(value, "sanity_check"):
        value.sanity_check()

  def __str__(self) -> str:
    return f"""
      CoreProfiles(
        T_i={self.T_i},
        T_e={self.T_e},
        psi={self.psi},
        n_e={self.n_e},
        n_impurity={self.n_impurity},
        n_i={self.n_i},
      )
    """


@chex.dataclass
class CoreTransport:
  """Coefficients for the plasma transport.

  These coefficients are computed by TORAX transport models. See the
  transport_model/ folder for more info.

  NOTE: The naming of this class is inspired by the IMAS `core_transport` IDS,
  but its schema is not a 1:1 mapping to that IDS.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
    chi_face_el_bohm: (Optional) Bohm contribution for electron heat
      conductivity.
    chi_face_el_gyrobohm: (Optional) GyroBohm contribution for electron heat
      conductivity.
    chi_face_ion_bohm: (Optional) Bohm contribution for ion heat conductivity.
    chi_face_ion_gyrobohm: (Optional) GyroBohm contribution for ion heat
      conductivity.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: Optional[jax.Array] = None
  chi_face_el_gyrobohm: Optional[jax.Array] = None
  chi_face_ion_bohm: Optional[jax.Array] = None
  chi_face_ion_gyrobohm: Optional[jax.Array] = None

  def __post_init__(self):
    # Use the array size of chi_face_el as a reference.
    if self.chi_face_el_bohm is None:
      self.chi_face_el_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_el_gyrobohm is None:
      self.chi_face_el_gyrobohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_bohm is None:
      self.chi_face_ion_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_gyrobohm is None:
      self.chi_face_ion_gyrobohm = jnp.zeros_like(self.chi_face_el)

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
        jnp.max(self.chi_face_ion * geo.g1_over_vpr2_face),
        jnp.max(self.chi_face_el * geo.g1_over_vpr2_face),
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
    )


@chex.dataclass
class SolverNumericOutputs:
  """Numerical quantities related to the solver.

  Attributes:
    outer_solver_iterations: Number of iterations performed in the outer loop
      of the solver.
    solver_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the solver.
    sawtooth_crash: True if a sawtooth model is active and the solver step
      corresponds to a sawtooth crash step.
  """

  outer_solver_iterations: int = 0
  solver_error_state: int = 0
  inner_solver_iterations: int = 0
  sawtooth_crash: bool = False


@enum.unique
class SimError(enum.Enum):
  """Integer enum for sim error handling."""

  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2
  NEGATIVE_CORE_PROFILES = 3

  def log_error(self):
    match self:
      case SimError.NEGATIVE_CORE_PROFILES:
        logging.error("""
            Simulation stopped due to negative values in core profiles.
            """)
      case SimError.NAN_DETECTED:
        logging.error("""
            Simulation stopped due to NaNs in state.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.QUASINEUTRALITY_BROKEN:
        logging.error("""
            Simulation stopped due to quasineutrality being violated.
            Possible cause is bad handling of impurity species.
            Output file contains all profiles up to the last valid step.
            """)
      case SimError.NO_ERROR:
        pass
      case _:
        raise ValueError(f"Unknown SimError: {self}")
