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

"""Core profiles and conductivity definitions."""

import dataclasses
import functools
from typing import Self

import jax
from jax import numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src.core_profiles import solver_profiles
from torax._src.core_profiles import solver_state
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import fast_ion as fast_ion_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class Conductivity:
  """Values returned by a conductivity model."""

  sigma: array_typing.FloatVectorCell
  sigma_face: array_typing.FloatVectorFace

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> Self:
    """Returns a Conductivity with all zero profiles."""
    return cls(
        sigma=jnp.zeros_like(geo.rho),
        sigma_face=jnp.zeros_like(geo.rho_face),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CoreProfiles(solver_profiles.SolverProfiles):
  """Dataclass for holding the evolving core plasma profiles.

  Extends `SolverProfiles` with physics-model-derived profiles (`fast_ions`,
  `conductivity`, `poloidal_velocity`, and `psidot`).

  Profiles are stored as `CellVariable` or JAX arrays. Array-based profiles
  are on either the cell or face grid; those on the face grid are denoted with a
  `_face` suffix.

  Attributes:
    fast_ions: A tuple of `fast_ion_lib.FastIon` objects. Fast ions are
      particles that have kinetic energy above the thermal equilibrium
      distribution of the bulk plasma. Each `FastIon` object contains density
      and temperature profiles as well as species and source information for a
      population of fast ions.
    conductivity: Values returned by a conductivity model.
    poloidal_velocity: Neoclassical poloidal velocity [m/s].
    psidot: Time derivative of poloidal flux (loop voltage) [V].
  """

  fast_ions: tuple[fast_ion_lib.FastIon, ...]
  conductivity: Conductivity
  poloidal_velocity: cell_variable.CellVariable
  psidot: cell_variable.CellVariable

  @functools.cached_property
  def v_loop_lcfs(self) -> array_typing.FloatScalar:
    """Loop voltage at the LCFS [V]."""
    return self.psidot.right_face_value[..., 0]

  @functools.cached_property
  def n_impurity_thermal(self) -> cell_variable.CellVariable:
    """True thermal impurity density without fast ions [m^-3]."""
    return self.calculate_n_impurity_thermal(self.fast_ions)

  @functools.cached_property
  def n_i_total_thermal_face(self) -> array_typing.FloatVectorFace:
    """Total thermal ion particle density (main ions + impurities) [m^-3]."""
    return self._n_i_total_thermal_face_from_impurity(self.n_impurity_thermal)

  @functools.cached_property
  def pressure_thermal_i(self) -> cell_variable.CellVariable:
    """Ion thermal pressure [Pa]."""
    return self._pressure_thermal_i_from_impurity(self.n_impurity_thermal)

  @functools.cached_property
  def pressure_thermal_total(self) -> cell_variable.CellVariable:
    """Total thermal pressure [Pa]."""
    return cell_variable.add_cell_variables(
        self.pressure_thermal_e, self.pressure_thermal_i
    )

  @functools.cached_property
  def pressure_fast_i(self) -> cell_variable.CellVariable:
    """Fast ion pressure [Pa].

    Sum over all fast ion species: n * T * keV_to_J.
    Note that this formulation is limited to Maxwellian fast ions, where we
    take some average energy to be T. This functionality may be extended in
    future work to include velocity distributions.
    """
    p_fast = jnp.zeros_like(self.kinetic.n_e.value)
    p_fast_right = jnp.array(0.0)
    for fast_ion in self.fast_ions:
      p_fast = (
          p_fast
          + fast_ion.n.value * fast_ion.T.value * constants.CONSTANTS.keV_to_J
      )
      if (
          fast_ion.n.right_face_constraint is None
          or fast_ion.T.right_face_constraint is None
      ):
        raise ValueError(
            'Both fast ion n and T must have a right_face_constraint.'
        )
      p_fast_right = (
          p_fast_right
          + fast_ion.n.right_face_constraint
          * fast_ion.T.right_face_constraint
          * constants.CONSTANTS.keV_to_J
      )
    return cell_variable.CellVariable(
        value=p_fast,
        face_centers=self.kinetic.n_e.face_centers,
        right_face_constraint=p_fast_right,
        right_face_grad_constraint=None,
    )

  @functools.cached_property
  def pressure_total_i(self) -> cell_variable.CellVariable:
    """Total ion pressure (thermal + fast) [Pa]."""
    return cell_variable.add_cell_variables(
        self.pressure_thermal_i, self.pressure_fast_i
    )

  @functools.cached_property
  def pressure_total(self) -> cell_variable.CellVariable:
    """Total plasma pressure [Pa]."""
    return cell_variable.add_cell_variables(
        self.pressure_thermal_total, self.pressure_fast_i
    )

  def to_solver_state(self) -> solver_state.SolverState:
    """Extracts the primary solvable/prescribed profiles as a SolverState."""
    return dataclasses.replace(
        super().to_solver_state(),
        psidot_right_face_grad_constraint=(
            self.psidot.right_face_grad_constraint
        ),
    )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Packs time-stacked core profiles into xarray-ready variables.

    Args:
      context: Grid context providing time and radial coordinates for shape
        validation and packing. Also registers `main_ion` species coordinates
        in-place when `ions.main_ion_fractions` is non-empty.

    Returns:
      Mapping from xarray output variable names to `(dims, data, attrs)`
      `OutputVar` tuples:
        - Cell+boundary profiles (including per-source/species fast-ion
          densities and temperatures): `('time', 'rho_norm')`, shape
          `(n_time, n_rho_cell + 2)`.
        - Face profiles (`q_face`, `s_face`, `I_p_profile`):
          `('time', 'rho_face_norm')`, shape `(n_time, n_rho_face)`.
        - Scalars (`v_loop_lcfs`, `A_i`, `I_p`): `('time',)`, shape `(n_time,)`.
        - Main ion fractions (if present): `('main_ion', 'time')`, shape
          `(n_main_ions, n_time)`.
    """
    out_dict = {}

    cell_vars = {
        output_keys.T_I: self.kinetic.T_i.cell_plus_boundaries(),
        output_keys.T_E: self.kinetic.T_e.cell_plus_boundaries(),
        output_keys.PSI: self.poloidal_flux.psi.cell_plus_boundaries(),
        output_keys.V_LOOP: self.psidot.cell_plus_boundaries(),
        output_keys.N_E: self.kinetic.n_e.cell_plus_boundaries(),
        output_keys.N_I: self.kinetic.n_i.cell_plus_boundaries(),
        output_keys.N_IMPURITY: self.kinetic.n_impurity.cell_plus_boundaries(),
        output_keys.TOROIDAL_ANGULAR_VELOCITY: (
            self.kinetic.toroidal_angular_velocity.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_THERMAL_E: (
            self.pressure_thermal_e.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_THERMAL_I: (
            self.pressure_thermal_i.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_THERMAL_TOTAL: (
            self.pressure_thermal_total.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_FAST_I: (
            self.pressure_fast_i.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_TOTAL_I: (
            self.pressure_total_i.cell_plus_boundaries()
        ),
        output_keys.PRESSURE_TOTAL: self.pressure_total.cell_plus_boundaries(),
    }
    for key, cell_val in cell_vars.items():
      out_dict[key] = context.pack(key, cell_val)

    cell_face_pairs = [
        (output_keys.Z_I, self.ions.Z_i, self.ions.Z_i_face),
        (
            output_keys.Z_IMPURITY,
            self.ions.Z_impurity,
            self.ions.Z_impurity_face,
        ),
        (
            output_keys.A_IMPURITY,
            self.ions.A_impurity,
            self.ions.A_impurity_face,
        ),
        (output_keys.Z_EFF, self.ions.Z_eff, self.ions.Z_eff_face),
        (
            output_keys.SIGMA_PARALLEL,
            self.conductivity.sigma,
            self.conductivity.sigma_face,
        ),
        (
            output_keys.J_TOROIDAL_TOTAL,
            self.poloidal_flux.j_toroidal_total,
            self.poloidal_flux.j_toroidal_total_face,
        ),
    ]
    for key, cell_val, face_val in cell_face_pairs:
      extended_data = output_grid_context.extend_cell_grid_to_boundaries(
          cell_val, face_val
      )
      out_dict[key] = context.pack(key, extended_data)

    face_vars = {
        output_keys.Q: self.poloidal_flux.q_face,
        output_keys.MAGNETIC_SHEAR: self.poloidal_flux.s_face,
        output_keys.I_P_PROFILE: self.poloidal_flux.I_p_profile_face,
    }
    for key, face_data in face_vars.items():
      out_dict[key] = context.pack(key, face_data)

    scalar_vars = {
        output_keys.V_LOOP_LCFS: self.v_loop_lcfs,
        output_keys.A_I: self.ions.A_i,
        output_keys.I_P: self.poloidal_flux.I_p_profile_face[..., -1],
    }
    for key, scalar_data in scalar_vars.items():
      out_dict[key] = context.pack(key, scalar_data)

    if self.ions.main_ion_fractions:
      main_ions = np.asarray(sorted(self.ions.main_ion_fractions.keys()))
      context.register_coord(output_keys.MAIN_ION, main_ions)
      data = np.stack(
          [self.ions.main_ion_fractions[ion] for ion in main_ions],
          axis=0,
      )
      out_dict[output_keys.MAIN_ION_FRACTIONS] = (
          (output_keys.MAIN_ION, output_keys.TIME),
          data,
          output_keys.get_units(output_keys.MAIN_ION_FRACTIONS),
      )

    for fi in self.fast_ions:
      source_key = f'{fi.source}_{fi.species}'
      n_key = output_keys.OutputKey(
          output_keys.n_fast_ion_key(source_key),
          units=output_keys.Units.INVERSE_CUBIC_METER,
          grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
      )
      t_key = output_keys.OutputKey(
          output_keys.T_fast_ion_key(source_key),
          units=output_keys.Units.KEV,
          grid_type=output_keys.GridType.CELL_PLUS_BOUNDARIES,
      )
      out_dict[n_key] = context.pack(n_key, fi.n.cell_plus_boundaries())
      out_dict[t_key] = context.pack(t_key, fi.T.cell_plus_boundaries())

    return out_dict
