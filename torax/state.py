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

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax.config import config_args
from torax.fvm import cell_variable
from torax.sources import source_profiles


@chex.dataclass(frozen=True)
class Currents:
  """Dataclass to group currents and related variables (e.g. conductivity).

  Not all fields are actually used by the library. For example,
  j_bootstrap and I_bootstrap are updated during the sim loop,
  but not read from. These fields are an output of the library
  that may be interesting for the end user to plot, etc.
  """

  jtot: jax.Array
  jtot_face: jax.Array
  johm: jax.Array
  johm_face: jax.Array
  jext: jax.Array
  jext_face: jax.Array
  j_bootstrap: jax.Array
  j_bootstrap_face: jax.Array
  # pylint: disable=invalid-name
  # Using physics notation naming convention
  I_bootstrap: jax.Array
  sigma: jax.Array
  jtot_hires: Optional[jax.Array] = None


@chex.dataclass(frozen=True, eq=False)
class CoreProfiles:
  """Dataclass for holding the evolving core plasma profiles.

  This dataclass is inspired by the IMAS `core_profiles` IDS.

  Many of the profiles in this class are evolved by the PDE system in TORAX, and
  therefore are stored as CellVariables. Other profiles are computed outside the
  internal PDE system, and are simple JAX arrays.
  """

  temp_ion: cell_variable.CellVariable  # Ion temperature
  temp_el: cell_variable.CellVariable  # Electron temperature
  psi: cell_variable.CellVariable  # Poloidal flux
  psidot: (
      cell_variable.CellVariable
  )  # Time derivative of poloidal flux (loop voltage)
  ne: cell_variable.CellVariable  # Electron density
  ni: cell_variable.CellVariable  # Main ion density
  currents: Currents
  q_face: jax.Array
  s_face: jax.Array
  nref: jax.Array  # Reference density for ion and electron density

  def history_elem(self) -> CoreProfiles:
    """Returns the current CoreProfiles as a history entry.

    Histories are CoreProfiles with all the tree leaves getting an extra
    dimension due to stacking, e.g. as the output of `jax.lax.scan`.
    Some CoreProfiles fields such as `cell_variable.CellVariable` cease to
    function after becoming histories.
    """

    return dataclasses.replace(
        self,
        temp_ion=self.temp_ion.history_elem(),
        temp_el=self.temp_el.history_elem(),
        psi=self.psi.history_elem(),
        psidot=self.psidot.history_elem(),
        ne=self.ne.history_elem(),
        ni=self.ni.history_elem(),
        currents=self.currents,
        q_face=self.q_face,
        s_face=self.s_face,
        nref=self.nref,
    )

  def index(self, i: int) -> CoreProfiles:
    """If the CoreProfiles is a history, returns the i-th CoreProfiles."""
    idx = lambda x: x[i]
    state = jax.tree_util.tree_map(idx, self)
    # These variables track whether they are histories, so when we collapse down
    # to a single state we need to explicitly clear the history flag.
    history_vars = ["temp_ion", "temp_el", "psi", "psidot", "ne", "ni"]
    history_replace = {"history": None}
    replace_dict = {var: history_replace for var in history_vars}
    state = config_args.recursive_replace(state, **replace_dict)
    return state

  def sanity_check(self):
    for field in CoreProfiles.__dataclass_fields__:
      value = getattr(self, field)
      if hasattr(value, "sanity_check"):
        value.sanity_check()

  def project(self, weights):
    project = lambda x: jnp.dot(weights, x)
    proj_currents = jax.tree_util.tree_map(project, self.currents)
    return dataclasses.replace(
        self,
        temp_ion=self.temp_ion.project(weights),
        temp_el=self.temp_el.project(weights),
        psi=self.psi.project(weights),
        psidot=self.psidot.project(weights),
        ne=self.ne.project(weights),
        ni=self.ni.project(weights),
        currents=proj_currents,
        q_face=project(self.q_face),
        s_face=project(self.s_face),
        nref=project(self.nref),
    )

  def __hash__(self):
    """Make CoreProfiles hashable.

    Be careful, if a CoreProfiles gets garbage collected a different
    CoreProfiles could have the same hash later, so it's important to always
    store the CoreProfiles (to prevent it from being garbage collected) not just
    its hash.

    Returns:
      hash: The hash, in this case, just the `id`, of the CoreProfiles.
    """
    return id(self)


@chex.dataclass(frozen=True, eq=False)
class CoreTransport:
  """Coefficients for the plasma transport.

  These coefficients are computed by TORAX transport models. See the
  transport_model/ folder for more info.

  NOTE: The naming of this class is inspired by the IMAS `core_transport` IDS,
  but it's schema is not a 1:1 mapping to that IDS.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array

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
  def zeros(cls, geo: geometry.Geometry) -> CoreTransport:
    """Returns a CoreTransport with all zeros. Useful for initializing."""
    return cls(
        chi_face_ion=jnp.zeros(geo.r_face.shape),
        chi_face_el=jnp.zeros(geo.r_face.shape),
        d_face_el=jnp.zeros(geo.r_face.shape),
        v_face_el=jnp.zeros(geo.r_face.shape),
    )


@chex.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves core_profiles which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which may evolve on each simulation step, such
  as sources and transport.

  This class includes both core_profiles and these additional elements.

  Attributes:
    t: time coordinate
    dt: timestep interval
    core_profiles: Core plasma profiles at time t.
    core_transport: Core plasma transport coefficients computed at time t.
    core_sources: Profiles for all sources/sinks. For any state-dependent source
      models, the profiles in this dataclass are computed based on the core
      profiles at time t, almost. When running `sim.run_simulation()`, any
      profile from an "explicit" state-dependent source will be computed with
      the core profiles at time t. Any profile from an "implicit"
      state-dependent source will be computed with an intermediate state from
      the previous time step's solver. This should be close to the core profiles
      at time t, but is not guaranteed to be. In case exact source profiles are
      required for each time step, they must be recomputed manually after
      running `run_simulation()`.
    stepper_iterations: number of stepper iterations carried out in previous
      step, i.e. the number of times dt was reduced when using the adaptive dt
      method.
    time_step_calculator_state: the state of the TimeStepper
    stepper_error_state: 0 for successful convergence of the PDE stepper, 1 for
      unsuccessful convergence, leading to recalculation at reduced timestep
  """

  # Time variables.
  t: jax.Array
  dt: jax.Array

  # Profiles evolved or calculated by the simulation.
  core_profiles: CoreProfiles
  core_transport: CoreTransport
  core_sources: source_profiles.SourceProfiles

  # Other "side" states used for logging and feeding to other components of
  # TORAX.
  stepper_iterations: int
  time_step_calculator_state: Any
  stepper_error_state: int
