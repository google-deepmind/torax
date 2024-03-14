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

"""variables that change over time.

For the JAX functional design of Torax, it is important that most variables
do not change. All variables that do change must be in State so that they
are explicitly updated by returning new copies of State.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import chex
import jax
from jax import numpy as jnp
from torax import config
from torax import fvm
from torax import geometry


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
  jtot_hires: jax.Array
  johm: jax.Array
  johm_face: jax.Array
  jext: jax.Array
  jext_face: jax.Array
  j_bootstrap: jax.Array
  j_bootstrap_face: jax.Array
  # pylint: disable=invalid-name
  # Using PINT / physics notation naming convention
  I_bootstrap: jax.Array
  sigma: jax.Array


@chex.dataclass(frozen=True, eq=False)
class State:
  """Dataclass for holding the evolving state of the system."""

  temp_ion: fvm.CellVariable  # Ion temperature
  temp_el: fvm.CellVariable  # Electron temperature
  psi: fvm.CellVariable  # Poloidal flux
  psidot: fvm.CellVariable  # Time derivative of poloidal flux (loop voltage)
  ne: fvm.CellVariable  # Electron density
  ni: fvm.CellVariable  # Main ion density
  currents: Currents
  q_face: jax.Array
  s_face: jax.Array

  def history_elem(self) -> State:
    """Returns the current State as a history entry.

    Histories are States with all the tree leaves getting an extra dimension
    due to stacking, e.g. as the output of `jax.lax.scan`.
    Some State fields such as `fvm.CellVariable` cease to function after
    becoming histories.
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
    )

  def index(self, i: int) -> State:
    """If the State is a history, returns State from step `i` of the history."""
    idx = lambda x: x[i]
    state = jax.tree_util.tree_map(idx, self)
    # These variables track whether they are histories, so when we collapse down
    # to a single state we need to explicitly clear the history flag.
    history_vars = ["temp_ion", "temp_el", "psi", "psidot", "ne", "ni"]
    history_replace = {"history": None}
    replace_dict = {var: history_replace for var in history_vars}
    state = config.recursive_replace(state, **replace_dict)
    return state

  def sanity_check(self):
    for field in State.__dataclass_fields__:
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
    )

  def __hash__(self):
    """Make States hashable.

    Be careful, if a State gets garbage collected a different State could
    have the same hash later, so it's important to always store the State
    (to prevent it from being garbage collected) not just its hash.

    Returns:
      hash: The hash, in this case, just the `id`, of the State.
    """
    return id(self)


@chex.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves the "mesh state" which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which evolve on each simulation step.

  This class includes both the mesh state and these additional elements.

  Attributes:
    t: time coordinate
    dt: timestep interval
    stepper_iterations: number of stepper iterations carried out in previous
      step, i.e. the number of times dt was reduced when using the adaptive dt
      method.
    mesh_state: all state variables defined on meshes
    time_step_calculator_state: the state of the TimeStepper
    stepper_error_state: 0 for successful convergence of the PDE stepper, 1 for
      unsuccessful convergence, leading to recalculation at reduced timestep
  """

  t: jax.Array
  dt: jax.Array
  stepper_iterations: int
  mesh_state: State
  time_step_calculator_state: Any
  stepper_error_state: int


@chex.dataclass
class AuxOutput:
  """Auxiliary output for each simulation step.

  Attributes:
    chi_face_ion: Extra output for inspecting the ion chi on the face grid.
    chi_face_el: Extra output for inspecting the electron chi on the face grid.
    source_ion: Extra output for inspecting the generic external ion source on
      the cell grid.
    source_el: Extra output for inspecting the generic external electron source
      on the cell grid.
    Pfus_i: Extra output for inspecting the fusion power source for heating ions
      on the cell grid.
    Pfus_e: Extra output for inspecting the fusion power source for heating
      electrons on the cell grid.
    Pohm: Extra output for inspecting Extra output for inspecting the ohmic
      heating on the cell grid.
    Qei: Extra output for inspecting the ion-el collisional heating source.
  """

  # pylint: disable=invalid-name
  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  source_ion: jax.Array
  source_el: jax.Array
  Pfus_i: jax.Array
  Pfus_e: jax.Array
  Pohm: jax.Array
  Qei: jax.Array
  # pylint: enable=invalid-name

  @classmethod
  def zero_output(cls, geo: geometry.Geometry) -> "AuxOutput":
    """Returns an AuxOutput with all zeros. Useful for initializing."""
    return cls(
        chi_face_ion=jnp.zeros(geo.r_face.shape),
        chi_face_el=jnp.zeros(geo.r_face.shape),
        source_ion=jnp.zeros(geo.r.shape),
        source_el=jnp.zeros(geo.r.shape),
        Pfus_i=jnp.zeros(geo.r.shape),
        Pfus_e=jnp.zeros(geo.r.shape),
        Pohm=jnp.zeros(geo.r.shape),
        Qei=jnp.zeros(geo.r.shape),
    )


@chex.dataclass
class ToraxOutput:
  """Full simulator output state.

  The simulation stepping in sim.py evolves the "mesh state" which includes all
  the attributes the simulation is advancing. But beyond those attributes, there
  are stateful elements which need to be tracked and outputted each step.
  This class includes both the mesh state and these additional elements.

  Attributes:
    state: Full simulator state for this time step.
    aux: Auxiliary outputs for this time step.
  """

  # TODO( b/320292127): Rename and rebundle state variables.

  state: ToraxSimState
  aux: AuxOutput


def build_history_from_outputs(
    torax_outputs: tuple[ToraxOutput, ...],
) -> tuple[State, AuxOutput]:
  mesh_states = [out.state.mesh_state.history_elem() for out in torax_outputs]
  aux = [out.aux for out in torax_outputs]
  stack = lambda *ys: jnp.stack(ys)
  return jax.tree_util.tree_map(stack, *mesh_states), jax.tree_util.tree_map(
      stack, *aux
  )


def build_time_history_from_outputs(
    torax_outputs: tuple[ToraxOutput, ...],
) -> jnp.ndarray:
  times = [out.state.t for out in torax_outputs]
  return jnp.array(times)
