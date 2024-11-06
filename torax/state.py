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
import enum
from typing import Any, Optional

import chex
import jax
from jax import numpy as jnp
from torax import array_typing
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

  jtot: array_typing.ArrayFloat
  jtot_face: array_typing.ArrayFloat
  johm: array_typing.ArrayFloat
  generic_current_source: array_typing.ArrayFloat
  j_bootstrap: array_typing.ArrayFloat
  j_bootstrap_face: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  # Using physics notation naming convention
  I_bootstrap: array_typing.ScalarFloat
  Ip: array_typing.ScalarFloat
  sigma: array_typing.ArrayFloat
  jtot_hires: Optional[array_typing.ArrayFloat] = None

  def has_nans(self) -> bool:
    """Checks for NaNs in all attributes of Currents."""

    def _check_for_nans(x: Any) -> bool:
      if isinstance(x, jax.Array):
        return jnp.any(jnp.isnan(x)).item()
      else:
        return False

    return any(
        _check_for_nans(getattr(self, field))
        for field in self.__dataclass_fields__
    )

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> "Currents":
    """Returns a Currents with all zeros."""
    return cls(
        jtot=jnp.zeros(geo.rho_face.shape),
        jtot_face=jnp.zeros(geo.rho_face.shape),
        johm=jnp.zeros(geo.rho_face.shape),
        generic_current_source=jnp.zeros(geo.rho_face.shape),
        j_bootstrap=jnp.zeros(geo.rho_face.shape),
        j_bootstrap_face=jnp.zeros(geo.rho_face.shape),
        I_bootstrap=jnp.array(0.0),
        Ip=jnp.array(0.0),
        sigma=jnp.zeros(geo.rho_face.shape),
        jtot_hires=jnp.zeros(geo.rho_face.shape),
    )


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
  # Impurity density (currently only 1 supported)
  nimp: cell_variable.CellVariable
  currents: Currents
  q_face: array_typing.ArrayFloat
  s_face: array_typing.ArrayFloat
  nref: array_typing.ScalarFloat  # Reference density
  # pylint: disable=invalid-name
  Zi: array_typing.ScalarFloat  # Main ion charge
  Ai: array_typing.ScalarFloat  # Main ion mass [amu]
  Zimp: array_typing.ScalarFloat  # Impurity charge
  Aimp: array_typing.ScalarFloat  # Impurity mass [amu]
  # pylint: enable=invalid-name

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
        nimp=self.nimp.history_elem(),
        currents=self.currents,
        q_face=self.q_face,
        s_face=self.s_face,
        nref=self.nref,
    )

  def has_nans(self) -> bool:
    """Checks for NaNs in all attributes of CoreProfiles."""

    def _check_for_nans(x: Any) -> bool:
      if isinstance(x, jax.Array):
        return jnp.any(jnp.isnan(x)).item()
      elif isinstance(x, (int, float)):
        return jnp.isnan(x).item()
      elif isinstance(x, Currents):
        return x.has_nans()  # Check for NaNs within nested Currents dataclass
      elif isinstance(x, cell_variable.CellVariable):
        return jnp.any(jnp.isnan(x.value)).item()
      else:
        return False

    return any(
        _check_for_nans(getattr(self, field))
        for field in self.__dataclass_fields__
    )

  def quasineutrality_satisfied(self) -> bool:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.ni.value * self.Zi + self.nimp.value * self.Zimp,
        self.ne.value,
    ).item()

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
        chi_face_ion=jnp.zeros(geo.rho_face.shape),
        chi_face_el=jnp.zeros(geo.rho_face.shape),
        d_face_el=jnp.zeros(geo.rho_face.shape),
        v_face_el=jnp.zeros(geo.rho_face.shape),
    )


@chex.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
  """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
    pressure_thermal_ion_face: Ion thermal pressure on the face grid [Pa]
    pressure_thermal_el_face: Electron thermal pressure on the face grid [Pa]
    pressure_thermal_tot_face: Total thermal pressure on the face grid [Pa]
    pprime_face: Derivative of total pressure with respect to poloidal flux on
      the face grid [Pa/Wb]
    W_thermal_ion: Ion thermal stored energy [J]
    W_thermal_el: Electron thermal stored energy [J]
    W_thermal_tot: Total thermal stored energy [J]
    FFprime_face: FF' on the face grid, where F is the toroidal flux function
    psi_norm_face: Normalized poloidal flux on the face grid [Wb]
    psi_face: Poloidal flux on the face grid [Wb]
    P_heating_tot_ion: Total ion heating power with all sources: auxiliary
      heating + ion-electron exchange + Ohmic + fusion [W]
    P_heating_tot_el: Total electron heating power, with all sources: auxiliary
      heating + ion-electron exchange + Ohmic + fusion [W]
    P_heating_tot: Total heating power, with all sources: auxiliary heating
      + ion-electron exchange + Ohmic + fusion [W]
    P_external_ion: Total external ion heating power: auxiliary heating + Ohmic
      [W]
    P_external_el: Total external electron heating power: auxiliary heating +
      Ohmic [W]
    P_external_tot: Total external heating power: auxiliary heating + Ohmic [W]
    P_ei_exchange_ion: Electron-ion heat exchange power to ions [W]
    P_ei_exchange_el: Electron-ion heat exchange power to electrons [W]
    P_generic_ion: Total generic_ion_el_heat_source power to ions [W]
    P_generic_el: Total generic_ion_el_heat_source power to electrons [W]
    P_generic_tot: Total generic_ion_el_heat power [W]
    P_alpha_ion: Total fusion power to ions [W]
    P_alpha_el: Total fusion power to electrons [W]
    P_alpha_tot: Total fusion power to plasma [W]
    P_ohmic: Ohmic heating power to electrons [W]
    P_brems: Bremsstrahlung electron heat sink [W]
    P_ecrh: Total electron cyclotron source power [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_generic: Total generic source current [A]
    Q_fusion: Fusion power gain
    E_cumulative_fusion: Total cumulative fusion energy [J]
    E_cumulative_external: Total external injected energy
      (Ohmic + auxiliary heating) [J]
  """

  pressure_thermal_ion_face: array_typing.ArrayFloat
  pressure_thermal_el_face: array_typing.ArrayFloat
  pressure_thermal_tot_face: array_typing.ArrayFloat
  pprime_face: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  W_thermal_ion: array_typing.ScalarFloat
  W_thermal_el: array_typing.ScalarFloat
  W_thermal_tot: array_typing.ScalarFloat
  FFprime_face: array_typing.ArrayFloat
  psi_norm_face: array_typing.ArrayFloat
  # psi_face included in post_processed output for convenience, since the
  # CellVariable history method destroys class methods like `face_value`.
  psi_face: array_typing.ArrayFloat
  # Integrated heat sources
  P_heating_tot_ion: array_typing.ScalarFloat
  P_heating_tot_el: array_typing.ScalarFloat
  P_heating_tot: array_typing.ScalarFloat
  P_external_ion: array_typing.ScalarFloat
  P_external_el: array_typing.ScalarFloat
  P_external_tot: array_typing.ScalarFloat
  P_ei_exchange_ion: array_typing.ScalarFloat
  P_ei_exchange_el: array_typing.ScalarFloat
  P_generic_ion: array_typing.ScalarFloat
  P_generic_el: array_typing.ScalarFloat
  P_generic_tot: array_typing.ScalarFloat
  P_alpha_ion: array_typing.ScalarFloat
  P_alpha_el: array_typing.ScalarFloat
  P_alpha_tot: array_typing.ScalarFloat
  P_ohmic: array_typing.ScalarFloat
  P_brems: array_typing.ScalarFloat
  P_ecrh: array_typing.ScalarFloat
  I_ecrh: array_typing.ScalarFloat
  I_generic: array_typing.ScalarFloat
  Q_fusion: array_typing.ScalarFloat
  E_cumulative_fusion: array_typing.ScalarFloat
  E_cumulative_external: array_typing.ScalarFloat
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> PostProcessedOutputs:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
        pressure_thermal_ion_face=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_el_face=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_tot_face=jnp.zeros(geo.rho_face.shape),
        pprime_face=jnp.zeros(geo.rho_face.shape),
        W_thermal_ion=jnp.array(0.0),
        W_thermal_el=jnp.array(0.0),
        W_thermal_tot=jnp.array(0.0),
        FFprime_face=jnp.zeros(geo.rho_face.shape),
        psi_norm_face=jnp.zeros(geo.rho_face.shape),
        psi_face=jnp.zeros(geo.rho_face.shape),
        P_heating_tot_ion=jnp.array(0.0),
        P_heating_tot_el=jnp.array(0.0),
        P_heating_tot=jnp.array(0.0),
        P_external_ion=jnp.array(0.0),
        P_external_el=jnp.array(0.0),
        P_external_tot=jnp.array(0.0),
        P_ei_exchange_ion=jnp.array(0.0),
        P_ei_exchange_el=jnp.array(0.0),
        P_generic_ion=jnp.array(0.0),
        P_generic_el=jnp.array(0.0),
        P_generic_tot=jnp.array(0.0),
        P_alpha_ion=jnp.array(0.0),
        P_alpha_el=jnp.array(0.0),
        P_alpha_tot=jnp.array(0.0),
        P_ohmic=jnp.array(0.0),
        P_brems=jnp.array(0.0),
        P_ecrh=jnp.array(0.0),
        I_ecrh=jnp.array(0.0),
        I_generic=jnp.array(0.0),
        Q_fusion=jnp.array(0.0),
        E_cumulative_fusion=jnp.array(0.0),
        E_cumulative_external=jnp.array(0.0),
    )


@chex.dataclass
class StepperNumericOutputs:
  """Numerical quantities related to the stepper.

  Attributes:
    outer_stepper_iterations: Number of iterations performed in the outer loop
      of the stepper.
    stepper_error_state: 0 if solver converged with fine tolerance for this step
      1 if solver did not converge for this step (was above coarse tol) 2 if
      solver converged within coarse tolerance. Allowed to pass with a warning.
      Occasional error=2 has low impact on final sim state.
    inner_solver_iterations: Total number of iterations performed in the solver
      across all iterations of the stepper.
  """

  outer_stepper_iterations: int = 0
  stepper_error_state: int = 0
  inner_solver_iterations: int = 0


@enum.unique
class SimError(enum.Enum):
  """Integer enum for sim error handling."""

  NO_ERROR = 0
  NAN_DETECTED = 1
  QUASINEUTRALITY_BROKEN = 2


@chex.dataclass
class ToraxSimState:
  """Full simulator state.

  The simulation stepping in sim.py evolves core_profiles which includes all
  the attributes the simulation is advancing. But beyond those, there are
  additional stateful elements which may evolve on each simulation step, such
  as sources and transport.

  This class includes both core_profiles and these additional elements.

  Attributes:
    t: time coordinate.
    dt: timestep interval.
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
    post_processed_outputs: variables for output or intermediate observations
      for overarching workflows, calculated after each simulation step.
    time_step_calculator_state: the state of the TimeStepper.
    stepper_numeric_outputs: Numerical quantities related to the stepper.
  """

  # Time variables.
  t: jax.Array
  dt: jax.Array

  # Profiles evolved or calculated by the simulation.
  core_profiles: CoreProfiles
  core_transport: CoreTransport
  core_sources: source_profiles.SourceProfiles

  # Post-processed outputs after a step.
  post_processed_outputs: PostProcessedOutputs

  # Other "side" states used for logging and feeding to other components of
  # TORAX.
  time_step_calculator_state: Any
  stepper_numeric_outputs: StepperNumericOutputs

  def check_for_errors(self) -> SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.has_nans():
      return SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return SimError.QUASINEUTRALITY_BROKEN
    else:
      return SimError.NO_ERROR
