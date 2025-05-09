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
from torax import jax_utils
from torax.config import config_args
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.sources import source_profiles
import typing_extensions


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
  external_current_source: array_typing.ArrayFloat
  j_bootstrap: array_typing.ArrayFloat
  j_bootstrap_face: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  # Using physics notation naming convention
  Ip_profile_face: array_typing.ArrayFloat  # [A]
  sigma: array_typing.ArrayFloat
  jtot_hires: Optional[array_typing.ArrayFloat] = None

  @property
  def Ip_total(self) -> array_typing.ScalarFloat:
    """Returns the total plasma current [A]."""
    return self.Ip_profile_face[..., -1]

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> "Currents":
    """Returns a Currents with all zeros."""
    return cls(
        jtot=jnp.zeros(geo.rho_face.shape),
        jtot_face=jnp.zeros(geo.rho_face.shape),
        johm=jnp.zeros(geo.rho_face.shape),
        external_current_source=jnp.zeros(geo.rho_face.shape),
        j_bootstrap=jnp.zeros(geo.rho_face.shape),
        j_bootstrap_face=jnp.zeros(geo.rho_face.shape),
        Ip_profile_face=jnp.zeros(geo.rho_face.shape),
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

  Attributes:
      temp_ion: Ion temperature [keV].
      temp_el: Electron temperature [keV].
      psi: Poloidal flux [Wb].
      psidot: Time derivative of poloidal flux (loop voltage) [V].
      n_e: Electron density [density_reference m^-3].
      ni: Main ion density [density_reference m^-3].
      nimp: Impurity density [density_reference m^-3].
      currents: Instance of the Currents dataclass.
      q_face: Safety factor.
      s_face: Magnetic shear.
      density_reference: Reference density [m^-3].
      vloop_lcfs: Loop voltage at LCFS (V).
      Zi: Main ion charge on cell grid [dimensionless].
      Zi_face: Main ion charge on face grid [dimensionless].
      Ai: Main ion mass [amu].
      Zimp: Impurity charge on cell grid [dimensionless].
      Zimp_face: Impurity charge on face grid [dimensionless].
      Aimp: Impurity mass [amu].
  """

  temp_ion: cell_variable.CellVariable
  temp_el: cell_variable.CellVariable
  psi: cell_variable.CellVariable
  psidot: cell_variable.CellVariable
  n_e: cell_variable.CellVariable
  ni: cell_variable.CellVariable
  nimp: cell_variable.CellVariable
  currents: Currents
  q_face: array_typing.ArrayFloat
  s_face: array_typing.ArrayFloat
  density_reference: array_typing.ScalarFloat
  vloop_lcfs: array_typing.ScalarFloat
  # pylint: disable=invalid-name
  Zi: array_typing.ArrayFloat
  Zi_face: array_typing.ArrayFloat
  Ai: array_typing.ScalarFloat
  Zimp: array_typing.ArrayFloat
  Zimp_face: array_typing.ArrayFloat
  Aimp: array_typing.ScalarFloat
  # pylint: enable=invalid-name

  def quasineutrality_satisfied(self) -> bool:
    """Checks if quasineutrality is satisfied."""
    return jnp.allclose(
        self.ni.value * self.Zi + self.nimp.value * self.Zimp,
        self.n_e.value,
    ).item()

  def negative_temperature_or_density(self) -> bool:
    """Checks if any temperature or density is negative."""
    profiles_to_check = (
        self.temp_ion,
        self.temp_el,
        self.n_e,
        self.ni,
        self.nimp,
    )
    return any(
        [jnp.any(jnp.less(x, 0.0)) for x in jax.tree.leaves(profiles_to_check)]
    )

  def index(self, i: int) -> typing_extensions.Self:
    """If the CoreProfiles is a history, returns the i-th CoreProfiles."""
    idx = lambda x: x[i]
    state = jax.tree_util.tree_map(idx, self)
    # These variables track whether they are histories, so when we collapse down
    # to a single state we need to explicitly clear the history flag.
    history_vars = ["temp_ion", "temp_el", "psi", "psidot", "n_e", "ni"]
    history_replace = {"history": None}
    replace_dict = {var: history_replace for var in history_vars}
    state = config_args.recursive_replace(state, **replace_dict)
    return state

  def sanity_check(self):
    for field in CoreProfiles.__dataclass_fields__:
      value = getattr(self, field)
      if hasattr(value, "sanity_check"):
        value.sanity_check()

  def __str__(self) -> str:
    return f"""
      CoreProfiles(
        temp_ion={self.temp_ion},
        temp_el={self.temp_el},
        psi={self.psi},
        n_e={self.n_e},
        nimp={self.nimp},
        ni={self.ni},
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


@chex.dataclass(frozen=True, eq=False)
class PostProcessedOutputs:
  """Collection of outputs calculated after each simulation step.

  These variables are not used internally, but are useful as outputs or
  intermediate observations for overarching workflows.

  Attributes:
    pressure_thermal_i: Ion thermal pressure on the face grid [Pa]
    pressure_thermal_e: Electron thermal pressure on the face grid [Pa]
    pressure_thermal_total: Total thermal pressure on the face grid [Pa]
    pprime: Derivative of total pressure with respect to poloidal flux on
      the face grid [Pa/Wb]
    W_thermal_i: Ion thermal stored energy [J]
    W_thermal_e: Electron thermal stored energy [J]
    W_thermal_total: Total thermal stored energy [J]
    tau_E: Thermal energy confinement time [s]
    H89P: L-mode confinement quality factor with respect to the ITER89P scaling
      law derived from the ITER L-mode confinement database
    H98: H-mode confinement quality factor with respect to the ITER98y2 scaling
      law derived from the ITER H-mode confinement database
    H97L: L-mode confinement quality factor with respect to the ITER97L scaling
      law derived from the ITER L-mode confinement database
    H20: H-mode confinement quality factor with respect to the ITER20 scaling
      law derived from the updated (2020) ITER H-mode confinement database
    FFprime: FF' on the face grid, where F is the toroidal flux function
    psi_norm: Normalized poloidal flux on the face grid [Wb]
    P_SOL_i: Total ion heating power exiting the plasma with all sources:
      auxiliary heating + ion-electron exchange + fusion [W]
    P_SOL_e: Total electron heating power exiting the plasma with all sources
      and sinks: auxiliary heating + ion-electron exchange + Ohmic + fusion +
      radiation sinks [W]
    P_SOL_total: Total heating power exiting the plasma with all sources and
      sinks
    P_external_ion: Total external ion heating power: auxiliary heating + Ohmic
      [W]
    P_external_el: Total external electron heating power: auxiliary heating +
      Ohmic [W]
    P_external_tot: Total external heating power: auxiliary heating + Ohmic [W]
    P_external_injected: Total external injected power before absorption [W]
    P_ei_exchange_i: Electron-ion heat exchange power to ions [W]
    P_ei_exchange_e: Electron-ion heat exchange power to electrons [W]
    P_aux_generic_i: Total generic_ion_el_heat_source power to ions [W]
    P_aux_generic_e: Total generic_ion_el_heat_source power to electrons [W]
    P_aux_generic_total: Total generic_ion_el_heat power [W]
    P_alpha_i: Total fusion power to ions [W]
    P_alpha_e: Total fusion power to electrons [W]
    P_alpha_total: Total fusion power to plasma [W]
    P_ohmic_e: Ohmic heating power to electrons [W]
    P_bremsstrahlung_e: Bremsstrahlung electron heat sink [W]
    P_cyclotron_e: Cyclotron radiation electron heat sink [W]
    P_ecrh_e: Total electron cyclotron source power [W]
    P_radiation_e: Impurity radiation heat sink [W]
    I_ecrh: Total electron cyclotron source current [A]
    I_aux_generic: Total generic source current [A]
    Q_fusion: Fusion power gain
    P_icrh_e: Ion cyclotron resonance heating to electrons [W]
    P_icrh_i: Ion cyclotron resonance heating to ions [W]
    P_icrh_total: Total ion cyclotron resonance heating power [W]
    P_LH_high_density: H-mode transition power for high density branch [W]
    P_LH_min: Minimum H-mode transition power for at n_e_min_P_LH [W]
    P_LH: H-mode transition power from maximum of P_LH_high_density and P_LH_min
      [W]
    n_e_min_P_LH: Density corresponding to the P_LH_min [density_reference]
    E_fusion: Total cumulative fusion energy [J]
    E_aux: Total external injected energy (Ohmic + auxiliary heating)
      [J]
    T_e_volume_avg: Volume average electron temperature [keV]
    T_i_volume_avg: Volume average ion temperature [keV]
    n_e_volume_avg: Volume average electron density [density_reference m^-3]
    n_e_volume_avg: Volume average electron density [density_reference m^-3]
    n_i_volume_avg: Volume average main ion density [density_reference m^-3]
    n_e_line_avg: Line averaged electron density [density_reference m^-3]
    n_i_line_avg: Line averaged main ion density [density_reference m^-3]
    fgw_n_e_volume_avg: Greenwald fraction from volume-averaged electron density
      [dimensionless]
    fgw_n_e_line_avg: Greenwald fraction from line-averaged electron density
      [dimensionless]
    q95: q at 95% of the normalized poloidal flux
    W_pol: Total magnetic energy [J]
    li3: Normalized plasma internal inductance, ITER convention [dimensionless]
    dW_thermal_dt: Time derivative of the total stored thermal energy [W]
    q_min: Minimum q value
    rho_q_min: rho_norm at the minimum q
    rho_q_3_2_first: First outermost rho_norm value that intercepts the
      q=3/2 plane. If no intercept is found, set to -inf.
    rho_q_2_1_first: First outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_first: First outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_2_second: Second outermost rho_norm value that intercepts the
      q=3/2 plane. If no intercept is found, set to -inf.
    rho_q_2_1_second: Second outermost rho_norm value that intercepts the q=2/1
      plane. If no intercept is found, set to -inf.
    rho_q_3_1_second: Second outermost rho_norm value that intercepts the q=3/1
      plane. If no intercept is found, set to -inf.
    I_bootstrap: Total bootstrap current [A]
  """

  pressure_thermal_i: array_typing.ArrayFloat
  pressure_thermal_e: array_typing.ArrayFloat
  pressure_thermal_total: array_typing.ArrayFloat
  pprime: array_typing.ArrayFloat
  # pylint: disable=invalid-name
  W_thermal_i: array_typing.ScalarFloat
  W_thermal_e: array_typing.ScalarFloat
  W_thermal_total: array_typing.ScalarFloat
  tau_E: array_typing.ScalarFloat
  H89P: array_typing.ScalarFloat
  H98: array_typing.ScalarFloat
  H97L: array_typing.ScalarFloat
  H20: array_typing.ScalarFloat
  FFprime: array_typing.ArrayFloat
  psi_norm: array_typing.ArrayFloat
  # Integrated heat sources
  P_SOL_i: array_typing.ScalarFloat
  P_SOL_e: array_typing.ScalarFloat
  P_SOL_total: array_typing.ScalarFloat
  P_external_ion: array_typing.ScalarFloat
  P_external_el: array_typing.ScalarFloat
  P_external_tot: array_typing.ScalarFloat
  P_external_injected: array_typing.ScalarFloat
  P_ei_exchange_i: array_typing.ScalarFloat
  P_ei_exchange_e: array_typing.ScalarFloat
  P_aux_generic_i: array_typing.ScalarFloat
  P_aux_generic_e: array_typing.ScalarFloat
  P_aux_generic_total: array_typing.ScalarFloat
  P_alpha_i: array_typing.ScalarFloat
  P_alpha_e: array_typing.ScalarFloat
  P_alpha_total: array_typing.ScalarFloat
  P_ohmic_e: array_typing.ScalarFloat
  P_bremsstrahlung_e: array_typing.ScalarFloat
  P_cyclotron_e: array_typing.ScalarFloat
  P_ecrh_e: array_typing.ScalarFloat
  P_radiation_e: array_typing.ScalarFloat
  I_ecrh: array_typing.ScalarFloat
  I_aux_generic: array_typing.ScalarFloat
  Q_fusion: array_typing.ScalarFloat
  P_icrh_e: array_typing.ScalarFloat
  P_icrh_i: array_typing.ScalarFloat
  P_icrh_total: array_typing.ScalarFloat
  P_LH_high_density: array_typing.ScalarFloat
  P_LH_min: array_typing.ScalarFloat
  P_LH: array_typing.ScalarFloat
  n_e_min_P_LH: array_typing.ScalarFloat
  E_fusion: array_typing.ScalarFloat
  E_aux: array_typing.ScalarFloat
  T_e_volume_avg: array_typing.ScalarFloat
  T_i_volume_avg: array_typing.ScalarFloat
  n_e_volume_avg: array_typing.ScalarFloat
  n_i_volume_avg: array_typing.ScalarFloat
  n_e_line_avg: array_typing.ScalarFloat
  n_i_line_avg: array_typing.ScalarFloat
  fgw_n_e_volume_avg: array_typing.ScalarFloat
  fgw_n_e_line_avg: array_typing.ScalarFloat
  q95: array_typing.ScalarFloat
  W_pol: array_typing.ScalarFloat
  li3: array_typing.ScalarFloat
  dW_thermal_dt: array_typing.ScalarFloat
  rho_q_min: array_typing.ScalarFloat
  q_min: array_typing.ScalarFloat
  rho_q_3_2_first: array_typing.ScalarFloat
  rho_q_3_2_second: array_typing.ScalarFloat
  rho_q_2_1_first: array_typing.ScalarFloat
  rho_q_2_1_second: array_typing.ScalarFloat
  rho_q_3_1_first: array_typing.ScalarFloat
  rho_q_3_1_second: array_typing.ScalarFloat
  I_bootstrap: array_typing.ScalarFloat
  # pylint: enable=invalid-name

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a PostProcessedOutputs with all zeros, used for initializing."""
    return cls(
        pressure_thermal_i=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_e=jnp.zeros(geo.rho_face.shape),
        pressure_thermal_total=jnp.zeros(geo.rho_face.shape),
        pprime=jnp.zeros(geo.rho_face.shape),
        W_thermal_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_thermal_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        tau_E=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H89P=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H98=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H97L=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        H20=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        FFprime=jnp.zeros(geo.rho_face.shape),
        psi_norm=jnp.zeros(geo.rho_face.shape),
        P_SOL_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_SOL_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_SOL_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_ion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_el=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_tot=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_external_injected=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ei_exchange_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_aux_generic_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_alpha_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ohmic_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_bremsstrahlung_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_cyclotron_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_ecrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_radiation_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_ecrh=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_aux_generic=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        Q_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_i=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_e=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_icrh_total=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_high_density=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_min_P_LH=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_fusion=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        E_aux=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        T_i_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_i_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_e_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        n_i_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_n_e_volume_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        fgw_n_e_line_avg=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q95=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        W_pol=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        li3=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        dW_thermal_dt=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        q_min=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_first=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_2_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_2_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        rho_q_3_1_second=jnp.array(0.0, dtype=jax_utils.get_dtype()),
        I_bootstrap=jnp.array(0.0, dtype=jax_utils.get_dtype()),
    )

  def check_for_errors(self):
    if has_nan(self):
      return SimError.NAN_DETECTED
    else:
      return SimError.NO_ERROR


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
  """

  outer_solver_iterations: int = 0
  solver_error_state: int = 0
  inner_solver_iterations: int = 0


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
    core_sources: Profiles for all sources/sinks. These are the profiles that
      are used to calculate the coefficients for the t+dt time step. For the
      explicit sources, these are calculated at the start of the time step, so
      are the values at time t. For the implicit sources, these are the most
      recent guess for time t+dt. The profiles here are the merged version of
      the explicit and implicit profiles.
    post_processed_outputs: variables for output or intermediate observations
      for overarching workflows, calculated after each simulation step.
    geometry: Geometry at this time step used for the simulation.
    time_step_calculator_state: the state of the TimeStepper.
    solver_numeric_outputs: Numerical quantities related to the solver.
    sawtooth_crash: True if a sawtooth model is active and the state
      corresponds to a post-sawtooth-crash state.
  """

  t: jax.Array
  dt: jax.Array
  core_profiles: CoreProfiles
  core_transport: CoreTransport
  core_sources: source_profiles.SourceProfiles
  geometry: geometry.Geometry
  solver_numeric_outputs: SolverNumericOutputs
  sawtooth_crash: bool = False

  def check_for_errors(self) -> SimError:
    """Checks for errors in the simulation state."""
    if self.core_profiles.negative_temperature_or_density():
      logging.info("%s", self.core_profiles)
      log_negative_profile_names(self.core_profiles)
      return SimError.NEGATIVE_CORE_PROFILES
    # If there are NaNs that occured without negative core profiles, log this
    # as a separate error.
    if has_nan(self):
      logging.info("%s", self.core_profiles)
      return SimError.NAN_DETECTED
    elif not self.core_profiles.quasineutrality_satisfied():
      return SimError.QUASINEUTRALITY_BROKEN
    else:
      return SimError.NO_ERROR


def has_nan(inputs: ToraxSimState | PostProcessedOutputs) -> bool:
  return any([jnp.any(jnp.isnan(x)) for x in jax.tree.leaves(inputs)])


def log_negative_profile_names(inputs: CoreProfiles):
  path_vals, _ = jax.tree.flatten_with_path(inputs)
  for path, value in path_vals:
    if jnp.any(jnp.less(value, 0.0)):
      logging.info("Found negative value in %s", jax.tree_util.keystr(path))


def check_for_errors(
    sim_state: ToraxSimState,
    post_processed_outputs: PostProcessedOutputs,
) -> SimError:
  """Checks for errors in the simulation state."""
  state_error = sim_state.check_for_errors()
  if state_error != SimError.NO_ERROR:
    return state_error
  else:
    return post_processed_outputs.check_for_errors()
