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

"""Source/sink profiles for all the sources in TORAX."""

from __future__ import annotations

import dataclasses

import chex
import jax.numpy as jnp
from torax import config_slice
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state as state_module
from torax.fvm import diffusion_terms
from torax.sources import bootstrap_current_source
from torax.sources import electron_density_sources
from torax.sources import external_current_source
from torax.sources import fusion_heat_source as fusion_heat_source_lib
from torax.sources import generic_ion_el_heat_source as generic_ion_el_heat_source_lib
from torax.sources import qei_source as qei_source_lib
from torax.sources import source as source_lib
from torax.sources import source_config


@chex.dataclass(frozen=True)
class SourceProfiles:
  """Collection of profiles for all sources in TORAX.

  Most profiles are stored in the `profiles` attribute, but special-case
  profiles are pulled out into their own attributes.

  The keys of profiles match the keys of the sources in the sources.Sources
  object used to compute them.
  """

  j_bootstrap: bootstrap_current_source.BootstrapCurrentProfile
  profiles: dict[str, jnp.ndarray]


def build_source_profiles(
    sources: Sources,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
    explicit: bool,
) -> SourceProfiles:
  """Builds explicit or implicit source profiles.

  Args:
    sources: All TORAX sources/sinks.
    dynamic_config_slice: Input config for this time step. Can change from time
      step to time step.
    geo: Geometry of the torus.
    sim_state: Full TORAX sim state which includes the mesh state, either at the
      start of the time step (if explicit) or the live state being evolved
      during the time step (if implicit).
    explicit: If True, this function should return profiles for all explicit
      sources. All implicit sources should be set to 0. And same vice versa.

  Returns:
    SourceProfiles for either explicit or implicit sources (and all others set
    to zero).
  """
  # Bootstrap current is a special-case source with multiple outputs, so handle
  # it here.
  # TODO( b/314308399): Add a new neoclassical directory with
  # different ways to compute sigma and bootstrap current.
  bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_config_slice,
      geo,
      sim_state,
      sources.j_bootstrap,
      explicit,
  )
  other_profiles = {}
  other_profiles.update(
      _build_psi_profiles(
          dynamic_config_slice, geo, sim_state, sources, explicit
      )
  )
  other_profiles.update(
      _build_ne_profiles(
          dynamic_config_slice, geo, sim_state, sources, explicit
      )
  )
  other_profiles.update(
      _build_temp_ion_el_profiles(
          dynamic_config_slice,
          geo,
          sim_state,
          sources,
          explicit,
      )
  )
  return SourceProfiles(
      j_bootstrap=bootstrap_profiles,
      profiles=other_profiles,
  )


def _build_bootstrap_profiles(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
    j_bootstrap_source: bootstrap_current_source.BootstrapCurrentSource,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> bootstrap_current_source.BootstrapCurrentProfile:
  """Computes the bootstrap current profile.

  Args:
    dynamic_config_slice: Input config for this time step. Can change from time
      step to time step.
    geo: Geometry of the torus.
    sim_state: Full TORAX sim state which includes the mesh state, either at the
      start of the time step (if explicit) or the live state being evolved
      during the time step (if implicit).
    j_bootstrap_source: Bootstrap current source used to compute the profile.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and the bootstrap current source is not
      explicit, then this should return all zeros. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit

  Returns:
    Bootstrap current profile.
  """
  bootstrap_profile = j_bootstrap_source.get_value(
      dynamic_config_slice=dynamic_config_slice,
      geo=geo,
      sim_state=sim_state,
  )
  sigma = jax_utils.select(
      jnp.logical_or(
          explicit
          == dynamic_config_slice.sources[j_bootstrap_source.name].is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.sigma,
      jnp.zeros_like(bootstrap_profile.sigma),
  )
  j_bootstrap = jax_utils.select(
      jnp.logical_or(
          explicit
          == dynamic_config_slice.sources[j_bootstrap_source.name].is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap,
      jnp.zeros_like(bootstrap_profile.j_bootstrap),
  )
  j_bootstrap_face = jax_utils.select(
      jnp.logical_or(
          explicit
          == dynamic_config_slice.sources[j_bootstrap_source.name].is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap_face,
      jnp.zeros_like(bootstrap_profile.j_bootstrap_face),
  )
  I_bootstrap = jax_utils.select(  # pylint: disable=invalid-name
      jnp.logical_or(
          explicit
          == dynamic_config_slice.sources[j_bootstrap_source.name].is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.I_bootstrap,
      jnp.zeros_like(bootstrap_profile.I_bootstrap),
  )
  return bootstrap_current_source.BootstrapCurrentProfile(
      sigma=sigma,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )


def _build_psi_profiles(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
    sources: Sources,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> dict[str, jnp.ndarray]:
  """Computes psi sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_config_slice: Input config for this time step. Can change from time
      step to time step.
    geo: Geometry of the torus.
    sim_state: Full TORAX sim state which includes the mesh state, either at the
      start of the time step (if explicit) or the live state being evolved
      during the time step (if implicit).
    sources: Collection of all TORAX sources.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and a given source is not explicit, then this
      function will return zeros for that source. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).
    calculate_anyway: If True, returns values regardless of explicit

  Returns:
    dict of psi source profiles.
  """
  psi_profiles = {}
  # jext is precomputed in the initial state and does not change.
  psi_profiles[sources.jext.name] = jax_utils.select(
      jnp.logical_or(
          explicit
          == dynamic_config_slice.sources[sources.jext.name].is_explicit,
          calculate_anyway,
      ),
      sim_state.mesh_state.currents.jext,
      jnp.zeros_like(geo.r),
  )
  # Iterate through the rest of the sources and compute profiles for the ones
  # which relate to psi. jext is not part of the "standard sources."
  for source_name, source in sources.psi_sources.items():
    dynamic_source_config = dynamic_config_slice.sources[source_name]
    psi_profiles[source_name] = jax_utils.select(
        jnp.logical_or(
            explicit == dynamic_source_config.is_explicit, calculate_anyway
        ),
        source.get_value(
            dynamic_source_config.source_type,
            dynamic_config_slice,
            geo,
            sim_state,
        ),
        jnp.zeros_like(geo.r),
    )
  return psi_profiles


def _build_ne_profiles(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
    sources: Sources,
    explicit: bool,
) -> dict[str, jnp.ndarray]:
  """Computes ne sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_config_slice: Input config for this time step. Can change from time
      step to time step.
    geo: Geometry of the torus.
    sim_state: Full TORAX sim state which includes the mesh state, either at the
      start of the time step (if explicit) or the live state being evolved
      during the time step (if implicit).
    sources: Collection of all TORAX sources.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and a given source is not explicit, then this
      function will return zeros for that source. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).

  Returns:
    dict of ne source profiles.
  """
  ne_profiles = {}
  # Iterate through the sources and compute profiles for the ones which relate
  # to ne.
  for source_name, source in sources.ne_sources.items():
    dynamic_source_config = dynamic_config_slice.sources[source_name]
    ne_profiles[source_name] = jax_utils.select(
        explicit == dynamic_source_config.is_explicit,
        source.get_value(
            dynamic_source_config.source_type,
            dynamic_config_slice,
            geo,
            sim_state,
        ),
        jnp.zeros_like(geo.r),
    )
  return ne_profiles


def _build_temp_ion_el_profiles(
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
    sources: Sources,
    explicit: bool,
) -> dict[str, jnp.ndarray]:
  """Computes ion and el sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_config_slice: Input config for this time step. Can change from time
      step to time step.
    geo: Geometry of the torus.
    sim_state: Full TORAX sim state which includes the mesh state, either at the
      start of the time step (if explicit) or the live state being evolved
      during the time step (if implicit).
    sources: Collection of all TORAX sources.
    explicit: If True, this function should return the profile for an explicit
      source. If explicit is True and a given source is not explicit, then this
      function will return zeros for that source. And same with implicit (if
      explicit=False and the source is set to be explicit, then this will return
      all zeros).

  Returns:
    dict of temp ion and temp el source profiles.
  """
  ion_el_profiles = {}
  # Calculate other ion and el heat sources/sinks.
  temp_ion_el_sources = sources.temp_ion_sources | sources.temp_el_sources
  for source_name, source in temp_ion_el_sources.items():
    zeros = jnp.zeros(
        source.output_shape_getter(dynamic_config_slice, geo, sim_state)
    )
    dynamic_source_config = dynamic_config_slice.sources[source_name]
    ion_el_profiles[source_name] = jax_utils.select(
        explicit == dynamic_source_config.is_explicit,
        source.get_value(
            dynamic_source_config.source_type,
            dynamic_config_slice,
            geo,
            sim_state,
        ),
        zeros,
    )
  return ion_el_profiles


def sum_sources_psi(
    sources: Sources,
    source_profile: SourceProfiles,
    geo: geometry.Geometry,
    Rmaj: float,  # pylint: disable=invalid-name
) -> jnp.ndarray:
  """Computes psi source values for sim.calc_coeffs."""
  total = (
      source_profile.j_bootstrap.j_bootstrap
      + source_profile.profiles[sources.jext.name]
  )
  for source_name, source in sources.psi_sources.items():
    total += source.get_profile_for_affected_state(
        profile=source_profile.profiles[source_name],
        affected_mesh_state=source_lib.AffectedMeshStateAttribute.PSI.value,
        geo=geo,
    )
  denom = 2 * jnp.pi * Rmaj * geo.J**2
  scale_source = lambda src: -geo.vpr * src * constants.CONSTANTS.mu0 / denom
  return scale_source(total)


def sum_sources_ne(
    sources: Sources,
    source_profile: SourceProfiles,
    geo: geometry.Geometry,
) -> jnp.ndarray:
  """Computes ne source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in sources.ne_sources.items():
    total += source.get_profile_for_affected_state(
        profile=source_profile.profiles[source_name],
        affected_mesh_state=source_lib.AffectedMeshStateAttribute.NE.value,
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_ion(
    sources: Sources,
    source_profile: SourceProfiles,
    geo: geometry.Geometry,
) -> jnp.ndarray:
  """Computes temp_ion source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in sources.temp_ion_sources.items():
    total += source.get_profile_for_affected_state(
        profile=source_profile.profiles[source_name],
        affected_mesh_state=(
            source_lib.AffectedMeshStateAttribute.TEMP_ION.value
        ),
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_el(
    sources: Sources,
    source_profile: SourceProfiles,
    geo: geometry.Geometry,
) -> jnp.ndarray:
  """Computes temp_el source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in sources.temp_el_sources.items():
    total += source.get_profile_for_affected_state(
        profile=source_profile.profiles[source_name],
        affected_mesh_state=(
            source_lib.AffectedMeshStateAttribute.TEMP_EL.value
        ),
        geo=geo,
    )
  return total * geo.vpr


def calc_and_sum_sources_psi(
    sources: Sources,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes sum of psi sources for psi_dot calculation."""

  # TODO(b/323504363): Revisit how to calculate this once we enable more
  # expensive source functions that might not jittable (like file-based or
  # RPC-based sources).
  psi_profiles = _build_psi_profiles(
      dynamic_config_slice, geo, sim_state, sources, calculate_anyway=True
  )
  total = 0
  for key in psi_profiles:
    total += psi_profiles[key]
  j_bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_config_slice,
      geo,
      sim_state,
      sources.j_bootstrap,
      calculate_anyway=True,
  )
  total += j_bootstrap_profiles.j_bootstrap
  denom = 2 * jnp.pi * dynamic_config_slice.Rmaj * geo.J**2
  scale_source = lambda src: -geo.vpr * src * constants.CONSTANTS.mu0 / denom

  return scale_source(total), j_bootstrap_profiles.sigma


def calc_psidot(
    sources: Sources,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
) -> jnp.ndarray:
  r"""Calculates psidot (loop voltage). Used for the Ohmic electron heat source.

  psidot is an interesting TORAX output, and is thus also saved in state.

  psidot = \partial psi / \partial t, and is derived from the same components
  that form the psi block in the coupled PDE equations. This, a similar
  (but abridged) formulation as in sim.calc_coeffs and fvm._calc_c is used here

  Args:
    sources: All TORAX source/sinks.
    dynamic_config_slice: Simulation configuration at this timestep
    geo: Torus geometry
    sim_state: Full TORAX simulation state including plasma state r

  Returns:
    psidot: on cell grid
  """
  consts = constants.CONSTANTS

  psi_sources, sigma = calc_and_sum_sources_psi(
      sources,
      dynamic_config_slice,
      geo,
      sim_state,
  )
  toc_psi = (
      1.0
      / dynamic_config_slice.resistivity_mult
      * geo.r
      * sigma
      * consts.mu0
      / geo.J**2
      / dynamic_config_slice.Rmaj
  )
  d_face_psi = geo.G2_face / geo.J_face / geo.rmax**2

  c_mat, c = diffusion_terms.make_diffusion_terms(
      d_face_psi, sim_state.mesh_state.psi
  )
  c += psi_sources

  psidot = (jnp.dot(c_mat, sim_state.mesh_state.psi.value) + c) / toc_psi

  return psidot


#  OhmicHeatSource is a special case and defined here to avoid circular
#  dependencies, since it depends on the psi sources
def _ohmic_heat_model(
    sources: Sources,
    dynamic_config_slice: config_slice.DynamicConfigSlice,
    geo: geometry.Geometry,
    sim_state: state_module.ToraxSimState,
) -> jnp.ndarray:
  """Returns the Ohmic source for electron heat equation."""
  jtot, _ = physics.calc_jtot_from_psi(
      geo,
      sim_state.mesh_state.psi,
      dynamic_config_slice.Rmaj,
  )

  psidot = calc_psidot(sources, dynamic_config_slice, geo, sim_state)

  pohm = jtot * psidot / (2 * jnp.pi * dynamic_config_slice.Rmaj)
  return pohm


@dataclasses.dataclass(frozen=True, kw_only=True)
class OhmicHeatSource(source_lib.SingleProfileSource):
  """Ohmic heat source for electron heat equation.

  Pohm = jtor * psidot /(2*pi*Rmaj), related to electric power formula P = IV
  """

  # Users must pass in a pointer to the complete set of sources to this object.
  sources: Sources

  name: str = 'ohmic_heat_source'

  supported_types: tuple[source_config.SourceType, ...] = (
      source_config.SourceType.ZERO,
      source_config.SourceType.MODEL_BASED,
  )

  # Freeze these params and do not include them in the __init__.
  affected_mesh_states: tuple[source_lib.AffectedMeshStateAttribute, ...] = (
      dataclasses.field(
          init=False,
          default=(source_lib.AffectedMeshStateAttribute.TEMP_EL,),
      )
  )
  model_func: source_config.SourceProfileFunction | None = dataclasses.field(
      init=False,
      default_factory=lambda: None,  # ignored.
  )

  def __post_init__(self):
    # Ignore the model provided above and set it to the function here.
    def _model_func(
        dynamic_config_slice: config_slice.DynamicConfigSlice,
        geo: geometry.Geometry,
        sim_state: state_module.ToraxSimState,
    ) -> jnp.ndarray:
      return _ohmic_heat_model(
          sources=self.sources,
          dynamic_config_slice=dynamic_config_slice,
          geo=geo,
          sim_state=sim_state,
      )

    # Must use object.__setattr__ instead of simply doing
    # self.model_func = _model_func
    # because this class is a frozen dataclass. Frozen classes cannot set any
    # self attributes after init, but this is a workaround. We cannot set the
    # model_func in the dataclass field above either because we need access to
    # self in the implementation.
    object.__setattr__(self, 'model_func', _model_func)


class Sources:
  """Source/sink terms for the different equations being evolved in Torax.

  Each source/sink (all called sources as the difference is only a sign change)
  can be explicit or implicit and signal to our solver on how to handle these
  terms. Their values are provided via model, file, perscribed function, etc.
  The specific approach used depends on how the source is initialized and what
  runtime configuration inputs are provided.

  NOTE for RAPTOR users: Sources in TORAX are similar to actuators in RAPTOR and
  provide a hook to provide custom profiles.

  You can both override the default set of sources in TORAX as well as define
  new custom sources inline when constructing this object. The example below
  shows how to define a new custom electron-density source.

  ```python
  # Define an electron-density source with a Gaussian profile.
  my_custom_source_name = 'custom_ne_source'
  my_custom_source = source.SingleProfileSource(
      name=my_custom_source_name,
      supported_types=(
          source_config.SourceType.ZERO,
          source_config.SourceType.FORMULA_BASED,
      ),
      affected_mesh_states=source.AffectedMeshStateAttribute.NE,
      formula=formulas.Gaussian(my_custom_source_name),
  )
  all_torax_sources = source_profiles.Sources(
      additional_sources=[
          my_custom_source,
      ]
  )
  ```

  You must also include a runtime config for the new custom source:

  ```python
  my_torax_config = config.Config(
      sources=dict(
          ...  # Configs for other sources.
          # Set some params for the new source
          custom_ne_source=source_config.SourceConfig(
              source_type=source_config.SourceType.FORMULA_BASED,
              formula=formula_config.FormulaConfig(
                  gaussian=formula_config.Gaussian(
                      total=1.0,
                      c1=2.0,
                      c2=3.0,
                  ),
              ),
          ),
      ),
  )
  ```

  See source_config.py for more details on how to configure all the source/sink
  terms.
  """

  def __init__(
      self,
      *,
      # All arguments must be provided as keyword arguments to ensure that
      # everything is set explicitly. Helps avoid unwarranted mistakes.
      # The sources below are on by default, which is why they are exposed
      # directly in the constructor.
      # The sources listed below are the default sources that are turned on as
      # well by default.
      # Current sources (for psi equation)
      j_bootstrap: (
          bootstrap_current_source.BootstrapCurrentSource | None
      ) = None,
      jext: external_current_source.ExternalCurrentSource | None = None,
      # Electron density sources/sink (for the ne equation).
      gas_puff_source: electron_density_sources.GasPuffSource | None = None,
      nbi_particle_source: (
          electron_density_sources.NBIParticleSource | None
      ) = None,
      pellet_source: electron_density_sources.PelletSource | None = None,
      # Ion and electron heat sources (for the temp-ion and temp-el eqs).
      generic_ion_el_heat_source: (
          generic_ion_el_heat_source_lib.GenericIonElectronHeatSource | None
      ) = None,
      fusion_heat_source: fusion_heat_source_lib.FusionHeatSource | None = None,
      ohmic_heat_source: OhmicHeatSource | None = None,
      qei_source: qei_source_lib.QeiSource | None = None,
      # Any additional sources that the user wants to provide.
      additional_sources: list[source_lib.Source] | None = None,
  ):
    """Constructs a collection of sources.

    This class defines which sources are available in a TORAX simulation run.
    Users can configure whether each source is actually on and what kind of
    profile it produces by changing its runtime configuration (see
    source_config.py).

    Some TORAX sources are required and on by default. These sources are in the
    argument list of this `__init__()` function. While these sources are on by
    default, they can be turned off by setting the source to ZERO.

    For example, to turn off the gas-puff source:

    ```python
    sources = source_profiles.Sources()
    my_torax_config = config.Config(
        sources=dict(
            gas_puff_source=source_config.SourceConfig(
                source_type=source_config.SourceType.ZERO,
            ),
        ),
    )
    ```

    Args:
      j_bootstrap: Bootstrap current density source for the psi equation. Is a
        "neoclassical" source.
      jext: External current density source for the psi equation.
      gas_puff_source: Gas puff particle source for the electron density ne
        equation.
      nbi_particle_source: Neutral beam injection particle source for the
        electron density ne equation.
      pellet_source: Pellet source for the electron density ne equation.
      generic_ion_el_heat_source: Generic heat source coupled for both the ion
        and electron heat equations.
      fusion_heat_source: Alpha heat source for coupled for both the ion and
        electron heat equations.
      ohmic_heat_source: Ohmic heating for electron temperatures.
      qei_source: Collisional ion-electron heat source. Special-case source used
        in both the explicit and implicit terms in the TORAX solver.
      additional_sources: Optional list of additional sources to include in
        TORAX. Remember that all additional sources need their corresponding
        runtime config to be included in config.Config(). All these additional
        sources are "standard" sources (they are not going to be treated as
        special cases like j_bootstrap, jext, and qei_source are). They will be
        accessible via the standard_sources property.
    """
    self._j_bootstrap = (
        bootstrap_current_source.BootstrapCurrentSource()
        if j_bootstrap is None
        else j_bootstrap
    )
    self._qei_source = (
        qei_source_lib.QeiSource() if qei_source is None else qei_source
    )

    self._jext = (
        external_current_source.ExternalCurrentSource()
        if jext is None
        else jext
    )
    gas_puff_source = (
        electron_density_sources.GasPuffSource()
        if gas_puff_source is None
        else gas_puff_source
    )
    nbi_particle_source = (
        electron_density_sources.NBIParticleSource()
        if nbi_particle_source is None
        else nbi_particle_source
    )
    pellet_source = (
        electron_density_sources.PelletSource()
        if pellet_source is None
        else pellet_source
    )
    generic_ion_el_heat_source = (
        generic_ion_el_heat_source_lib.GenericIonElectronHeatSource()
        if generic_ion_el_heat_source is None
        else generic_ion_el_heat_source
    )
    fusion_heat_source = (
        fusion_heat_source_lib.FusionHeatSource()
        if fusion_heat_source is None
        else fusion_heat_source
    )
    ohmic_heat_source = (
        OhmicHeatSource(sources=self)
        if ohmic_heat_source is None
        else ohmic_heat_source
    )
    additional_sources = (
        [] if additional_sources is None else additional_sources
    )

    # All sources which are "standard" and can be accessed as
    # source_lib.Source objects when computing profiles.
    self._standard_sources: dict[str, source_lib.Source] = dict(
        gas_puff_source=gas_puff_source,
        nbi_particle_source=nbi_particle_source,
        pellet_source=pellet_source,
        generic_ion_el_heat_source=generic_ion_el_heat_source,
        fusion_heat_source=fusion_heat_source,
        ohmic_heat_source=ohmic_heat_source,
    )
    for additional_source in additional_sources:
      self._standard_sources[additional_source.name] = additional_source
    self._psi_sources: dict[str, source_lib.Source] = {}
    self._ne_sources: dict[str, source_lib.Source] = {}
    self._temp_ion_sources: dict[str, source_lib.Source] = {}
    self._temp_el_sources: dict[str, source_lib.Source] = {}

    for source_name, source in self._standard_sources.items():
      if (
          source_lib.AffectedMeshStateAttribute.PSI
          in source.affected_mesh_states
      ):
        self._psi_sources[source_name] = source
      if (
          source_lib.AffectedMeshStateAttribute.NE
          in source.affected_mesh_states
      ):
        self._ne_sources[source_name] = source
      if (
          source_lib.AffectedMeshStateAttribute.TEMP_ION
          in source.affected_mesh_states
      ):
        self._temp_ion_sources[source_name] = source
      if (
          source_lib.AffectedMeshStateAttribute.TEMP_EL
          in source.affected_mesh_states
      ):
        self._temp_el_sources[source_name] = source

    self._all_sources = self._standard_sources | {
        self._j_bootstrap.name: self._j_bootstrap,
        self._jext.name: self._jext,
        self._qei_source.name: self._qei_source,
    }

  # Some sources require direct access, so this class defines properties for
  # those sources.

  @property
  def j_bootstrap(self) -> bootstrap_current_source.BootstrapCurrentSource:
    return self._j_bootstrap

  @property
  def jext(self) -> external_current_source.ExternalCurrentSource:
    return self._jext

  @property
  def qei_source(self) -> qei_source_lib.QeiSource:
    return self._qei_source

  @property
  def psi_sources(self) -> dict[str, source_lib.Source]:
    return self._psi_sources

  @property
  def ne_sources(self) -> dict[str, source_lib.Source]:
    return self._ne_sources

  @property
  def temp_ion_sources(self) -> dict[str, source_lib.Source]:
    return self._temp_ion_sources

  @property
  def temp_el_sources(self) -> dict[str, source_lib.Source]:
    return self._temp_el_sources

  @property
  def standard_sources(self) -> dict[str, source_lib.Source]:
    """Returns all sources that are not used in special cases.

    Practically, this means this includes all sources other than j_bootstrap,
    jext, qei_source.
    """
    return self._standard_sources

  @property
  def all_sources(self) -> dict[str, source_lib.Source]:
    return self._all_sources
