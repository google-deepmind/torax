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

"""Functions for building source profiles in TORAX."""

from __future__ import annotations

import dataclasses
import functools

import jax.numpy as jnp
from torax import constants
from torax import geometry
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import diffusion_terms
from torax.sources import bootstrap_current_source
from torax.sources import external_current_source
from torax.sources import qei_source as qei_source_lib
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
    ],
)
def build_source_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool,
) -> source_profiles.SourceProfiles:
  """Builds explicit or implicit source profiles.

  Args:
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Functions computing profiles for all TORAX sources/sinks.
    explicit: If True, this function should return profiles for all explicit
      sources. All implicit sources should be set to 0. And same vice versa.

  Returns:
    SourceProfiles for either explicit or implicit sources (and all others set
    to zero).
  """
  # Bootstrap current is a special-case source with multiple outputs, so handle
  # it here.
  dynamic_bootstrap_runtime_params = dynamic_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      explicit=explicit,
  )
  other_profiles = {}
  other_profiles.update(
      _build_psi_profiles(
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
          explicit,
      )
  )
  other_profiles.update(
      _build_ne_profiles(
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
          explicit,
      )
  )
  other_profiles.update(
      _build_temp_ion_el_profiles(
          dynamic_runtime_params_slice,
          geo,
          core_profiles,
          source_models,
          explicit,
      )
  )
  return source_profiles.SourceProfiles(
      profiles=other_profiles,
      j_bootstrap=bootstrap_profiles,
      # Qei is computed within calc_coeffs and will replace this value. This is
      # here as a placeholder with correct shapes.
      qei=source_profiles.QeiInfo.zeros(geo),
  )


def _build_bootstrap_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    j_bootstrap_source: bootstrap_current_source.BootstrapCurrentSource,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> source_profiles.BootstrapCurrentProfile:
  """Computes the bootstrap current profile.

  Args:
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    dynamic_source_runtime_params: Input runtime parameters for this time step,
      specific to the bootstrap current source.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
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
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_source_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
  )
  sigma = jax_utils.select(
      jnp.logical_or(
          explicit == dynamic_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.sigma,
      jnp.zeros_like(bootstrap_profile.sigma),
  )
  j_bootstrap = jax_utils.select(
      jnp.logical_or(
          explicit == dynamic_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap,
      jnp.zeros_like(bootstrap_profile.j_bootstrap),
  )
  j_bootstrap_face = jax_utils.select(
      jnp.logical_or(
          explicit == dynamic_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.j_bootstrap_face,
      jnp.zeros_like(bootstrap_profile.j_bootstrap_face),
  )
  I_bootstrap = jax_utils.select(  # pylint: disable=invalid-name
      jnp.logical_or(
          explicit == dynamic_source_runtime_params.is_explicit,
          calculate_anyway,
      ),
      bootstrap_profile.I_bootstrap,
      jnp.zeros_like(bootstrap_profile.I_bootstrap),
  )
  return source_profiles.BootstrapCurrentProfile(
      sigma=sigma,
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
      I_bootstrap=I_bootstrap,
  )


def _build_psi_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool = True,
    calculate_anyway: bool = False,
) -> dict[str, jnp.ndarray]:
  """Computes psi sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Collection of all TORAX sources.
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
  # jext is precomputed in the core profiles.
  dynamic_jext_runtime_params = dynamic_runtime_params_slice.sources[
      source_models.jext_name
  ]
  psi_profiles[source_models.jext_name] = jax_utils.select(
      jnp.logical_or(
          explicit == dynamic_jext_runtime_params.is_explicit,
          calculate_anyway,
      ),
      core_profiles.currents.jext,
      jnp.zeros_like(geo.r),
  )
  # Iterate through the rest of the sources and compute profiles for the ones
  # which relate to psi. jext is not part of the "standard sources."
  for source_name, source in source_models.psi_sources.items():
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        source_name
    ]
    psi_profiles[source_name] = jax_utils.select(
        jnp.logical_or(
            explicit == dynamic_source_runtime_params.is_explicit,
            calculate_anyway,
        ),
        source.get_value(
            dynamic_runtime_params_slice,
            dynamic_source_runtime_params,
            geo,
            core_profiles,
        ),
        jnp.zeros_like(geo.r),
    )
  return psi_profiles


def _build_ne_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool,
) -> dict[str, jnp.ndarray]:
  """Computes ne sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Collection of all TORAX sources.
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
  for source_name, source in source_models.ne_sources.items():
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        source_name
    ]
    ne_profiles[source_name] = jax_utils.select(
        explicit == dynamic_source_runtime_params.is_explicit,
        source.get_value(
            dynamic_runtime_params_slice,
            dynamic_source_runtime_params,
            geo,
            core_profiles,
        ),
        jnp.zeros_like(geo.r),
    )
  return ne_profiles


def _build_temp_ion_el_profiles(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
    explicit: bool,
) -> dict[str, jnp.ndarray]:
  """Computes ion and el sources and builds a kwargs dict for SourceProfiles.

  Args:
    dynamic_runtime_params_slice: Input config for this time step. Can change
      from time step to time step.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles, either at the start of the time step
      (if explicit) or the live profiles being evolved during the time step (if
      implicit).
    source_models: Collection of all TORAX sources.
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
  temp_ion_el_sources = (
      source_models.temp_ion_sources | source_models.temp_el_sources
  )
  for source_name, source in temp_ion_el_sources.items():
    zeros = jnp.zeros(source.output_shape_getter(geo))
    dynamic_source_runtime_params = dynamic_runtime_params_slice.sources[
        source_name
    ]
    ion_el_profiles[source_name] = jax_utils.select(
        explicit == dynamic_source_runtime_params.is_explicit,
        source.get_value(
            dynamic_runtime_params_slice,
            dynamic_source_runtime_params,
            geo,
            core_profiles,
        ),
        zeros,
    )
  return ion_el_profiles


def sum_sources_psi(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  """Computes psi source values for sim.calc_coeffs."""
  total = (
      source_profile.j_bootstrap.j_bootstrap
      + source_profile.profiles[source_models.jext_name]
  )
  for source_name, source in source_models.psi_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.PSI.value,
        geo=geo,
    )
  denom = 2 * jnp.pi * geo.Rmaj * geo.J**2
  scale_source = lambda src: -geo.vpr * src * constants.CONSTANTS.mu0 / denom
  return scale_source(total)


def sum_sources_ne(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  """Computes ne source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in source_models.ne_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=source_lib.AffectedCoreProfile.NE.value,
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_ion(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  """Computes temp_ion source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in source_models.temp_ion_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_ION.value),
        geo=geo,
    )
  return total * geo.vpr


def sum_sources_temp_el(
    geo: geometry.Geometry,
    source_profile: source_profiles.SourceProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  """Computes temp_el source values for sim.calc_coeffs."""
  total = jnp.zeros_like(geo.r)
  for source_name, source in source_models.temp_el_sources.items():
    total += source.get_source_profile_for_affected_core_profile(
        profile=source_profile.profiles[source_name],
        affected_core_profile=(source_lib.AffectedCoreProfile.TEMP_EL.value),
        geo=geo,
    )
  return total * geo.vpr


def calc_and_sum_sources_psi(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes sum of psi sources for psi_dot calculation."""

  # TODO(b/335597108): Revisit how to calculate this once we enable more
  # expensive source functions that might not jittable (like file-based or
  # RPC-based sources).
  psi_profiles = _build_psi_profiles(
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
      calculate_anyway=True,
  )
  total = 0
  for key in psi_profiles:
    total += psi_profiles[key]
  dynamic_bootstrap_runtime_params = dynamic_runtime_params_slice.sources[
      source_models.j_bootstrap_name
  ]
  j_bootstrap_profiles = _build_bootstrap_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      dynamic_source_runtime_params=dynamic_bootstrap_runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      j_bootstrap_source=source_models.j_bootstrap,
      calculate_anyway=True,
  )
  total += j_bootstrap_profiles.j_bootstrap
  denom = 2 * jnp.pi * geo.Rmaj * geo.J**2
  scale_source = lambda src: -geo.vpr * src * constants.CONSTANTS.mu0 / denom

  return scale_source(total), j_bootstrap_profiles.sigma


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
    ],
)
def calc_psidot(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  r"""Calculates psidot (loop voltage). Used for the Ohmic electron heat source.

  psidot is an interesting TORAX output, and is thus also saved in
  core_profiles.

  psidot = \partial psi / \partial t, and is derived from the same components
  that form the psi block in the coupled PDE equations. This, a similar
  (but abridged) formulation as in sim.calc_coeffs and fvm._calc_c is used here

  Args:
    dynamic_runtime_params_slice: Simulation configuration at this timestep
    geo: Torus geometry
    core_profiles: Core plasma profiles.
    source_models: All TORAX source/sinks.

  Returns:
    psidot: on cell grid
  """
  consts = constants.CONSTANTS

  psi_sources, sigma = calc_and_sum_sources_psi(
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )
  toc_psi = (
      1.0
      / dynamic_runtime_params_slice.numerics.resistivity_mult
      * geo.r
      * sigma
      * consts.mu0
      / geo.J**2
      / geo.Rmaj
  )
  d_face_psi = geo.G2_face / geo.J_face / geo.rmax**2

  c_mat, c = diffusion_terms.make_diffusion_terms(d_face_psi, core_profiles.psi)
  c += psi_sources

  psidot = (jnp.dot(c_mat, core_profiles.psi.value) + c) / toc_psi

  return psidot


#  OhmicHeatSource is a special case and defined here to avoid circular
#  dependencies, since it depends on the psi sources
def _ohmic_heat_model(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: SourceModels,
) -> jnp.ndarray:
  """Returns the Ohmic source for electron heat equation."""
  jtot, _ = physics.calc_jtot_from_psi(
      geo,
      core_profiles.psi,
  )

  psidot = calc_psidot(
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )

  pohm = jtot * psidot / (2 * jnp.pi * geo.Rmaj)
  return pohm


@dataclasses.dataclass(kw_only=True)
class OhmicHeatSource(source_lib.SingleProfileSource):
  """Ohmic heat source for electron heat equation.

  Pohm = jtor * psidot /(2*pi*Rmaj), related to electric power formula P = IV.

  Because this source requires access to the rest of the Sources, it must be
  added to the SourceModels object after creation:

  ```python
  source_models = SourceModels(sources={...})
  # Now add the ohmic heat source and turn it on.
  source_models.add_source(
      source_name='ohmic_heat_source',
      source=OhmicHeatSource(
          source_models=source_models,
          runtime_params=runtime_params.RuntimeParams(
              mode=runtime_params.Mode.MODEL_BASED,  # turns the source on.
          ),
      ),
  )
  ```
  """

  # Users must pass in a pointer to the complete set of sources to this object.
  source_models: SourceModels

  supported_modes: tuple[runtime_params_lib.Mode, ...] = (
      runtime_params_lib.Mode.ZERO,
      runtime_params_lib.Mode.MODEL_BASED,
  )

  # Freeze these params and do not include them in the __init__.
  affected_core_profiles: tuple[source_lib.AffectedCoreProfile, ...] = (
      dataclasses.field(
          init=False,
          default=(source_lib.AffectedCoreProfile.TEMP_EL,),
      )
  )
  model_func: source_lib.SourceProfileFunction | None = dataclasses.field(
      init=False,
      default_factory=lambda: None,  # ignored.
  )

  def __post_init__(self):
    # Ignore the model provided above and set it to the function here.
    def _model_func(
        dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
        dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
        geo: geometry.Geometry,
        core_profiles: state.CoreProfiles,
    ) -> jnp.ndarray:
      del dynamic_source_runtime_params
      return _ohmic_heat_model(
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          geo=geo,
          core_profiles=core_profiles,
          source_models=self.source_models,
      )

    self.model_func = _model_func


class SourceModels:
  """Source/sink models for the different equations being evolved in Torax.

  Each source/sink (all called sources as the difference is only a sign change)
  can be explicit or implicit and signal to our solver on how to handle these
  terms. Their values are provided via model, file, prescribed function, etc.
  The specific approach used depends on how the source is initialized and what
  runtime configuration inputs are provided.

  NOTE for RAPTOR users: Sources in TORAX are similar to actuators in RAPTOR and
  provide a hook to provide custom profiles.

  You can both override the default set of sources in TORAX as well as define
  new custom sources inline when constructing this object. The example below
  shows how to define a new custom electron-density source.

  ```python
  # Define an electron-density source with a time-dependent Gaussian profile.
  my_custom_source = source.SingleProfileSource(
      supported_modes=(
          runtime_params_lib.Mode.ZERO,
          runtime_params_lib.Mode.FORMULA_BASED,
      ),
      affected_core_profiles=source.AffectedCoreProfile.NE,
      formula=formulas.Gaussian(),
      # Define (possibly) time-dependent parameters to feed to the formula.
      runtime_params=runtime_params_lib.RuntimeParams(
          formula=formula_config.Gaussian(
              total={0.0: 1.0, 5.0: 2.0, 10.0: 1.0},  # time-dependent.
              c1=2.0,
              c2=3.0,
          ),
      ),
  )
  # Define the collection of sources here, which in this example only includes
  # one source.
  all_torax_sources = source_models_lib.SourceModels(
      sources={'my_custom_source': my_custom_source}
  )
  ```

  See runtime_params.py for more details on how to configure all the source/sink
  terms.
  """

  def __init__(
      self,
      sources: dict[str, source_lib.Source] | None = None,
  ):
    """Constructs a collection of sources.

    This class defines which sources are available in a TORAX simulation run.
    Users can configure whether each source is actually on and what kind of
    profile it produces by changing its runtime configuration (see
    runtime_params_lib.py).

    Args:
      sources: Mapping of source model names to the Source objects. The names
        (i.e. the keys of this dictionary) also define the keys in the output
        SourceProfiles which are computed from this SourceModels object. NOTE -
        Some sources are "special-case": bootstrap current, external current,
        and Qei. SourceModels will always instantiate default objects for these
        types of sources unless they are provided by this `sources` argument.
        Also, their default names are reserved, meaning the input dictionary
        `sources` should not have the keys 'j_bootstrap', 'jext', or
        'qei_source' unless those sources are one of these "special-case"
        sources.

    Raises:
      ValueError if there is a naming collision with the reserved names as
      described above.
    """
    sources = sources or {}
    # Some sources are accessed for specific use cases, so we extract those
    # ones and expose them directly.
    self._j_bootstrap = None
    self._j_bootstrap_name = 'j_bootstrap'  # default, can be overridden below.
    self._jext = None
    self._jext_name = 'jext'  # default, can be overridden below.
    self._qei_source = None
    self._qei_source_name = 'qei_source'  # default, can be overridden below.
    # The rest of the sources are "standard".
    self._standard_sources = {}

    # Divide up the sources based on which core profiles they affect.
    self._psi_sources: dict[str, source_lib.Source] = {}
    self._ne_sources: dict[str, source_lib.Source] = {}
    self._temp_ion_sources: dict[str, source_lib.Source] = {}
    self._temp_el_sources: dict[str, source_lib.Source] = {}

    for source_name, source in sources.items():
      if isinstance(source, bootstrap_current_source.BootstrapCurrentSource):
        self._j_bootstrap_name = source_name
        self._j_bootstrap = source
      elif isinstance(source, external_current_source.ExternalCurrentSource):
        self._jext_name = source_name
        self._jext = source
      elif isinstance(source, qei_source_lib.QeiSource):
        self._qei_source_name = source_name
        self._qei_source = source
      else:
        self.add_source(source_name, source)

    # Make sure defaults are set.
    if self._j_bootstrap is None:
      self._j_bootstrap = bootstrap_current_source.BootstrapCurrentSource()
    if self._jext is None:
      self._jext = external_current_source.ExternalCurrentSource()
    if self._qei_source is None:
      self._qei_source = qei_source_lib.QeiSource()

  def add_source(
      self,
      source_name: str,
      source: source_lib.Source,
  ) -> None:
    """Adds a source to the collection of sources.

    Do NOT directly add new sources to `SourceModels.standard_sources`. Users
    should call this function instead. Cannot add additional bootstrap current,
    external current, or Qei sources - those must be defined in the __init__.

    Args:
      source_name: Name of the new source being added. This will be the key
        under which the source's output profile will be found in the output
        SourceProfiles object.
      source: The new standard source being added.

    Raises:
      ValueError if a "special-case" source is provided.
    """
    if (
        isinstance(source, bootstrap_current_source.BootstrapCurrentSource)
        or isinstance(source, external_current_source.ExternalCurrentSource)
        or isinstance(source, qei_source_lib.QeiSource)
    ):
      raise ValueError(
          'Cannot add a source with the following types: '
          'bootstrap_current_source.BootstrapCurrentSource,'
          ' external_current_source.ExternalCurrentSource, or'
          ' qei_source_lib.QeiSource.'
      )
    reserved_names = [
        self._j_bootstrap_name,
        self._jext_name,
        self._qei_source_name,
    ]
    if source_name in reserved_names:
      raise ValueError(
          f'Cannot add a source with one of these names: {reserved_names}.'
      )
    self._standard_sources[source_name] = source
    if source_lib.AffectedCoreProfile.PSI in source.affected_core_profiles:
      self._psi_sources[source_name] = source
    if source_lib.AffectedCoreProfile.NE in source.affected_core_profiles:
      self._ne_sources[source_name] = source
    if source_lib.AffectedCoreProfile.TEMP_ION in source.affected_core_profiles:
      self._temp_ion_sources[source_name] = source
    if source_lib.AffectedCoreProfile.TEMP_EL in source.affected_core_profiles:
      self._temp_el_sources[source_name] = source

  # Some sources require direct access, so this class defines properties for
  # those sources.

  @property
  def j_bootstrap(self) -> bootstrap_current_source.BootstrapCurrentSource:
    assert self._j_bootstrap is not None
    return self._j_bootstrap

  @property
  def j_bootstrap_name(self) -> str:
    return self._j_bootstrap_name

  @property
  def jext(self) -> external_current_source.ExternalCurrentSource:
    assert self._jext is not None
    return self._jext

  @property
  def jext_name(self) -> str:
    return self._jext_name

  @property
  def qei_source(self) -> qei_source_lib.QeiSource:
    assert self._qei_source is not None
    return self._qei_source

  @property
  def qei_source_name(self) -> str:
    return self._qei_source_name

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
  def ion_el_sources(self) -> dict[str, source_lib.Source]:
    """Returns all source models which output both ion and el temp profiles."""
    return {
        name: source
        for name, source in self._standard_sources.items()
        if source.affected_core_profiles
        == (
            source_lib.AffectedCoreProfile.TEMP_ION,
            source_lib.AffectedCoreProfile.TEMP_EL,
        )
    }

  @property
  def standard_sources(self) -> dict[str, source_lib.Source]:
    """Returns all sources that are not used in special cases.

    Practically, this means this includes all sources other than j_bootstrap,
    jext, qei_source.
    """
    return self._standard_sources

  @property
  def sources(self) -> dict[str, source_lib.Source]:
    return self._standard_sources | {
        self._j_bootstrap_name: self.j_bootstrap,
        self._jext_name: self.jext,
        self._qei_source_name: self.qei_source,
    }

  @property
  def runtime_params(self) -> dict[str, runtime_params_lib.RuntimeParams]:
    """Returns all the runtime params for all sources."""
    return {
        source_name: source.runtime_params
        for source_name, source in self.sources.items()
    }


def build_all_zero_profiles(
    geo: geometry.Geometry,
    source_models: SourceModels,
) -> source_profiles.SourceProfiles:
  """Returns a SourceProfiles object with all zero profiles."""
  profiles = {
      source_name: jnp.zeros(source_model.output_shape_getter(geo))
      for source_name, source_model in source_models.standard_sources.items()
  }
  profiles[source_models.jext_name] = jnp.zeros_like(geo.r)
  return source_profiles.SourceProfiles(
      profiles=profiles,
      j_bootstrap=source_profiles.BootstrapCurrentProfile.zero_profile(geo),
      qei=source_profiles.QeiInfo.zeros(geo),
  )
