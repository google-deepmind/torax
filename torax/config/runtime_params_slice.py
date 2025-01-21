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

"""Inputs to TORAX steppers and functions based on the input runtime parameters.

When running a TORAX simulation, the stepper is (by default) a JAX-compiled
function, meaning it has two types of arguments: "dynamic" and "static".

The "dynamic" arguments can change from call to call. These arguments must be
arrays, scalars, or standard (possibly nested) Python containers. See the JAX
docs for more info on allowed types. They cannot influence the logical branches
the SimulationStepFn may take (again, see the sharp bits in the JAX docs to
learn more about the how these "dynamic" args can be used within the function).

Note that the "dynamic" arguments are NOT necessarily time-dependent. They do
not need to vary from time step to time step (though they can). They can change
from time step to time step, or from simulation run to simulation run, without
triggering a recompile. Changing these params without needing to recompile the
stepper is the defining quality of the dynamic arguments.

The "static" arguments are compile-time constant. Any changes to them would
trigger a recompilation of the stepper. These arguments don't have the same
restrictions as the dynamic arguments both in terms of types and how they are
used.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses

import chex
from torax.config import numerics
from torax.config import plasma_composition
from torax.config import profile_conditions
from torax.config import runtime_params as general_runtime_params_lib
from torax.geometry import geometry
from torax.geometry import geometry_provider as geometry_provider_lib
from torax.pedestal_model import runtime_params as pedestal_model_params
from torax.sources import runtime_params as sources_params
from torax.stepper import runtime_params as stepper_params
from torax.transport_model import runtime_params as transport_model_params


# Many of the variables follow scientific or mathematical notation, so disable
# pylint complaints.
# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class DynamicRuntimeParamsSlice:
  """Input params that are ok to use as inputs to a JAX-compiled function.

  This PyTree of params is input to the sim.SimulationStepFn, which updates
  the joint state and evolves the mesh state. This config includes various
  "dynamic" parameters which can change from step to step, or from
  simulation run to simulation run, without requiring components in the
  SimulationStepFn to recompile.

  Note that "dynamic" does NOT mean time dependent necessarily (though these
  params can be time dependent). Here "dynamic" means these params can change
  without trigerring or requiring a recompile.

  While the parameters are not necessarily time-dependent, that is how the class
  gets its name: a config "slice" refers to a subset of the overall TORAX config
  at a specific time t.

  This class contains "slices" of various RuntimeParams attributes defined
  throughout TORAX:

  - from the "general" runtime params
  - from the transport model's runtime params
  - from the stepper's runtime params
  - from each of the sources' runtime params

  This class packages all these together for convenience, as it simplifies many
  of the internal APIs within TORAX.
  """

  transport: transport_model_params.DynamicRuntimeParams
  stepper: stepper_params.DynamicRuntimeParams
  plasma_composition: plasma_composition.DynamicPlasmaComposition
  profile_conditions: profile_conditions.DynamicProfileConditions
  numerics: numerics.DynamicNumerics
  sources: Mapping[str, sources_params.DynamicRuntimeParams]
  pedestal: pedestal_model_params.DynamicRuntimeParams


@chex.dataclass(frozen=True)
class StaticRuntimeParamsSlice:
  """Static arguments to SimulationStepFn which cannot be changed.

  If any changes are made to these arguments, then components in
  SimulationStepFn must be recompiled.

  NOTE: These are not the only parameters which can trigger recompilations! For
  instance, if the geometry changes its shape (i.e. nr or hires_fac change),
  that can also trigger a recompile. This is just to note that this list is not
  an exhaustive list of what can cause recompilations.

  TODO(b/335596447): Add function to help users detect whether their
  change in config will trigger a recompile.
  """

  stepper: stepper_params.StaticRuntimeParams
  # Mapping of source name to source-specific static runtime params.
  sources: Mapping[str, sources_params.StaticRuntimeParams]
  # Torax mesh used to construct the geometry.
  torax_mesh: geometry.Grid1D
  # Solve the ion heat equation (ion temperature evolves over time)
  ion_heat_eq: bool
  # Solve the electron heat equation (electron temperature evolves over time)
  el_heat_eq: bool
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  current_eq: bool
  # Solve the density equation (n evolves over time)
  dens_eq: bool
  # Ion symbols for main ion and impurity (which each could be mixtures of ions)
  # These are static to simplify source functions for fusion power and radiation
  # which are species-dependent.
  # TODO(b/390279669): add guards against changing ion information
  # inconsistently between the static and dynamic runtime params slices.
  main_ion_names: tuple[str, ...]
  impurity_names: tuple[str, ...]

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool

  def __hash__(self):
    return hash((
        self.stepper,
        tuple(sorted(self.sources.items())),  # Hashable version of sources
        hash(self.torax_mesh),  # Grid1D has a hash method defined.
        self.ion_heat_eq,
        self.el_heat_eq,
        self.current_eq,
        self.dens_eq,
        self.main_ion_names,
        self.impurity_names,
        self.adaptive_dt,
    ))

  def validate_new(self, new_params: StaticRuntimeParamsSlice):
    """Validates that the new static runtime params slice is compatible."""
    if set(new_params.sources) != set(self.sources):
      raise ValueError('New static runtime params slice has different sources.')


def _build_dynamic_sources(
    sources: dict[str, sources_params.RuntimeParamsProvider],
    t: chex.Numeric,
) -> dict[str, sources_params.DynamicRuntimeParams]:
  """Builds a dict of DynamicSourceConfigSlice based on the input config."""
  return {
      source_name: input_source_config.build_dynamic_params(
          t,
      )
      for source_name, input_source_config in sources.items()
  }


def build_static_runtime_params_slice(
    *,
    runtime_params: general_runtime_params_lib.GeneralRuntimeParams,
    source_runtime_params: dict[str, sources_params.RuntimeParams],
    torax_mesh: geometry.Grid1D,
    stepper: stepper_params.RuntimeParams | None = None,
) -> StaticRuntimeParamsSlice:
  """Builds a StaticRuntimeParamsSlice.

  Args:
    runtime_params: General runtime params from which static params are taken,
      which are the choices on equations being solved, and adaptive dt.
    source_runtime_params: data from which the source related static variables
      are taken, which are the explicit/implicit toggle and calculation mode for
      each source.
    torax_mesh: The torax mesh, e.g. the grid used to construct the geometry.
      This is static for the entire simulation and any modification implies
      changed array sizes, and hence would require a recompilation. Useful to
      have a static (concrete) mesh for various internal calculations.
    stepper: stepper runtime params from which stepper static variables are
      extracted, related to solver methods. If None, defaults to the default
      stepper runtime params.

  Returns:
    A StaticRuntimeParamsSlice.
  """
  stepper = stepper or stepper_params.RuntimeParams()
  return StaticRuntimeParamsSlice(
      sources={
          source_name: specific_source_runtime_params.build_static_params()
          for source_name, specific_source_runtime_params in source_runtime_params.items()
      },
      torax_mesh=torax_mesh,
      stepper=stepper.build_static_params(),
      ion_heat_eq=runtime_params.numerics.ion_heat_eq,
      el_heat_eq=runtime_params.numerics.el_heat_eq,
      current_eq=runtime_params.numerics.current_eq,
      dens_eq=runtime_params.numerics.dens_eq,
      main_ion_names=runtime_params.plasma_composition.get_main_ion_names(),
      impurity_names=runtime_params.plasma_composition.get_impurity_names(),
      adaptive_dt=runtime_params.numerics.adaptive_dt,
  )


class DynamicRuntimeParamsSliceProvider:
  """Provides a DynamicRuntimeParamsSlice to use during time t of the sim.

  The DynamicRuntimeParamsSlice may change from time step to time step, so this
  class interpolates any time-dependent params in the input config to the values
  they should be at time t.

  NOTE: In order to maintain consistency between the DynamicRuntimeParamsSlice
  and the geometry,
  `sim.get_consistent_dynamic_runtime_params_slice_and_geometry`
  should be used to get a slice of the DynamicRuntimeParamsSlice and a
  corresponding geometry.

  See `run_simulation()` for how this callable is used.

  After this object has been constructed changes any runtime params may not
  be picked up if they are updated and it is safest to construct a new provider
  object (if for example updating the simulation).

  In more detail if you are updating any interpolated variable constructors
  (e.g. `runtime_params.profile_conditions.Ti_bound_right`) you will need to
  construct a new provider object. If you are only updating static variables
  (e.g. `runtime_params.profile_conditions.normalize_to_nbar`) then you can
  update the runtime params object in place and the changes will be picked up in
  the provider you have.

  For example to update the Ti_bound_right interpolated var constructor:
  ```
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
      runtime_params=runtime_params,
      torax_mesh=torax_mesh,
  )
  runtime_params.profile_conditions.Ti_bound_right = new_value
  provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
      runtime_params=runtime_params,
      torax_mesh=torax_mesh,
  )
  ```
  """

  def __init__(
      self,
      runtime_params: general_runtime_params_lib.GeneralRuntimeParams,
      pedestal: pedestal_model_params.RuntimeParams | None = None,
      transport: transport_model_params.RuntimeParams | None = None,
      sources: dict[str, sources_params.RuntimeParams] | None = None,
      stepper: stepper_params.RuntimeParams | None = None,
      torax_mesh: geometry.Grid1D | None = None,
  ):
    """Constructs a DynamicRuntimeParamsSliceProvider.

    Args:
      runtime_params: The general runtime params to use.
      pedestal: The pedestal model runtime params to use. If None, defaults to
        the default pedestal model runtime params.
      transport: The transport model runtime params to use. If None, defaults to
        the default transport model runtime params.
      sources: A dict of source name to source runtime params to use. If None,
        defaults to an empty dict (i.e. no sources).
      stepper: The stepper runtime params to use. If None, defaults to the
        default stepper runtime params.
      torax_mesh: The torax mesh to use. If the slice provider doesn't need to
        construct any rho interpolated values, this can be None, else an error
        will be raised within the constructor of the interpolated variable.
    """
    transport = transport or transport_model_params.RuntimeParams()
    sources = sources or {}
    stepper = stepper or stepper_params.RuntimeParams()
    pedestal = pedestal or pedestal_model_params.RuntimeParams()
    self._torax_mesh = torax_mesh
    self._sources = sources
    self._runtime_params = runtime_params
    self._transport_runtime_params = transport
    self._stepper = stepper
    self._pedestal_runtime_params = pedestal
    self._construct_providers()

  @property
  def sources(self) -> dict[str, sources_params.RuntimeParams]:
    return self._sources

  def validate_new(
      self,
      new_provider: DynamicRuntimeParamsSliceProvider,
  ):
    """Validates that the new provider is compatible."""
    if set(new_provider.sources) != set(self.sources):
      raise ValueError(
          'New dynamic runtime params slice provider has different sources.'
      )

  @property
  def runtime_params_provider(
      self,
  ) -> general_runtime_params_lib.GeneralRuntimeParamsProvider:
    return self._runtime_params_provider

  def _construct_providers(self):
    """Construct the providers that will give us the dynamic params."""
    self._runtime_params_provider = self._runtime_params.make_provider(
        self._torax_mesh
    )
    self._transport_runtime_params_provider = (
        self._transport_runtime_params.make_provider(self._torax_mesh)
    )
    self._sources_providers = {
        key: source.make_provider(self._torax_mesh)
        for key, source in self._sources.items()
    }
    self._pedestal_runtime_params_provider = (
        self._pedestal_runtime_params.make_provider(self._torax_mesh)
    )

  def __call__(
      self,
      t: chex.Numeric,
  ) -> DynamicRuntimeParamsSlice:
    """Returns a DynamicRuntimeParamsSlice to use during time t of the sim."""
    dynamic_general_runtime_params = (
        self._runtime_params_provider.build_dynamic_params(t)
    )
    return DynamicRuntimeParamsSlice(
        transport=self._transport_runtime_params_provider.build_dynamic_params(
            t
        ),
        stepper=self._stepper.build_dynamic_params(t),
        sources=_build_dynamic_sources(
            self._sources_providers,
            t,
        ),
        plasma_composition=dynamic_general_runtime_params.plasma_composition,
        profile_conditions=dynamic_general_runtime_params.profile_conditions,
        numerics=dynamic_general_runtime_params.numerics,
        pedestal=self._pedestal_runtime_params_provider.build_dynamic_params(t),
    )


def get_consistent_dynamic_runtime_params_slice_and_geometry(
    *,
    t: chex.Numeric,
    dynamic_runtime_params_slice_provider: DynamicRuntimeParamsSliceProvider,
    geometry_provider: geometry_provider_lib.GeometryProvider,
) -> tuple[DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Returns the dynamic runtime params and geometry for a given time."""
  geo = geometry_provider(t)
  dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(
      t=t,
  )
  dynamic_runtime_params_slice, geo = make_ip_consistent(
      dynamic_runtime_params_slice, geo
  )
  return dynamic_runtime_params_slice, geo


def make_ip_consistent(
    dynamic_runtime_params_slice: DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> tuple[DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Fixes Ip to be the same across dynamic_runtime_params_slice and geo."""
  if isinstance(geo, geometry.StandardGeometry):
    if geo.Ip_from_parameters:
      # If Ip is from parameters, renormalise psi etc to match the Ip in the
      # parameters.
      # pylint: disable=invalid-name
      param_Ip = dynamic_runtime_params_slice.profile_conditions.Ip_tot
      Ip_scale_factor = param_Ip * 1e6 / geo.Ip_profile_face[-1]
      geo = dataclasses.replace(
          geo,
          Ip_profile_face=geo.Ip_profile_face * Ip_scale_factor,
          psi_from_Ip=geo.psi_from_Ip * Ip_scale_factor,
          jtot=geo.jtot * Ip_scale_factor,
          jtot_face=geo.jtot_face * Ip_scale_factor,
      )
      # pylint: enable=invalid-name
    else:
      # If Ip is from the geometry, update the parameters to match.
      dynamic_runtime_params_slice = dataclasses.replace(
          dynamic_runtime_params_slice,
          profile_conditions=dataclasses.replace(
              dynamic_runtime_params_slice.profile_conditions,
              Ip_tot=geo.Ip_profile_face[-1] / 1e6,
          ),
      )
  return dynamic_runtime_params_slice, geo
