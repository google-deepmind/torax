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
from torax import geometry
from torax.config import numerics
from torax.config import plasma_composition
from torax.config import profile_conditions
from torax.config import runtime_params as general_runtime_params_lib
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
  # Solve the ion heat equation (ion temperature evolves over time)
  ion_heat_eq: bool
  # Solve the electron heat equation (electron temperature evolves over time)
  el_heat_eq: bool
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  current_eq: bool
  # Solve the density equation (n evolves over time)
  dens_eq: bool

  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool


# pylint: enable=invalid-name
def _build_dynamic_sources(
    sources: dict[str, sources_params.RuntimeParams],
    t: chex.Numeric,
    geo: geometry.Geometry,
) -> dict[str, sources_params.DynamicRuntimeParams]:
  """Builds a dict of DynamicSourceConfigSlice based on the input config."""
  return {
      source_name: input_source_config.build_dynamic_params(t, geo)
      for source_name, input_source_config in sources.items()
  }


def build_static_runtime_params_slice(
    runtime_params: general_runtime_params_lib.GeneralRuntimeParams,
    stepper: stepper_params.RuntimeParams | None = None,
) -> StaticRuntimeParamsSlice:
  """Builds a StaticRuntimeParamsSlice."""
  # t set to None because there shouldnt be time-dependent params in the static
  # config.
  stepper = stepper or stepper_params.RuntimeParams()
  return StaticRuntimeParamsSlice(
      stepper=stepper.build_static_params(),
      ion_heat_eq=runtime_params.numerics.ion_heat_eq,
      el_heat_eq=runtime_params.numerics.el_heat_eq,
      current_eq=runtime_params.numerics.current_eq,
      dens_eq=runtime_params.numerics.dens_eq,
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

  After this object has been constructed changes to any runtime params may not
  be picked up if they are updated and it is safest to construct a new provider
  object (if for example updating the simulation).
  """

  def __init__(
      self,
      runtime_params: general_runtime_params_lib.GeneralRuntimeParams,
      transport: transport_model_params.RuntimeParams | None = None,
      sources: dict[str, sources_params.RuntimeParams] | None = None,
      stepper: stepper_params.RuntimeParams | None = None,
      torax_mesh: geometry.Grid1D | None = None,
  ):
    """Constructs a DynamicRuntimeParamsSliceProvider.

    Args:
      runtime_params: The general runtime params to use.
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
    self._runtime_params = runtime_params.make_provider(torax_mesh)
    self._transport_runtime_params = transport
    self._sources = sources
    self._stepper = stepper

  def __call__(
      self,
      t: chex.Numeric,
      geo: geometry.Geometry,
  ) -> DynamicRuntimeParamsSlice:
    """Returns a DynamicRuntimeParamsSlice to use during time t of the sim."""
    # For each dataclass attribute under DynamicRuntimeParamsSlice, build those
    # objects explicitly, and then for all scalar attributes, fetch their values
    # directly from the input runtime params using config_args.get_init_kwargs.
    dynamic_general_runtime_params = self._runtime_params.build_dynamic_params(
        t
    )
    return DynamicRuntimeParamsSlice(
        transport=self._transport_runtime_params.build_dynamic_params(t),
        stepper=self._stepper.build_dynamic_params(t),
        sources=_build_dynamic_sources(self._sources, t, geo),
        plasma_composition=dynamic_general_runtime_params.dynamic_plasma_composition,
        profile_conditions=dynamic_general_runtime_params.dynamic_profile_conditions,
        numerics=dynamic_general_runtime_params.dynamic_numerics,
    )


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
      param_Ip = dynamic_runtime_params_slice.profile_conditions.Ip
      Ip_scale_factor = param_Ip * 1e6 / geo.Ip
      geo = dataclasses.replace(
          geo,
          Ip=geo.Ip * Ip_scale_factor,
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
              Ip=geo.Ip / 1e6,
          ),
      )
  return dynamic_runtime_params_slice, geo
