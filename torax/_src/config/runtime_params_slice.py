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
from collections.abc import Mapping
import dataclasses

import chex
from torax._src.config import numerics
from torax._src.config import plasma_composition
from torax._src.config import profile_conditions
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.neoclassical import runtime_params as neoclassical_params
from torax._src.pedestal_model import runtime_params as pedestal_model_params
from torax._src.sources import runtime_params as sources_params
from torax._src.stepper import runtime_params as solver_params
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import runtime_params as transport_model_params
import typing_extensions

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
  without triggering or requiring a recompile.

  While the parameters are not necessarily time-dependent, that is how the class
  gets its name: a config "slice" refers to a subset of the overall TORAX config
  at a specific time t.

  This class contains "slices" of various RuntimeParams attributes defined
  throughout TORAX.
  This class packages all these together for convenience, as it simplifies many
  of the internal APIs within TORAX.
  """

  mhd: mhd_runtime_params.DynamicMHDParams
  neoclassical: neoclassical_params.DynamicRuntimeParams
  numerics: numerics.DynamicNumerics
  pedestal: pedestal_model_params.DynamicRuntimeParams
  plasma_composition: plasma_composition.DynamicPlasmaComposition
  profile_conditions: profile_conditions.DynamicProfileConditions
  solver: solver_params.DynamicRuntimeParams
  sources: Mapping[str, sources_params.DynamicRuntimeParams]
  transport: transport_model_params.DynamicRuntimeParams


@chex.dataclass(frozen=True)
class StaticRuntimeParamsSlice:
  """Static arguments to SimulationStepFn which cannot be changed.

  If any changes are made to these arguments, then components in
  SimulationStepFn must be recompiled.

  NOTE: These are not the only parameters which can trigger recompilations! For
  instance, if the geometry changes its shape (i.e. nr or hires_factor change),
  that can also trigger a recompile. This is just to note that this list is not
  an exhaustive list of what can cause recompilations.

  TODO(b/335596447): Add function to help users detect whether their
  change in config will trigger a recompile.
  """
  # Solver-specific static runtime params.
  solver: solver_params.StaticRuntimeParams
  # Mapping of source name to source-specific static runtime params.
  sources: Mapping[str, sources_params.StaticRuntimeParams]
  # Torax mesh used to construct the geometry.
  torax_mesh: torax_pydantic.Grid1D
  # Solve the ion heat equation (ion temperature evolves over time)
  evolve_ion_heat: bool
  # Solve the electron heat equation (electron temperature evolves over time)
  evolve_electron_heat: bool
  # Solve the current equation (psi evolves over time driven by the solver;
  # q and s evolve over time as a function of psi)
  evolve_current: bool
  # Solve the density equation (n evolves over time)
  evolve_density: bool
  # Ion symbols for main ion and impurity (which each could be mixtures of ions)
  # These are static to simplify source functions for fusion power and radiation
  # which are species-dependent.
  # TODO(b/390279669): add guards against changing ion information
  # inconsistently between the static and dynamic runtime params slices.
  main_ion_names: tuple[str, ...]
  impurity_names: tuple[str, ...]
  profile_conditions: profile_conditions.StaticRuntimeParams
  # Iterative reduction of dt if nonlinear step does not converge,
  # If nonlinear step does not converge, then the step is redone
  # iteratively at successively lower dt until convergence is reached
  adaptive_dt: bool

  def __hash__(self):
    return hash((
        self.solver,
        tuple(sorted(self.sources.items())),  # Hashable version of sources
        hash(self.torax_mesh),  # Grid1D has a hash method defined.
        self.evolve_ion_heat,
        self.evolve_electron_heat,
        self.evolve_current,
        self.evolve_density,
        self.main_ion_names,
        self.impurity_names,
        self.adaptive_dt,
        self.profile_conditions,
    ))

  def validate_new(self, new_params: typing_extensions.Self):
    """Validates that the new static runtime params slice is compatible."""
    if set(new_params.sources) != set(self.sources):
      raise ValueError('New static runtime params slice has different sources.')


def make_ip_consistent(
    dynamic_runtime_params_slice: DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
) -> tuple[DynamicRuntimeParamsSlice, geometry.Geometry]:
  """Fixes Ip to be the same across dynamic_runtime_params_slice and geo."""
  if isinstance(geo, standard_geometry.StandardGeometry):
    if geo.Ip_from_parameters:
      # If Ip is from parameters, renormalise psi etc to match the Ip in the
      # parameters.
      # pylint: disable=invalid-name
      param_Ip = dynamic_runtime_params_slice.profile_conditions.Ip
      Ip_scale_factor = param_Ip / geo.Ip_profile_face[-1]
      geo = dataclasses.replace(
          geo,
          Ip_profile_face=geo.Ip_profile_face * Ip_scale_factor,
          psi_from_Ip=geo.psi_from_Ip * Ip_scale_factor,
          psi_from_Ip_face=geo.psi_from_Ip_face * Ip_scale_factor,
          j_total=geo.j_total * Ip_scale_factor,
          j_total_face=geo.j_total_face * Ip_scale_factor,
      )
      # pylint: enable=invalid-name
    else:
      # If Ip is from the geometry, update the parameters to match.
      dynamic_runtime_params_slice = dataclasses.replace(
          dynamic_runtime_params_slice,
          profile_conditions=dataclasses.replace(
              dynamic_runtime_params_slice.profile_conditions,
              Ip=geo.Ip_profile_face[-1],
          ),
      )
  return dynamic_runtime_params_slice, geo
