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

"""Inputs to TORAX solvers and functions based on the input runtime parameters.

When running a TORAX simulation, the solver is (by default) a JAX-compiled
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
solver is the defining quality of the dynamic arguments.

The "static" arguments are compile-time constant. Any changes to them would
trigger a recompilation of the solver. These arguments don't have the same
restrictions as the dynamic arguments both in terms of types and how they are
used.
"""
from collections.abc import Mapping
import dataclasses

import jax
from torax._src.config import numerics
from torax._src.core_profiles import profile_conditions
from torax._src.core_profiles.plasma_composition import plasma_composition
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.mhd import runtime_params as mhd_runtime_params
from torax._src.neoclassical import runtime_params as neoclassical_params
from torax._src.pedestal_model import runtime_params as pedestal_model_params
from torax._src.solver import runtime_params as solver_params
from torax._src.sources import runtime_params as sources_params
from torax._src.time_step_calculator import runtime_params as time_step_calculator_runtime_params
from torax._src.transport_model import runtime_params as transport_model_params

# Many of the variables follow scientific or mathematical notation, so disable
# pylint complaints.
# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """A slice of the parameters at a specific time t.

  This PyTree is a slice of the overall TORAX config at a specific time t
  excluding the geometry, grouping the parameters for ease of passing into
  various downstream functions. It includes both parameters which are
  time-dependent and parameters which are not.

  The parameters which are not time-dependent are marked as static in their
  definition, this means that they are marked as static when input to JAX,
  which means that they are compile-time constants. This means that they cannot
  be changed without recompilation.
  """

  mhd: mhd_runtime_params.RuntimeParams
  neoclassical: neoclassical_params.RuntimeParams
  numerics: numerics.RuntimeParams
  pedestal: pedestal_model_params.RuntimeParams
  plasma_composition: plasma_composition.RuntimeParams
  profile_conditions: profile_conditions.RuntimeParams
  solver: solver_params.RuntimeParams
  sources: Mapping[str, sources_params.RuntimeParams]
  transport: transport_model_params.RuntimeParams
  time_step_calculator: time_step_calculator_runtime_params.RuntimeParams


def make_ip_consistent(
    runtime_params: RuntimeParams,
    geo: geometry.Geometry,
) -> tuple[RuntimeParams, geometry.Geometry]:
  """Fixes Ip to be the same across runtime_params and geo."""
  if isinstance(geo, standard_geometry.StandardGeometry):
    if geo.Ip_from_parameters:
      # If Ip is from parameters, renormalise psi etc to match the Ip in the
      # parameters.
      param_Ip = runtime_params.profile_conditions.Ip
      Ip_scale_factor = param_Ip / geo.Ip_profile_face[-1]
      geo = dataclasses.replace(
          geo,
          Ip_profile_face=geo.Ip_profile_face * Ip_scale_factor,
          psi_from_Ip=geo.psi_from_Ip * Ip_scale_factor,
          psi_from_Ip_face=geo.psi_from_Ip_face * Ip_scale_factor,
          j_total=geo.j_total * Ip_scale_factor,
          j_total_face=geo.j_total_face * Ip_scale_factor,
      )
    else:
      # If Ip is from the geometry, update the parameters to match.
      runtime_params = dataclasses.replace(
          runtime_params,
          profile_conditions=dataclasses.replace(
              runtime_params.profile_conditions,
              Ip=geo.Ip_profile_face[-1],
          ),
      )
  return runtime_params, geo
