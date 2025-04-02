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
import typing

import chex
from torax.config import numerics
from torax.config import plasma_composition
from torax.config import profile_conditions
from torax.geometry import geometry
from torax.geometry import standard_geometry
from torax.mhd import runtime_params as mhd_runtime_params
from torax.pedestal_model import runtime_params as pedestal_model_params
from torax.sources import runtime_params as sources_params
from torax.stepper import runtime_params as stepper_params
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import runtime_params as transport_model_params
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
  without trigerring or requiring a recompile.

  While the parameters are not necessarily time-dependent, that is how the class
  gets its name: a config "slice" refers to a subset of the overall TORAX config
  at a specific time t.

  This class contains "slices" of various RuntimeParams attributes defined
  throughout TORAX:

  - from the profile_conditions runtime params
  - from the numerics runtime params
  - from the plasma_composition runtime params
  - from the transport model's runtime params
  - from the stepper's runtime params
  - from each of the sources' runtime params
  - from the pedestal model's runtime params
  - from each of the mhd models' runtime params

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
  mhd: mhd_runtime_params.DynamicMHDParams


@chex.dataclass(frozen=True)
class StaticRuntimeParamsSlice:
  """Static runtime parameters for a simulation time step.

  This config slice contains all the parameters that, if they change, should
  trigger a recompilation of the simulation step function. The parameters here
  are ones which are accessed in the JIT-compiled forward pass. They should be
  immutable (hence the @chex.dataclass(frozen=True)), and be hashable.

  Changing the elements of this config slice should be paired with a
  recompilation of the simulation step.

  Technically the params in this config slice are static in that they don't
  change during the course of one simulation step. But they could change between
  simulation steps. E.g. if you want to simulate an ITER scenario with
  additional NBI at a certain point in time, you could encode it via a large
  step function that recompiles at specific time points with different static
  params. We do not provide that currently.
  """

  # ----- What to evolve with transport equations -----
  # Whether to evolve ion and electron temperatures separately or consider
  # them equal.
  ion_heat_eq: bool
  el_heat_eq: bool
  current_eq: bool
  dens_eq: bool

  # ----- Source helpers -----
  # Compute the bootstrap current from the plasma pressure, density and
  # temperature during the sim.
  use_bootstrap_calc: bool
  # For the current diffusion, imposes a loop voltage boundary condition at LCFS
  # instead of boundary condition on total plasma current.
  use_vloop_lcfs_boundary_condition: bool
  # Whether to use adaptive time stepping for large nonlinear PDE steps
  adaptive_dt: bool
  # Whether to show the progress bar during simulations
  show_progress_bar: bool
  # Whether to run sanity checks during the simulation
  enable_sanity_checks: bool

  def __eq__(self, other: typing.Any) -> bool:
    """Implements StaticRuntimeParamsSlice equality.

    Args:
      other: rhs to __eq__

    Returns:
      True if StaticRuntimeParamsSlice and other are the same.
    """
    # All members of this class are atomic Python types, so simple equality
    # suffices.
    if not isinstance(other, StaticRuntimeParamsSlice):
      return False
    
    # List of primary fields we care about for equality
    primary_fields = [
        'ion_heat_eq', 'el_heat_eq', 'current_eq', 'dens_eq',
        'use_bootstrap_calc', 'use_vloop_lcfs_boundary_condition',
        'adaptive_dt', 'show_progress_bar'
    ]
    
    # Compare all primary fields
    for field in primary_fields:
        if getattr(self, field) != getattr(other, field):
            return False
    
    # Handle enable_sanity_checks separately for backward compatibility
    self_enable_checks = getattr(self, 'enable_sanity_checks', False)
    other_enable_checks = getattr(other, 'enable_sanity_checks', False)
    if self_enable_checks != other_enable_checks:
        return False
    
    return True
  
  def __hash__(self) -> int:
    """Implements StaticRuntimeParamsSlice hash.

    Returns:
      Hash value for this instance.
    """
    # List of fields to include in hash
    primary_fields = [
        'ion_heat_eq', 'el_heat_eq', 'current_eq', 'dens_eq',
        'use_bootstrap_calc', 'use_vloop_lcfs_boundary_condition',
        'adaptive_dt', 'show_progress_bar'
    ]
    
    # Create a tuple of field values for hashing
    values = tuple(getattr(self, field) for field in primary_fields)
    
    # Add enable_sanity_checks with backward compatibility
    return hash(values + (getattr(self, 'enable_sanity_checks', False),))


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
      param_Ip = dynamic_runtime_params_slice.profile_conditions.Ip_tot
      Ip_scale_factor = param_Ip * 1e6 / geo.Ip_profile_face[-1]
      geo = dataclasses.replace(
          geo,
          Ip_profile_face=geo.Ip_profile_face * Ip_scale_factor,
          psi_from_Ip=geo.psi_from_Ip * Ip_scale_factor,
          psi_from_Ip_face=geo.psi_from_Ip_face * Ip_scale_factor,
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
