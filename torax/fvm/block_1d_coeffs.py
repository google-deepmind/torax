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

"""The Block1DCoeffs dataclass and a callback protocol for communicating it.

This is the key interface between the `fvm` module, which is abstracted to the
level of a coupled 1D fluid dynamics PDE, and the rest of `torax`, which
includes
calculations specific to plasma physics to provide these coefficients.
"""

from typing import Any, Optional, Protocol, TypeAlias

import chex
import jax
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable


# An optional argument, consisting of a 2D matrix of nested tuples, with each
# leaf being either None or a JAX Array. Used to define block matrices.
# examples:
#
# ((a, b), (c, d)) where a, b, c, d are each jax.Array
#
# ((a, None), (None, d)) : represents a diagonal block matrix
OptionalTupleMatrix: TypeAlias = Optional[
    tuple[tuple[Optional[jax.Array], ...], ...]
]


# Alias for better readability.
AuxiliaryOutput: TypeAlias = Any


@chex.dataclass(frozen=True)
class Block1DCoeffs:
  # pyformat: disable  # pyformat removes line breaks needed for readability
  """The coefficients of coupled 1D fluid dynamics PDEs.

  The differential equation is:
  transient_out_coeff partial x transient_in_coeff / partial t = F
  where F =
  divergence(diffusion_coeff * grad(x))
  - divergence(convection_coeff * x)
  + source_mat_coeffs * u
  + sources.

  source_mat_coeffs exists for specific classes of sources where this
  decomposition is valid, allowing x to be treated implicitly in linear solvers,
  even if source_mat_coeffs contains state-dependent terms

  This class captures a snapshot of the coefficients of the equation at one
  instant in time, discretized spatially across a mesh.

  This class imposes the following structure on the problem:
  - It assumes the variables are arranged on a 1-D, evenly spaced grid.
  - It assumes the x variable is broken up into "channels," so the resulting
  matrix equation has one block per channel.

  Attributes:
    transient_out_cell: Tuple with one entry per channel, transient_out_cell[i]
      gives the transient coefficients outside the time derivative for channel i
      on the cell grid.
    transient_in_cell: Tuple with one entry per channel, transient_in_cell[i]
      gives the transient coefficients inside the time derivative for channel i
      on the cell grid.
    d_face: Tuple, with d_face[i] containing diffusion term coefficients for
      channel i on the face grid.
    v_face: Tuple, with v_face[i] containing convection term coefficients for
      channel i on the face grid.
    source_mat_cell: 2-D matrix of Tuples, with source_mat_cell[i][j] adding to
      block-row i a term of the form source_cell[j] * u[channel j]. Depending on
      the source runtime_params, may be constant values for a timestep, or
      updated iteratively with new states in a nonlinear solver
    source_cell: Additional source terms on the cell grid for each channel.
      Depending on the source runtime_params, may be constant values for a
      timestep, or updated iteratively with new states in a nonlinear solver
    auxiliary_outputs: Optional extra output which can include auxiliary state
      or information useful for inspecting the computation inside the callback
      which calculated these coeffs.
  """
  transient_in_cell: tuple[jax.Array, ...]
  transient_out_cell: Optional[tuple[jax.Array, ...]] = None
  d_face: Optional[tuple[jax.Array, ...]] = None
  v_face: Optional[tuple[jax.Array, ...]] = None
  source_mat_cell: OptionalTupleMatrix = None
  source_cell: Optional[tuple[Optional[jax.Array], ...]] = None
  auxiliary_outputs: Optional[AuxiliaryOutput] = None


class Block1DCoeffsCallback(Protocol):
  """Protocol for calculating the Block1DCoeffs for a state."""

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      x: tuple[cell_variable.CellVariable, ...],
      allow_pereverzev: bool = False,
      explicit_call: bool = False,
  ) -> Block1DCoeffs:
    """Returns coefficients given a state. Can be called in various modes.

    The typical sequence is:
    call 1: x=x_old
    call 2 (optional): x=x_old, allow_pereverzev=True
    Subsequent calls until iterations are finished: x=x_new

    x_old is the initial state at the beginning of the iterations.
    The optional call 2 is done when a linear step is used for the initial
    guess for the nonlinear iterations, and may involve the transport coeffs
    including the Pereverzev Corrigan terms.
    All subsequent calls are within the nonlinear iteration loops, where
    x_new is a solution from within the iterations and may not yet be the
    final output x_new.

    Args:
      dynamic_runtime_params_slice: Runtime configuration parameters. These
        values are potentially time-dependent and should correspond to the time
        step of the state x.
      geo: The geometry of the system at this time step.
      core_profiles: The core profiles of the system at this time step.
      x: The state.
      allow_pereverzev: If True, then the coeffs are being called for an initial
        guess based on a linear step as opposed to just passing the iniitial
        state. This is a special case which may lead to the pereverzev-corrigan
        terms being included in calc_coeffs.
      explicit_call: If True, then if theta_imp=1, only a reduced Block1DCoeffs
        is calculated since most explicit coefficients will not be used.

    Returns:
      coeffs: The diffusion, convection, etc. coefficients for this state.
    """
    ...
