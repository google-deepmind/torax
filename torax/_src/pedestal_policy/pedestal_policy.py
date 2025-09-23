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

"""The PedestalPolicy abstract base class.

Determines potentially changing pedestal settings during simulation.
"""
import abc
import dataclasses
from typing import Optional

import jax
from torax._src import array_typing

# pylint: disable=invalid-name
# Using physics notation naming convention


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalPolicyState:
  """State of the PedestalPolicy."""

  # Whether to use the pedestal on this time step
  use_pedestal: array_typing.BoolScalar
  # Factor to scale pedestal height by.
  # Not all pedestal policies use this.
  scale_pedestal: Optional[array_typing.FloatScalar] = None


class PedestalPolicy(abc.ABC):
  """Determines potentially changing pedestal settings during simulation.

  Subclass responsbilities:
  - Must set _frozen = True at the end of the subclass __init__ method to
    activate immutability.
  """

  @abc.abstractmethod
  def initial_state(self, t: float) -> PedestalPolicyState:
    """Creates the initial state for policy-related variables.

    Args:
      t: Time.

    Returns:
      initial_state: A PedestalPolicyState.
    """

  def __setattr__(self, attr, value):
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """Override __setattr__ to make the class (sort of) immutable.

    Note that you can still do obj.field.subfield = x, so it is not true
    immutability, but this to helps to avoid some careless errors.
    """
    if getattr(self, "_frozen", False):
      raise AttributeError("PedestalPolicy is immutable.")
    return super().__setattr__(attr, value)

  # @abc.abstractmethod
  # def _call_implementation(
  #    self,
  #    runtime_params: runtime_params_slice.RuntimeParams,
  #    geo: geometry.Geometry,
  #    core_profiles: state.CoreProfiles,
  # ) -> PedestalModelOutput:
  #  """Calculate the pedestal values."""

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Hash function for the pedestal policy.

    Needed for jax.jit caching to work.
    """
    ...

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Equality function for the pedestal model."""
    ...
