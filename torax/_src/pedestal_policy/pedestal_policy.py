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
from __future__ import annotations
import abc
import dataclasses
import jax
from torax._src import array_typing
from torax._src import static_dataclass
from torax._src.pedestal_policy import runtime_params as pedestal_policy_runtime_params

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
  scale_pedestal: array_typing.FloatScalar | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalPolicy(static_dataclass.StaticDataclass, abc.ABC):
  """Determines potentially changing pedestal settings during simulation."""

  @abc.abstractmethod
  def initial_state(
      self,
      t: float,
      runtime_params: pedestal_policy_runtime_params.PedestalPolicyRuntimeParams,
  ) -> PedestalPolicyState:
    """Creates the initial state for policy-related variables.

    Args:
      t: Time.
      runtime_params: Dynamic parameters for the policy.

    Returns:
      initial_state: A PedestalPolicyState.
    """

  @abc.abstractmethod
  def update(
      self,
      t: float,
      runtime_params: pedestal_policy_runtime_params.PedestalPolicyRuntimeParams,
  ) -> PedestalPolicyState:
    """Calculate updated pedestal policy state.

    Args:
      t: Time.
      runtime_params: Dynamic parameters for the policy.

    Returns:
      state: Updated PedestalPolicyState.
    """
