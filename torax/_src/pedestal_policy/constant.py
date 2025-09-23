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

"""The Constant policy.

Pedestal is always off or always on.
"""
from torax._src.pedestal_policy import pedestal_policy

# pylint: disable=invalid-name
# Using physics notation naming convention


class Constant(pedestal_policy.PedestalPolicy):
  """PedestalPolicy that sets the pedestal to always off or always on.

  Attributes:
    use_pedestal: Constant value of `use_pedestal` used on every time step.
  """

  def __init__(self, use_pedestal):
    self.constant_state = pedestal_policy.PedestalPolicyState(
        use_pedestal=use_pedestal
    )
    self._frozen = True

  def initial_state(self, t: float) -> pedestal_policy.PedestalPolicyState:
    """Creates the initial state for policy-related variables.

    Args:
      t: Time

    Returns:
      initial_state: A PedestalPolicyState.
    """

    return self.constant_state

  def __hash__(self) -> int:
    """Hash function needed for jax.jit caching to work."""
    return hash(("Constant", self.constant_state))

  def __eq__(self, other) -> bool:
    """Equality function for the pedestal model."""
    return (
        isinstance(other, Constant)
        and other.constant_state == self.constant_state
    )
