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

"""The TimeVarying policy.

Determines potentially changing pedestal settings during simulation.
"""
from __future__ import annotations
import dataclasses
from torax._src.pedestal_policy import pedestal_policy
from torax._src.pedestal_policy import runtime_params as pedestal_policy_runtime_params

# pylint: disable=invalid-name
# Using physics notation naming convention


@dataclasses.dataclass(frozen=True, eq=False)
class TimeVarying(pedestal_policy.PedestalPolicy):
  """Set `use_pedestal` following a TimeVaryingScalar."""

  def initial_state(
      self,
      t: float,
      runtime_params: pedestal_policy_runtime_params.PedestalPolicyRuntimeParams,
  ) -> pedestal_policy.PedestalPolicyState:
    """Creates the initial state for policy-related variables."""
    if not isinstance(
        runtime_params, pedestal_policy_runtime_params.TimeVaryingRP
    ):
      raise TypeError(f"Expected TimeVaryingRP, got {type(runtime_params)}")
    if not runtime_params.is_bool_param:
      raise ValueError(
          "TimeVarying pedestal policy must use data with is_bool_param=True."
      )
    use_pedestal = runtime_params.get_value(t)
    return pedestal_policy.PedestalPolicyState(use_pedestal=use_pedestal)

  def update(
      self,
      t: float,
      runtime_params: pedestal_policy_runtime_params.PedestalPolicyRuntimeParams,
  ) -> pedestal_policy.PedestalPolicyState:
    """Calculate updated pedestal policy state."""
    return self.initial_state(t, runtime_params)
