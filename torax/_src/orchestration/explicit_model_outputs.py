# Copyright 2026 DeepMind Technologies Limited
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
"""Container for physics model outputs evaluated explicitly at time t."""

import dataclasses
import jax
from torax._src.pedestal_model import pedestal_model_output
from torax._src.sources import source_profiles


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExplicitModelOutputs:
  """Outputs of physics models evaluated explicitly at the start of a step.

  These outputs are computed once in `pre_step` at time `t` and held fixed
  across solver iterations from `t` to `t + dt`.

  Attributes:
    sources: Source profiles from sources configured in EXPLICIT mode.
    pedestal: Pedestal structure evaluated explicitly at time `t` when
      `explicit_pedestal` is True, or None when evaluated implicitly.
  """

  sources: source_profiles.SourceProfiles
  pedestal: pedestal_model_output.PedestalStructure | None
