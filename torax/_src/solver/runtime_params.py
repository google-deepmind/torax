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
"""Runtime params for the solver."""
import dataclasses

import jax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Input params for the solver which can be used as compiled args."""
  theta_implicit: float = dataclasses.field(metadata={'static': True})
  use_predictor_corrector: bool = dataclasses.field(metadata={'static': True})
  n_corrector_steps: int = dataclasses.field(metadata={'static': True})
  convection_dirichlet_mode: str = dataclasses.field(metadata={'static': True})
  convection_neumann_mode: str = dataclasses.field(metadata={'static': True})
  use_pereverzev: bool = dataclasses.field(metadata={'static': True})
  chi_pereverzev: float
  D_pereverzev: float  # pylint: disable=invalid-name
