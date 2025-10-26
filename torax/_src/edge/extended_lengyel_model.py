# Copyright 2025 DeepMind Technologies Limited
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

"""Implementation of extended_lengyel instance of EdgeModel."""

import dataclasses
from typing import Mapping
import jax
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.edge import base
from torax._src.edge import extended_lengyel_solvers
from torax._src.geometry import geometry

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(base.RuntimeParamsBase):
  """Runtime parameters for the extended Lengyel edge model."""

  pass
  # TODO(b/446608829) - to be completed in a later PR.


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ExtendedLengyelOutputs(base.EdgeModelOutputsBase):
  """Additional outputs from the extended Lengyel model.

  Attributes:
    alpha_t: Turbulence broadening factor alpha_t.
    separatrix_Z_eff: Z_eff at the separatrix.
    seed_impurity_concentrations: A mapping from ion symbol to its n_e_ratio.
    solver_status: Status of the solver.
  """

  alpha_t: jax.Array
  separatrix_Z_eff: jax.Array
  seed_impurity_concentrations: Mapping[str, jax.Array]
  solver_status: extended_lengyel_solvers.ExtendedLengyelSolverStatus


@dataclasses.dataclass(frozen=True, eq=False)
class ExtendedLengyelModel(base.EdgeModelBase):

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> ExtendedLengyelOutputs:
    # TODO(b/446608829) - to be completed in a later PR.
    raise NotImplementedError
