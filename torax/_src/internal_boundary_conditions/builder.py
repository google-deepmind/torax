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

"""Builder for active simulation internal boundary conditions."""

import jax
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import profiles
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import (
    internal_boundary_conditions as internal_boundary_conditions_lib,
)
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


def build_internal_boundary_conditions(
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    core_profiles: profiles.CoreProfiles,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
    internal_boundary_condition_model: (
        base_model.InternalBoundaryConditionModel
    ),
    source_profiles: source_profiles_lib.SourceProfiles,
) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
  """Builds the active internal boundary conditions for this time step.

  When a pedestal model is actively controlling the edge (either in H-mode or
  during an L-H/H-L transition), user internal boundary conditions from
  profile_conditions are turned off to prevent conflicting boundary conditions
  or discontinuous cliffs at the edge.

  Args:
    runtime_params: Runtime parameters for the simulation.
    geo: Geometry of the torus.
    core_profiles: Core plasma profiles.
    pedestal_model_output: Output of the pedestal model for this step.
    internal_boundary_condition_model: Model used to evaluate profile-condition
      internal boundary conditions.
    source_profiles: Core source profiles.

  Returns:
    The active InternalBoundaryConditions object.
  """
  config_internal_boundary_conditions = internal_boundary_condition_model(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_profiles=source_profiles,
  )
  if isinstance(
      pedestal_model_output, pedestal_model_output_lib.IBCPedestalModelOutput
  ):
    pedestal_internal_boundary_conditions = (
        pedestal_model_output.to_internal_boundary_conditions(
            geo=geo,
            core_profiles=core_profiles,
            pedestal_profile_form=(
                runtime_params.pedestal.structure.pedestal_profile_form
            ),
        )
    )
    return jax.tree.map(
        lambda p, u: jax.lax.select(pedestal_model_output.is_active, p, u),
        pedestal_internal_boundary_conditions,
        config_internal_boundary_conditions,
    )
  return config_internal_boundary_conditions
