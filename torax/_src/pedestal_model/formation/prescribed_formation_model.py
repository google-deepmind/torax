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

"""Prescribed pedestal formation model."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base
from torax._src.sources import source_profiles as source_profiles_lib


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PrescribedFormationRuntimeParams(
    pedestal_runtime_params_lib.FormationRuntimeParams
):
  """Runtime params for prescribed pedestal formation model."""

  pedestal_active: array_typing.BoolScalar


@dataclasses.dataclass(frozen=True, eq=False)
class PrescribedFormationModel(base.FormationModel):
  """Prescribed pedestal formation model driven by a boolean schedule."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
      pedestal_transition_state: pedestal_transition_state_lib.PedestalTransitionState,
  ) -> pedestal_model_output.TransportMultipliers:
    del geo, core_profiles, source_profiles, pedestal_transition_state
    assert isinstance(
        runtime_params.pedestal.formation, PrescribedFormationRuntimeParams
    )
    transport_multiplier = jnp.where(
        runtime_params.pedestal.formation.pedestal_active,
        runtime_params.pedestal.formation.base_multiplier,
        1.0,
    )
    return pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=transport_multiplier,
        chi_i_multiplier=transport_multiplier,
        D_e_multiplier=transport_multiplier,
        v_e_multiplier=transport_multiplier,
    )

  def evaluate_transition_conditions(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      source_profiles: source_profiles_lib.SourceProfiles,
  ) -> tuple[array_typing.BoolScalar, array_typing.BoolScalar]:
    del geo, core_profiles, source_profiles
    assert isinstance(
        runtime_params.pedestal.formation, PrescribedFormationRuntimeParams
    )
    trigger_l_to_h_transition = jnp.asarray(
        runtime_params.pedestal.formation.pedestal_active, dtype=bool
    )
    return trigger_l_to_h_transition, ~trigger_l_to_h_transition
