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

"""Power scaling pedestal formation model."""

import dataclasses
import enum
from typing import Literal
import jax
from torax._src import array_typing
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.formation import base
from torax._src.physics import scaling_laws
from torax._src.sources import source_profiles as source_profiles_lib

# pylint: disable=invalid-name


@enum.unique
class ScalingLaw(enum.Enum):
  """Defines which scaling law to use for pedestal formation."""

  MARTIN = "MARTIN"
  DELABIE = "DELABIE"


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PowerScalingFormationRuntimeParams(
    pedestal_runtime_params_lib.FormationRuntimeParams
):
  """Runtime params for power scaling pedestal formation models."""

  P_LH_prefactor: array_typing.FloatScalar = 1.0


def _calculate_P_SOL_total(
    internal_plasma_energy: state.PlasmaInternalEnergy,
    core_sources: source_profiles_lib.SourceProfiles,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates the total power out of the separatrix."""
  P_heat_e = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_e.values()
  )
  P_heat_i = sum(
      math_utils.volume_integration(source, geo)
      for source in core_sources.T_i.values()
  )
  return P_heat_e + P_heat_i - internal_plasma_energy.dW_thermal_dt_smoothed


@dataclasses.dataclass(frozen=True, eq=False)
class PowerScalingFormationModel(base.FormationModel):
  """Pedestal formation based on P_SOL and P_LH thresholds.

  Attributes:
    scaling_law: The scaling law to use for pedestal formation.
    divertor_configuration: The divertor configuration. Only used for the
      Delabie scaling law.
  """

  scaling_law: ScalingLaw
  divertor_configuration: Literal["HT", "VT"] | None = None

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      core_sources: source_profiles_lib.SourceProfiles,
  ) -> pedestal_model_output.TransportMultipliers:
    """Calculates transport decrease multipliers based on P_SOL and P_LH."""
    assert isinstance(
        runtime_params.pedestal.formation, PowerScalingFormationRuntimeParams
    )

    P_SOL_total = _calculate_P_SOL_total(
        core_profiles.internal_plasma_energy, core_sources, geo
    )

    match self.scaling_law:
      case ScalingLaw.MARTIN:
        _, _, P_LH, _ = scaling_laws.calculate_plh_martin(geo, core_profiles)
      case ScalingLaw.DELABIE:
        P_LH = scaling_laws.calculate_plh_delabie(
            geo,
            core_profiles,
            divertor_configuration=self.divertor_configuration,
        )
      case _:
        raise ValueError(
            "Unknown scaling law:"
            f" {runtime_params.pedestal.formation.scaling_law}"
        )

    rescaled_P_LH = P_LH * runtime_params.pedestal.formation.P_LH_prefactor

    # Calculate transport_multiplier
    # If P_SOL > P_LH, multiplier tends to 0.0
    # If P_SOL < P_LH, multiplier tends to 1.0
    # TODO(b/488393318): Add hysteresis to the LH-HL transition.
    sharpness = runtime_params.pedestal.formation.sharpness
    offset = runtime_params.pedestal.formation.offset
    base_multiplier = runtime_params.pedestal.formation.base_multiplier
    normalized_deviation = (P_SOL_total - rescaled_P_LH) / rescaled_P_LH
    shifted_deviation = normalized_deviation - offset
    alpha = jax.nn.sigmoid(shifted_deviation * sharpness)
    transport_multiplier = (1.0 - alpha) * 1.0 + alpha * base_multiplier

    return pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=transport_multiplier,
        chi_i_multiplier=transport_multiplier,
        D_e_multiplier=transport_multiplier,
        v_e_multiplier=transport_multiplier,
    )
