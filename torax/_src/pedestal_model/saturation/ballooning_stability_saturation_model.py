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

"""Saturation model based on ballooning stability limit."""

import dataclasses
import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src import math_utils
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.pedestal_model.saturation import base
from torax._src.physics import formulas

# pylint: disable=invalid-name


def calculate_normalized_pressure_gradient(
    core_profiles: state.CoreProfiles,
    geo: geometry.Geometry,
) -> array_typing.FloatVector:
  """Calculates the normalized pressure gradient (alpha).

  Equation:
    alpha = -2*mu_0 * (dV/dpsi) * (1/(2*pi)^2) * sqrt(V / (2*pi^2*R_0)) *
    (dp/dpsi)

  Args:
    core_profiles: CoreProfiles object containing information on pressures and
      psi.
    geo: Geometry object.

  Returns:
    alpha: Normalized pressure gradient evaluated on the face grid.
  """
  dp_dpsi = formulas.calc_pprime(core_profiles)
  dpsi_drhon = core_profiles.psi.face_grad()
  # vpr is dV/drhon, so dV/dpsi = vpr / dpsi/drhon
  dV_dpsi = math_utils.safe_divide(geo.vpr_face, dpsi_drhon)

  # Plasma volume enclosed by the flux surface (V) and major radius (R_0)
  V = geo.volume_face
  R_0 = geo.R_major

  # Calculate alpha
  return (
      -2.0
      * constants.CONSTANTS.mu_0
      * dV_dpsi
      * (1.0 / (2.0 * jnp.pi) ** 2)
      * jnp.sqrt(V / (2.0 * jnp.pi**2 * R_0))
      * dp_dpsi
  )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BallooningStabilitySaturationRuntimeParams(
    pedestal_runtime_params_lib.SaturationRuntimeParams
):
  """Runtime params for ballooning stability saturation models."""

  alpha_crit: array_typing.FloatScalar


@dataclasses.dataclass(frozen=True, eq=False)
class BallooningStabilitySaturationModel(base.SaturationModel):
  """Saturation model based on the maximum pressure gradient alpha_crit."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_output: pedestal_model_output.PedestalModelOutput,
  ) -> array_typing.FloatScalar:
    """Calculates transport increase multipliers."""
    assert isinstance(
        runtime_params.pedestal.saturation,
        BallooningStabilitySaturationRuntimeParams,
    )

    alpha = calculate_normalized_pressure_gradient(core_profiles, geo)
    max_alpha_ped = jnp.max(
        jnp.where(
            geo.rho_face_norm >= pedestal_output.rho_norm_ped_top, alpha, 0.0
        )
    )
    multiplier = self._calculate_multiplier(
        current=max_alpha_ped,
        target=runtime_params.pedestal.saturation.alpha_crit,
        pedestal_runtime_params=runtime_params.pedestal,
    )
    return pedestal_model_output.TransportMultipliers(
        chi_e_multiplier=multiplier,
        chi_i_multiplier=multiplier,
        D_e_multiplier=multiplier,
        v_e_multiplier=multiplier,
    )

  def _calculate_multiplier(
      self,
      current: array_typing.FloatScalar,
      target: array_typing.FloatScalar,
      pedestal_runtime_params: pedestal_runtime_params_lib.RuntimeParams,
  ) -> array_typing.FloatScalar:
    """Calculates the transport increase multiplier.

    If current >> target, multiplier -> infinity.
    If current << target, multiplier -> 1.

    Args:
      current: The current value of the profile at the pedestal top.
      target: The target value of the profile at the pedestal top.
      pedestal_runtime_params: The runtime parameters for the pedestal model.

    Returns:
      The transport increase multiplier.
    """
    steepness = pedestal_runtime_params.saturation.steepness
    offset = pedestal_runtime_params.saturation.offset
    base_multiplier = pedestal_runtime_params.saturation.base_multiplier
    normalized_deviation = (current - target) / target - offset
    transport_multiplier = 1 + base_multiplier * jax.nn.softplus(
        normalized_deviation * steepness
    )
    return transport_multiplier
