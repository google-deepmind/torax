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

"""Component transport model base class and output types.

Defines the `ComponentTransportModel` abstract base class for individual
transport physics models (such as Bohm-GyroBohm, QLKNN, or TGLF) that compute
turbulent heat and particle transport coefficients.
"""

import abc
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coeffs

# pylint: disable=invalid-name


@dataclasses.dataclass(frozen=True, eq=False)
class ComponentTransportModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates various coefficients related to heat and particle transport."""

  def __call__(
      self,
      transport_runtime_params: transport_runtime_params_lib.ComponentRuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    """Computes transport coefficients and zeros out disabled channels.

    Delegates to call_implementation to compute the raw transport coefficients,
    then zeros out any channels that are disabled in the runtime params.

    Args:
      transport_runtime_params: Runtime parameters for the transport model.
      runtime_params: Runtime parameters for the simulation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      two_point_mask: Boolean mask on the face grid indicating where to use
        2-point central differencing instead of 3-point polynomial interpolation
        for gradients.

    Returns:
      Transport coefficients with disabled channels zeroed out.
    """
    coeffs = self.call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        two_point_mask=two_point_mask,
    )
    coeffs = self.zero_out_disabled_channels(transport_runtime_params, coeffs)
    return coeffs

  @abc.abstractmethod
  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.ComponentRuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> transport_coeffs.TransportCoeffs:
    pass

  def zero_out_disabled_channels(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      coeffs: transport_coeffs.TransportCoeffs,
  ) -> transport_coeffs.TransportCoeffs:
    """Sets coefficients to zero for channels that are disabled."""
    return dataclasses.replace(
        coeffs,
        chi_face_ion=jnp.where(
            transport_runtime_params.disable_chi_i,
            0.0,
            coeffs.chi_face_ion,
        ),
        chi_face_el=jnp.where(
            transport_runtime_params.disable_chi_e,
            0.0,
            coeffs.chi_face_el,
        ),
        d_face_el=jnp.where(
            transport_runtime_params.disable_D_e,
            0.0,
            coeffs.d_face_el,
        ),
        v_face_el=jnp.where(
            transport_runtime_params.disable_V_e,
            0.0,
            coeffs.v_face_el,
        ),
    )


def compute_core_domain_mask(
    transport_runtime_params: transport_runtime_params_lib.ComponentRuntimeParams,
    runtime_params: runtime_params_lib.RuntimeParams,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
) -> jax.Array:
  """Calculates the active domain mask for core transport models.

  Args:
    transport_runtime_params: Runtime parameters for the transport model.
    runtime_params: Runtime parameters for the simulation.
    geo: Geometry of the torus.
    pedestal_model_output: Output of the pedestal model.

  Returns:
    active_mask: A boolean array indicating the active domain.
  """
  # Active range is rho_min < rho <= rho_max
  # (AND rho <= rho_norm_ped_top, if pedestal is in INTERNAL_BOUNDARY_CONDITION
  # mode)
  active_mask = (geo.rho_face_norm > transport_runtime_params.rho_min) & (
      geo.rho_face_norm <= transport_runtime_params.rho_max
  )
  if (
      runtime_params.pedestal.mode
      == pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
  ):
    active_mask = active_mask & (
        jnp.logical_not(runtime_params.pedestal.set_pedestal)
        | (geo.rho_face_norm < pedestal_model_output.rho_norm_ped_top)
    )

  # Special case: if rho_min is 0, lower bound of active range is the first
  # grid point.
  active_mask = (
      jnp.asarray(active_mask).at[0].set(transport_runtime_params.rho_min == 0)
  )
  return active_mask

