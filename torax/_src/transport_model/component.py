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
from typing import ClassVar, Mapping, Sequence

import immutabledict
import jax
from jax import numpy as jnp
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TurbulentTransport:
  """Turbulent transport coefficients calculated by a transport model.

  Attributes:
    chi_face_ion: Ion heat conductivity, on the face grid.
    chi_face_el: Electron heat conductivity, on the face grid.
    d_face_el: Diffusivity of electron density, on the face grid.
    v_face_el: Convection strength of electron density, on the face grid.
    chi_face_el_bohm: (Optional) Bohm contribution for electron heat
      conductivity.
    chi_face_el_gyrobohm: (Optional) GyroBohm contribution for electron heat
      conductivity.
    chi_face_ion_bohm: (Optional) Bohm contribution for ion heat conductivity.
    chi_face_ion_gyrobohm: (Optional) GyroBohm contribution for ion heat
      conductivity.
    chi_face_ion_itg: (Optional) ITG contribution for ion heat conductivity.
    chi_face_ion_tem: (Optional) TEM contribution for ion heat conductivity.
    chi_face_el_itg: (Optional) ITG contribution for electron heat conductivity.
    chi_face_el_tem: (Optional) TEM contribution for electron heat conductivity.
    chi_face_el_etg: (Optional) ETG contribution for electron heat conductivity.
    d_face_el_itg: (Optional) ITG contribution for electron diffusivity.
    d_face_el_tem: (Optional) TEM contribution for electron diffusivity.
    v_face_el_itg: (Optional) ITG contribution for electron convection.
    v_face_el_tem: (Optional) TEM contribution for electron convection.
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None
  chi_face_ion_itg: jax.Array | None = None
  chi_face_ion_tem: jax.Array | None = None
  chi_face_el_itg: jax.Array | None = None
  chi_face_el_tem: jax.Array | None = None
  chi_face_el_etg: jax.Array | None = None
  d_face_el_itg: jax.Array | None = None
  d_face_el_tem: jax.Array | None = None
  v_face_el_itg: jax.Array | None = None
  v_face_el_tem: jax.Array | None = None


@dataclasses.dataclass(frozen=True, eq=False)
class ComponentTransportModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates various coefficients related to heat and particle transport."""

  # Map main channels to their sub-channels (if any) and disable flags
  # TODO(b/434175938): Upgrade ComponentTransportModel to encapsulate this
  # structure.
  CHANNEL_CONFIG: ClassVar[
      Mapping[str, dict[str, Sequence[str] | str]]
  ] = (
      immutabledict.immutabledict({
          'chi_face_ion': {
              'sub_channels': [
                  'chi_face_ion_bohm',
                  'chi_face_ion_gyrobohm',
                  'chi_face_ion_itg',
                  'chi_face_ion_tem',
              ],
              'disable_flag': 'disable_chi_i',
          },
          'chi_face_el': {
              'sub_channels': [
                  'chi_face_el_bohm',
                  'chi_face_el_gyrobohm',
                  'chi_face_el_itg',
                  'chi_face_el_tem',
                  'chi_face_el_etg',
              ],
              'disable_flag': 'disable_chi_e',
          },
          'd_face_el': {
              'sub_channels': ['d_face_el_itg', 'd_face_el_tem'],
              'disable_flag': 'disable_D_e',
          },
          'v_face_el': {
              'sub_channels': ['v_face_el_itg', 'v_face_el_tem'],
              'disable_flag': 'disable_V_e',
          },
      })
  )

  def __call__(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    """Computes transport coefficients and zeros out disabled channels.

    Delegates to call_implementation to compute the raw transport coefficients,
    then zeros out any channels that are disabled in the runtime params.

    Args:
      transport_runtime_params: Runtime parameters for the transport model.
      runtime_params: Runtime parameters for the simulation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      Transport coefficients with disabled channels zeroed out.
    """
    coeffs = self.call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    )
    coeffs = self.zero_out_disabled_channels(transport_runtime_params, coeffs)
    return coeffs

  @abc.abstractmethod
  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    pass

  def zero_out_disabled_channels(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Sets coefficients to zero for channels that are disabled."""
    to_replace = {}

    for channel_name, config in self.CHANNEL_CONFIG.items():
      disable_flag = getattr(transport_runtime_params, config['disable_flag'])  # pyrefly: ignore[bad-argument-type]

      # Handle main channel
      val = getattr(transport_coeffs, channel_name)
      to_replace[channel_name] = jnp.where(disable_flag, 0.0, val)

      # Handle sub-channels
      for sub_channel in config['sub_channels']:
        sub_value = getattr(transport_coeffs, sub_channel)
        if sub_value is not None:
          sub_value = jnp.where(disable_flag, 0.0, sub_value)
        to_replace[sub_channel] = sub_value

    return dataclasses.replace(transport_coeffs, **to_replace)


def compute_core_domain_mask(
    transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
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
