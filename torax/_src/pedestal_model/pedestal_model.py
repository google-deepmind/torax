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

"""The PedestalModel abstract base class.

The pedestal model calculates quantities relevant to the pedestal.
"""
import abc
import dataclasses

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import state
from torax._src import static_dataclass
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.internal_boundary_conditions import internal_boundary_conditions as internal_boundary_conditions_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.transport_model import turbulent_transport as turbulent_transport_lib

# pylint: disable=invalid-name
# Using physics notation naming convention


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveSourcePedestalModelOutput:
  """Output of a PedestalModel in ADAPTIVE_SOURCE mode."""

  # The location of the pedestal top.
  rho_norm_ped_top: array_typing.FloatScalar
  # The index of the pedestal top in rho_norm.
  rho_norm_ped_top_idx: array_typing.IntScalar
  # The ion temperature at the pedestal top.
  T_i_ped: array_typing.FloatScalar
  # The electron temperature at the pedestal top.
  T_e_ped: array_typing.FloatScalar
  # The electron density at the pedestal top.
  n_e_ped: array_typing.FloatScalar

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions."""
    # In this case, the mask is only the pedestal top, not the whole pedestal
    # region. This is because we are adding a source/sink term only at the
    # pedestal top.
    pedestal_mask = (
        jnp.zeros_like(geo.rho, dtype=bool)
        .at[self.rho_norm_ped_top_idx]
        .set(True)
    )
    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
        T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
        n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveTransportPedestalModelOutput:
  """Output of a PedestalModel in ADAPTIVE_TRANSPORT mode."""

  # The location of the pedestal top.
  rho_norm_ped_top: array_typing.FloatScalar
  # The index of the pedestal top in rho_norm.
  rho_norm_ped_top_idx: array_typing.IntScalar
  # The multipliers for the turbulent transport coefficients.
  chi_e_multiplier: array_typing.FloatScalar
  chi_i_multiplier: array_typing.FloatScalar
  D_e_multiplier: array_typing.FloatScalar
  v_e_multiplier: array_typing.FloatScalar

  def combine_with_turbulent_transport(
      self,
      turbulent_transport: turbulent_transport_lib.TurbulentTransport,
      geo: geometry.Geometry,
  ) -> turbulent_transport_lib.TurbulentTransport:
    """Combine the pedestal model output with the turbulent transport coefficients."""

    # In this case, the mask is the whole pedestal region, not just the top.
    # This is because we are modifying the transport coefficients in the whole
    # pedestal region.
    pedestal_mask_face = (
        jnp.zeros_like(geo.rho_face, dtype=bool)
        .at[self.rho_norm_ped_top_idx :]
        .set(True)
    )

    modified_chi_face_ion = jnp.where(
        pedestal_mask_face,
        turbulent_transport.chi_face_ion * self.chi_i_multiplier,
        turbulent_transport.chi_face_ion,
    )
    modified_chi_face_el = jnp.where(
        pedestal_mask_face,
        turbulent_transport.chi_face_el * self.chi_e_multiplier,
        turbulent_transport.chi_face_el,
    )
    modified_d_face_el = jnp.where(
        pedestal_mask_face,
        turbulent_transport.d_face_el * self.D_e_multiplier,
        turbulent_transport.d_face_el,
    )
    modified_v_face_el = jnp.where(
        pedestal_mask_face,
        turbulent_transport.v_face_el * self.v_e_multiplier,
        turbulent_transport.v_face_el,
    )
    return turbulent_transport_lib.TurbulentTransport(
        chi_face_ion=modified_chi_face_ion,
        chi_face_el=modified_chi_face_el,
        d_face_el=modified_d_face_el,
        v_face_el=modified_v_face_el,
    )


PedestalModelOutput = (
    AdaptiveSourcePedestalModelOutput | AdaptiveTransportPedestalModelOutput
)


@dataclasses.dataclass(frozen=True, eq=False)
class PedestalModel(static_dataclass.StaticDataclass, abc.ABC):
  """Calculates properties of the pedestal."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    if (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE
    ):
      # Set the pedestal location to infinite to indicate that the pedestal is
      # not present.
      # Set the index to outside of bounds of the mesh to indicate that the
      # pedestal is not present.
      dummy_output = AdaptiveSourcePedestalModelOutput(
          rho_norm_ped_top=jnp.inf,
          T_i_ped=0.0,
          T_e_ped=0.0,
          n_e_ped=0.0,
          rho_norm_ped_top_idx=geo.torax_mesh.nx,
      )
    elif (
        runtime_params.pedestal.mode
        == pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT
    ):
      # Set the pedestal location to infinite to indicate that the pedestal is
      # not present.
      # Set the index to outside of bounds of the mesh to indicate that the
      # pedestal is not present.
      # Set the multipliers to 1.0 to indicate that the transport coefficients
      # are not modified.
      dummy_output = AdaptiveTransportPedestalModelOutput(
          rho_norm_ped_top=jnp.inf,
          rho_norm_ped_top_idx=geo.torax_mesh.nx,
          chi_e_multiplier=1.0,
          chi_i_multiplier=1.0,
          D_e_multiplier=1.0,
          v_e_multiplier=1.0,
      )
    else:
      raise ValueError(
          f'Unsupported pedestal model mode: {runtime_params.pedestal.mode}'
      )

    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        lambda: self._call_implementation(runtime_params, geo, core_profiles),
        lambda: dummy_output,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    """Calculate the pedestal properties."""
