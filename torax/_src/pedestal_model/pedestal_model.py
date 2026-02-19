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

# pylint: disable=invalid-name
# Using physics notation naming convention


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PedestalModelOutput(abc.ABC):
  """Output of a PedestalModel.

  Attributes:
    rho_norm_ped_top: The requested location of the pedestal top in rho_norm,
      not quantized to the mesh.
    rho_norm_ped_top_nearest_cell_idx: The index of the nearest cell to
      rho_norm_ped_top.
    rho_norm_ped_top_nearest_face_idx: The index of the nearest face to
      rho_norm_ped_top.
  """

  rho_norm_ped_top: array_typing.FloatScalar
  rho_norm_ped_top_nearest_cell_idx: array_typing.IntScalar
  rho_norm_ped_top_nearest_face_idx: array_typing.IntScalar


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveSourcePedestalModelOutput(PedestalModelOutput):
  """Output of a PedestalModel in ADAPTIVE_SOURCE mode.

  Attributes:
    T_i_ped: The ion temperature at the pedestal top in keV.
    T_e_ped: The electron temperature at the pedestal top in keV.
    n_e_ped: The electron density at the pedestal top in m^-3.
  """

  T_i_ped: array_typing.FloatScalar
  T_e_ped: array_typing.FloatScalar
  n_e_ped: array_typing.FloatScalar

  def to_internal_boundary_conditions(
      self,
      geo: geometry.Geometry,
  ) -> internal_boundary_conditions_lib.InternalBoundaryConditions:
    """Convert the pedestal model output to internal boundary conditions."""
    # In this case, the mask is only the pedestal top, not the whole pedestal
    # region. This is because we are adding a source/sink term only at the
    # pedestal top.
    # We are using the cell grid here, since internal boundary conditions are
    # applied using an adaptive source (which acts on the cell grid).
    pedestal_mask = (
        jnp.zeros_like(geo.rho, dtype=bool)
        .at[self.rho_norm_ped_top_nearest_cell_idx]
        .set(True)
    )
    return internal_boundary_conditions_lib.InternalBoundaryConditions(
        T_i=jnp.where(pedestal_mask, self.T_i_ped, 0.0),
        T_e=jnp.where(pedestal_mask, self.T_e_ped, 0.0),
        n_e=jnp.where(pedestal_mask, self.n_e_ped, 0.0),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class AdaptiveTransportPedestalModelOutput(PedestalModelOutput):
  """Output of a PedestalModel in ADAPTIVE_TRANSPORT mode.

  Attributes:
    chi_e_multiplier: Scaling factor for the electron thermal diffusivity in the
      pedestal region.
    chi_i_multiplier: Scaling factor for the ion thermal diffusivity in the
      pedestal region.
    D_e_multiplier: Scaling factor for the electron diffusion coefficient.
    v_e_multiplier: Scaling factor for the electron particle diffusion
      coefficient.
  """

  chi_e_multiplier: array_typing.FloatScalar
  chi_i_multiplier: array_typing.FloatScalar
  D_e_multiplier: array_typing.FloatScalar
  v_e_multiplier: array_typing.FloatScalar

  def modify_core_transport(
      self,
      core_transport: state.CoreTransport,
      geo: geometry.Geometry,
  ) -> state.CoreTransport:
    """Modify the turbulent+Pereverzev transport coefficients in the pedestal region."""
    # In this case, the mask is the whole pedestal region, not just the top.
    # This is because we are modifying the transport coefficients in the whole
    # pedestal region.
    # We are using the face grid here, since transport coefficients are
    # applied on the face grid.

    # TODO(b/485147781): As this happens after the transport model __call__ is
    # executed, transport coefficients in the pedestal will have been masked (to
    # 0 in the pedestal region) and clipped (to chi_min everywhere), so in the
    # case where we have a single transport model, we are in fact scaling
    # chi_min, rather than chi_turb.
    # In the case where we have a CombinedTransportModel with a pedestal
    # transport model specified, we are correctly scaling the coefficients from
    # the pedestal transport model.
    # We will tackle the unintended behavior in the single transport model case
    # when we merge this logic with transport model masking and combining.
    # The impact of scaling chi_min is actually not too bad at the moment, but
    # we should still address it.
    transport_multiplier_mapping = {
        # Main terms
        'chi_face_ion': self.chi_i_multiplier,
        'chi_face_el': self.chi_e_multiplier,
        'd_face_el': self.D_e_multiplier,
        'v_face_el': self.v_e_multiplier,
        'chi_face_ion_pereverzev': self.chi_i_multiplier,
        'chi_face_el_pereverzev': self.chi_e_multiplier,
        'full_v_heat_face_ion_pereverzev': self.v_e_multiplier,
        'full_v_heat_face_el_pereverzev': self.v_e_multiplier,
        'd_face_el_pereverzev': self.D_e_multiplier,
        'v_face_el_pereverzev': self.v_e_multiplier,
        # Additional terms
        'chi_face_ion_bohm': self.chi_i_multiplier,
        'chi_face_ion_gyrobohm': self.chi_i_multiplier,
        'chi_face_ion_itg': self.chi_i_multiplier,
        'chi_face_ion_tem': self.chi_i_multiplier,
        'chi_face_el_bohm': self.chi_e_multiplier,
        'chi_face_el_gyrobohm': self.chi_e_multiplier,
        'chi_face_el_itg': self.chi_e_multiplier,
        'chi_face_el_etg': self.chi_e_multiplier,
        'chi_face_el_tem': self.chi_e_multiplier,
        'd_face_el_itg': self.D_e_multiplier,
        'd_face_el_tem': self.D_e_multiplier,
        'v_face_el_itg': self.v_e_multiplier,
        'v_face_el_tem': self.v_e_multiplier,
    }
    pedestal_mask_face = (
        jnp.arange(len(geo.rho_face)) >= self.rho_norm_ped_top_nearest_face_idx
    )
    return jax.tree.map(
        lambda coeff, multiplier: jnp.where(
            pedestal_mask_face, coeff * multiplier, coeff
        ),
        core_transport,
        transport_multiplier_mapping,
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
    # If set_pedestal is False, we return a dummy output with no pedestal.
    # If so, we set the rho of pedestal top to infinite and the index to outside
    # of bounds of the mesh to indicate that the pedestal is not present.
    match runtime_params.pedestal.mode:
      case pedestal_runtime_params_lib.Mode.ADAPTIVE_SOURCE:
        # Source values will not be used, so they can be anything.
        output_if_set_pedestal_false = AdaptiveSourcePedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            rho_norm_ped_top_nearest_cell_idx=geo.torax_mesh.nx,
            rho_norm_ped_top_nearest_face_idx=geo.torax_mesh.nx + 1,
            T_i_ped=0.0,
            T_e_ped=0.0,
            n_e_ped=0.0,
        )
      case pedestal_runtime_params_lib.Mode.ADAPTIVE_TRANSPORT:
        # Set the multipliers to 1.0 to indicate that the transport coefficients
        # are not modified.
        output_if_set_pedestal_false = AdaptiveTransportPedestalModelOutput(
            rho_norm_ped_top=jnp.inf,
            rho_norm_ped_top_nearest_cell_idx=geo.torax_mesh.nx,
            rho_norm_ped_top_nearest_face_idx=geo.torax_mesh.nx + 1,
            chi_e_multiplier=1.0,
            chi_i_multiplier=1.0,
            D_e_multiplier=1.0,
            v_e_multiplier=1.0,
        )
      case _:
        raise ValueError(
            f'Unsupported pedestal model mode: {runtime_params.pedestal.mode}'
        )

    return jax.lax.cond(
        runtime_params.pedestal.set_pedestal,
        lambda: self._call_implementation(runtime_params, geo, core_profiles),
        lambda: output_if_set_pedestal_false,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> PedestalModelOutput:
    """Calculate the pedestal properties."""
