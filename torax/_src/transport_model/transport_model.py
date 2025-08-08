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

"""The TransportModel abstract base class.

The transport model calculates heat and particle turbulent transport
coefficients.
"""

import abc
import dataclasses

import jax
from jax import numpy as jnp
from torax._src import constants
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
import typing_extensions


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
  """

  chi_face_ion: jax.Array
  chi_face_el: jax.Array
  d_face_el: jax.Array
  v_face_el: jax.Array
  chi_face_el_bohm: jax.Array | None = None
  chi_face_el_gyrobohm: jax.Array | None = None
  chi_face_ion_bohm: jax.Array | None = None
  chi_face_ion_gyrobohm: jax.Array | None = None

  def __post_init__(self):
    # Use the array size of chi_face_el as a reference.
    if self.chi_face_el_bohm is None:
      self.chi_face_el_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_el_gyrobohm is None:
      self.chi_face_el_gyrobohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_bohm is None:
      self.chi_face_ion_bohm = jnp.zeros_like(self.chi_face_el)
    if self.chi_face_ion_gyrobohm is None:
      self.chi_face_ion_gyrobohm = jnp.zeros_like(self.chi_face_el)

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> typing_extensions.Self:
    """Returns a CoreTransport with all zeros. Useful for initializing."""
    shape = geo.rho_face.shape
    return cls(
        chi_face_ion=jnp.zeros(shape),
        chi_face_el=jnp.zeros(shape),
        d_face_el=jnp.zeros(shape),
        v_face_el=jnp.zeros(shape),
    )


class TransportModel(abc.ABC):
  """Calculates various coefficients related to heat and particle transport.

  Subclass responsbilities:
  - Must implement __hash__, __eq__, and be immutable, so that the class can
    be used as a static argument (or a subcomponent of a larger static
    argument) to jax.jit
  - Must set _frozen = True at the end of the subclass __init__ method to
    activate immutability.
  """

  def __setattr__(self, attr, value):
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """Override __setattr__ to make the class (sort of) immutable.

    Note that you can still do obj.field.subfield = x, so it is not true
    immutability, but this to helps to avoid some careless errors.
    """
    if getattr(self, "_frozen", False):
      raise AttributeError("TransportModels are immutable.")
    return super().__setattr__(attr, value)

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    if not getattr(self, "_frozen", False):
      raise RuntimeError(
          f"Subclass implementation {type(self)} forgot to "
          "freeze at the end of __init__."
      )

    transport_runtime_params = dynamic_runtime_params_slice.transport

    # Calculate the transport coefficients
    transport_coeffs = self._call_implementation(
        transport_runtime_params,
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        pedestal_model_output,
    )

    # Restrict the model to operating in its permissible rho domain
    transport_coeffs = self._apply_domain_restriction(
        transport_runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

    # Apply min/max clipping
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )

    # Apply inner and outer transport patch
    transport_coeffs = self._apply_transport_patches(
        transport_runtime_params,
        dynamic_runtime_params_slice,
        geo,
        transport_coeffs,
    )

    # Return smoothed coefficients if smoothing is enabled
    return self._smooth_coeffs(
        transport_runtime_params,
        dynamic_runtime_params_slice,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      transport_dynamic_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    pass

  @abc.abstractmethod
  def __hash__(self) -> int:
    """Returns a hash of the transport model.

    Should be implemented to support jax.jit caching.
    """

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    """Returns whether the transport model is equal to the other.

    Should be implemented to support jax.jit caching.

    Args:
      other: The object to compare to.
    """

  def _apply_domain_restriction(
      self,
      transport_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    """Sets transport coefficients to zero outside the model's domain."""
    # Standard case: active range is
    # rho_min < rho <= rho_norm_ped_top
    active_mask = (
        (geo.rho_face_norm > transport_runtime_params.rho_min)
        & (geo.rho_face_norm <= transport_runtime_params.rho_max)
        & (geo.rho_face_norm <= pedestal_model_output.rho_norm_ped_top)
    )
    # Special case: if rho_min is 0, active range is
    # rho_min <= rho <= rho_norm_ped_top
    active_mask = (
        jnp.asarray(active_mask)
        .at[0]
        .set(transport_runtime_params.rho_min == 0)
    )

    chi_face_ion = jnp.where(active_mask, transport_coeffs.chi_face_ion, 0.0)
    chi_face_el = jnp.where(active_mask, transport_coeffs.chi_face_el, 0.0)
    d_face_el = jnp.where(active_mask, transport_coeffs.d_face_el, 0.0)
    v_face_el = jnp.where(active_mask, transport_coeffs.v_face_el, 0.0)

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _apply_clipping(
      self,
      transport_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Applies min/max clipping to transport coefficients for PDE stability."""
    chi_face_ion = jnp.clip(
        transport_coeffs.chi_face_ion,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    chi_face_el = jnp.clip(
        transport_coeffs.chi_face_el,
        transport_runtime_params.chi_min,
        transport_runtime_params.chi_max,
    )
    d_face_el = jnp.clip(
        transport_coeffs.d_face_el,
        transport_runtime_params.D_e_min,
        transport_runtime_params.D_e_max,
    )
    v_face_el = jnp.clip(
        transport_coeffs.v_face_el,
        transport_runtime_params.V_e_min,
        transport_runtime_params.V_e_max,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _apply_transport_patches(
      self,
      transport_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
  ) -> TurbulentTransport:
    """Applies inner and outer transport patches to transport coefficients."""
    consts = constants.CONSTANTS

    # Apply inner and outer patch constant transport coefficients. rho_inner and
    # rho_outer are shifted by consts.eps (1e-7) to avoid ambiguities if their
    # values are close to and geo.rho_face_norm values.
    chi_face_ion = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_i_inner,
        transport_coeffs.chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.chi_e_inner,
        transport_coeffs.chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.D_e_inner,
        transport_coeffs.d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            transport_runtime_params.apply_inner_patch,
            geo.rho_face_norm < transport_runtime_params.rho_inner + consts.eps,
        ),
        transport_runtime_params.V_e_inner,
        transport_coeffs.v_face_el,
    )

    # Apply outer patch constant transport coefficients.
    # Due to Pereverzev-Corrigan convection, it is required
    # for the convection modes to be 'ghost' to avoid numerical instability
    chi_face_ion = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    dynamic_runtime_params_slice.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_i_outer,
        chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    dynamic_runtime_params_slice.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.chi_e_outer,
        chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    dynamic_runtime_params_slice.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.D_e_outer,
        d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            jnp.logical_and(
                transport_runtime_params.apply_outer_patch,
                jnp.logical_not(
                    dynamic_runtime_params_slice.pedestal.set_pedestal
                ),
            ),
            geo.rho_face_norm > transport_runtime_params.rho_outer - consts.eps,
        ),
        transport_runtime_params.V_e_outer,
        v_face_el,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def _smooth_coeffs(
      self,
      transport_dynamic_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      transport_coeffs: TurbulentTransport,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> TurbulentTransport:
    """Gaussian smoothing of turbulent transport coefficients."""
    smoothing_matrix = _build_smoothing_matrix(
        transport_dynamic_runtime_params,
        dynamic_runtime_params_slice,
        geo,
        pedestal_model_output,
    )

    # Iterate over fields of the CoreTransport dataclass.
    # Ignore optional fields that are made all zero in post_init.
    def smooth_single_coeff(coeff):
      return jax.lax.cond(
          jnp.all(coeff == 0.0),
          lambda: coeff,
          lambda: jnp.dot(smoothing_matrix, coeff),
      )

    return jax.tree_util.tree_map(smooth_single_coeff, transport_coeffs)


def _build_smoothing_matrix(
    transport_dynamic_runtime_params: transport_runtime_params_lib.DynamicRuntimeParams,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
) -> jax.Array:
  """Builds a smoothing matrix for the turbulent transport model.

  Uses a Gaussian kernel of HWHM defined in the transport config.

  Args:
    transport_dynamic_runtime_params:  Input runtime parameters of this model
      that can change without triggering a JAX recompilation.
    dynamic_runtime_params_slice: Input runtime parameters of the simulation
      that can change without triggering a JAX recompilation.
    geo: Geometry of the torus.
    pedestal_model_output: Output of the pedestal model.

  Returns:
    kernel: A smoothing matrix for convolution with the transport outputs.
  """

  # To reduce the range of the convolution, weights under lower_cutoff are
  # clipped to zero
  lower_cutoff = 0.01

  # used for eps, small number to avoid divisions by zero for sigma = 0
  consts = constants.CONSTANTS

  # 1. Kernel matrix
  kernel = jnp.exp(
      -jnp.log(2)
      * (geo.rho_face_norm[:, jnp.newaxis] - geo.rho_face_norm) ** 2
      / (transport_dynamic_runtime_params.smoothing_width**2 + consts.eps)
  )

  # 2. Masking: we do not want transport coefficients calculated in pedestal
  # region or in inner and outer transport patch regions, to impact
  # transport_model calculated coefficients

  # If set pedestal is False and apply_outer_patch is True, we want to mask
  # according to rho_outer, otherwise we want to mask according to
  # rho_norm_ped_top. In the case where set_pedestal is False this is inf
  # which when we use to make the mask means that we will not mask anything.
  # If set pedestal is True, we want to mask according to rho_norm_ped_top.
  mask_outer_edge = jax.lax.cond(
      jnp.logical_and(
          jnp.logical_not(dynamic_runtime_params_slice.pedestal.set_pedestal),
          transport_dynamic_runtime_params.apply_outer_patch,
      ),
      lambda: transport_dynamic_runtime_params.rho_outer - consts.eps,
      lambda: pedestal_model_output.rho_norm_ped_top - consts.eps,
  )

  mask_inner_edge = jax.lax.cond(
      transport_dynamic_runtime_params.apply_inner_patch,
      lambda: transport_dynamic_runtime_params.rho_inner + consts.eps,
      lambda: 0.0,
  )

  mask = jnp.where(
      jnp.logical_or(
          transport_dynamic_runtime_params.smooth_everywhere,
          jnp.logical_and(
              geo.rho_face_norm > mask_inner_edge,
              geo.rho_face_norm < mask_outer_edge,
          ),
      ),
      1.0,
      0.0,
  )

  # remove impact of smoothing on inner and outer patch, or pedestal zone

  # first zero out all rows corresponding to grid points not to be impacted
  diag_mask = jnp.diag(mask)
  kernel = jnp.dot(diag_mask, kernel)
  # now zero out all columns corresponding to grid points not to be impacted,
  # such that they don't impact the smoothing of the other grid points
  num_rows = len(mask)
  mask_mat = jnp.tile(mask, (num_rows, 1))
  kernel *= mask_mat
  # now restore identity to the zero rows, such that smoothing is a no-op for
  # on the grid points where it shouldn't impact
  zero_row_mask = jnp.all(kernel == 0, axis=1)
  kernel = jnp.where(
      zero_row_mask[:, jnp.newaxis], jnp.eye(kernel.shape[0]), kernel
  )

  # 3. Normalization
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]

  # 4. Remove small numbers
  kernel = jnp.where(kernel < lower_cutoff, 0.0, kernel)

  # 5. Final Normalization following removal of small numbers
  row_sums = jnp.sum(kernel, axis=1)
  kernel /= row_sums[:, jnp.newaxis]
  return kernel
