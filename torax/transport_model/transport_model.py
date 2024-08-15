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
from torax import constants
from torax import geometry
from torax import state
from torax.config import runtime_params_slice
from torax.transport_model import runtime_params as runtime_params_lib


# TODO(b/335593209) all clipping of of transport coefficients to minimum or
# maximum values, along with setting minimum chi and D and V=0 for r>Ped_top,
# can be done via a TransportModel class method instead of within each child
# class. Note: this will slightly change some of the reference results, e.g.
# there is presently no pedestal region clipping for the constant chi model.
class TransportModel(abc.ABC):
  """Calculates various coefficients related to heat and particle transport.

  Subclass responsbilities:
  - Must implement __hash__, __eq__, and be immutable, so that the class can
    be used as a static argument (or a subcomponent of a larger static
    argument) to jax.jit
  - __hash__ and __eq__ must be invariant to copy-construction (e.g., no
    hashing by `id`) so that the jax persistent cache will work between
    runs of the python interpreter
  - __hash__ must be a function of `torax.versioning.torax_hash` so that the
    jax persistent cache will work correctly.
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
  ) -> state.CoreTransport:
    if not getattr(self, "_frozen", False):
      raise RuntimeError(
          f"Subclass implementation {type(self)} forgot to "
          "freeze at the end of __init__."
      )

    # Calculate the transport coefficients
    transport_coeffs = self._call_implementation(
        dynamic_runtime_params_slice, geo, core_profiles
    )

    # Apply min/max clipping and pedestal region clipping
    transport_coeffs = self.apply_clipping(
        dynamic_runtime_params_slice,
        geo,
        transport_coeffs,
    )

    # Return smoothed coefficients if smoothing is enabled
    return self.smooth_coeffs(
        geo,
        dynamic_runtime_params_slice,
        transport_coeffs,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    pass

  def apply_clipping(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      transport_coeffs: state.CoreTransport,
  ) -> state.CoreTransport:
    """Applies min/max and pedestal region clipping to transport coefficients."""

    # set minimum and maximum transport coefficents for PDE stability
    chi_face_ion = jnp.clip(
        transport_coeffs.chi_face_ion,
        dynamic_runtime_params_slice.transport.chimin,
        dynamic_runtime_params_slice.transport.chimax,
    )

    # set minimum and maximum chi for PDE stability
    chi_face_el = jnp.clip(
        transport_coeffs.chi_face_el,
        dynamic_runtime_params_slice.transport.chimin,
        dynamic_runtime_params_slice.transport.chimax,
    )

    d_face_el = jnp.clip(
        transport_coeffs.d_face_el,
        dynamic_runtime_params_slice.transport.Demin,
        dynamic_runtime_params_slice.transport.Demax,
    )
    v_face_el = jnp.clip(
        transport_coeffs.v_face_el,
        dynamic_runtime_params_slice.transport.Vemin,
        dynamic_runtime_params_slice.transport.Vemax,
    )

    # set low transport in pedestal region to facilitate PDE solver
    # (more consistency between desired profile and transport coefficients)
    # if runtime_params.profile_conditions.set_pedestal:
    mask = (
        geo.rho_face_norm
        >= dynamic_runtime_params_slice.profile_conditions.Ped_top
    )
    chi_face_ion = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal, mask
        ),
        dynamic_runtime_params_slice.transport.chimin,
        chi_face_ion,
    )
    chi_face_el = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal, mask
        ),
        dynamic_runtime_params_slice.transport.chimin,
        chi_face_el,
    )
    d_face_el = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal, mask
        ),
        dynamic_runtime_params_slice.transport.Demin,
        d_face_el,
    )
    v_face_el = jnp.where(
        jnp.logical_and(
            dynamic_runtime_params_slice.profile_conditions.set_pedestal, mask
        ),
        0.0,
        v_face_el,
    )

    return dataclasses.replace(
        transport_coeffs,
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def smooth_coeffs(
      self,
      geo: geometry.Geometry,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      transport_coeffs: state.CoreTransport,
  ) -> state.CoreTransport:
    """Gaussian smoothing of turbulent transport coefficients."""
    smoothing_matrix = build_smoothing_matrix(geo, dynamic_runtime_params_slice)
    smoothed_coeffs = {}
    for coeff in transport_coeffs:
      smoothed_coeff = jnp.dot(smoothing_matrix, transport_coeffs[coeff])
      smoothed_coeffs[coeff] = smoothed_coeff
    return state.CoreTransport(**smoothed_coeffs)


def build_smoothing_matrix(
    geo: geometry.Geometry,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
) -> jax.Array:
  """Builds a smoothing matrix for the turbulent transport model.

  Uses a Gaussian kernel of HWHM defined in the transport config.

  Args:
    geo: Geometry of the torus.
    dynamic_runtime_params_slice: Input runtime parameters that can change
      without triggering a JAX recompilation.

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
      / (dynamic_runtime_params_slice.transport.smoothing_sigma**2 + consts.eps)
  )

  # 2. Masking: we do not want transport coefficients calculated in pedestal
  # region or in inner and outer transport patch regions, to impact
  # transport_model calculated coefficients
  mask_outer_edge_ped = jax.lax.cond(
      dynamic_runtime_params_slice.profile_conditions.set_pedestal,
      lambda: dynamic_runtime_params_slice.profile_conditions.Ped_top
      - consts.eps,
      lambda: 1.0,
  )

  mask_outer_edge = jax.lax.cond(
      jnp.logical_and(
          jnp.logical_not(
              dynamic_runtime_params_slice.profile_conditions.set_pedestal
          ),
          dynamic_runtime_params_slice.transport.apply_outer_patch,
      ),
      lambda: dynamic_runtime_params_slice.transport.rho_outer - consts.eps,
      lambda: mask_outer_edge_ped,
  )

  mask_inner_edge = jax.lax.cond(
      dynamic_runtime_params_slice.transport.apply_inner_patch,
      lambda: dynamic_runtime_params_slice.transport.rho_inner + consts.eps,
      lambda: 0.0,
  )

  mask = jnp.where(
      jnp.logical_or(
          dynamic_runtime_params_slice.transport.smooth_everywhere,
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


@dataclasses.dataclass(kw_only=True)
class TransportModelBuilder(abc.ABC):
  """Factory for Stepper objects."""

  @abc.abstractmethod
  def __call__(
      self,
  ) -> TransportModel:
    """Builds a TransportModel instance."""

  # Input parameters to the TransportModel built by this class.
  runtime_params: runtime_params_lib.RuntimeParams = dataclasses.field(
      default_factory=runtime_params_lib.RuntimeParams
  )
