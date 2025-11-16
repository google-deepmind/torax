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

"""The CombinedTransportModel class.

A class for combining transport models.
"""
import dataclasses
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib

# pylint: disable=protected-access


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  transport_model_params: Sequence[runtime_params_lib.RuntimeParams]
  pedestal_transport_model_params: Sequence[runtime_params_lib.RuntimeParams]


@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportModel(transport_model_lib.TransportModel):
  """Combines coefficients from a tuple of transport models."""

  transport_models: tuple[transport_model_lib.TransportModel, ...]
  pedestal_transport_models: tuple[transport_model_lib.TransportModel, ...]

  def __call__(
      self,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:

    transport_runtime_params = runtime_params.transport

    # Calculate the transport coefficients - includes contribution from pedestal
    # and core transport models.
    transport_coeffs = self._call_implementation(
        transport_runtime_params,
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_output,
    )

    # In contrast to the base TransportModel, we do not apply domain restriction
    # as this is handled at the component model level

    # Apply min/max clipping
    transport_coeffs = self._apply_clipping(
        transport_runtime_params,
        transport_coeffs,
    )

    # In contrast to the base TransportModel, we do not apply patches, as these
    # should be handled by instantiating constant component models instead.
    # However, the rho_inner and rho_outer arguments are currently required
    # in the case where the inner/outer region are to be excluded from
    # smoothing. Smoothing is applied to
    # rho_inner < rho_norm < min(rho_ped_top, rho_outer) unless
    # smooth_everywhere is True.
    return self._smooth_coeffs(
        transport_runtime_params,
        runtime_params,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

  def _call_implementation(
      self,
      transport_runtime_params: runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Combined model.

    Args:
      transport_runtime_params: Input runtime parameters for this
        transport model. Can change without triggering a JAX recompilation.
      runtime_params: Runtime parameters for the simulation at the current time.
      geo: Geometry of the torus at the current time.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    # Required for pytype
    assert isinstance(transport_runtime_params, RuntimeParams)

    def apply_and_restrict(
        component_model: transport_model_lib.TransportModel,
        component_params: runtime_params_lib.RuntimeParams,
        restriction_fn: Callable[
            [
                runtime_params_lib.RuntimeParams,
                geometry.Geometry,
                transport_model_lib.TurbulentTransport,
                pedestal_model_lib.PedestalModelOutput,
            ],
            transport_model_lib.TurbulentTransport,
        ],
    ) -> transport_model_lib.TurbulentTransport:
      # TODO(b/434175682): Consider only computing transport coefficients for
      # the active domain, rather than masking them out later. This could be
      # significantly more efficient especially for pedestal models, as these
      # are only active in a small region of the domain.
      component_transport_coeffs = component_model._call_implementation(
          component_params,
          runtime_params,
          geo,
          core_profiles,
          pedestal_model_output,
      )
      component_transport_coeffs = restriction_fn(
          component_params,
          geo,
          component_transport_coeffs,
          pedestal_model_output,
      )
      return component_transport_coeffs

    pedestal_coeffs = [
        apply_and_restrict(
            model, params, self._apply_pedestal_domain_restriction
        )
        for model, params in zip(
            self.pedestal_transport_models,
            transport_runtime_params.pedestal_transport_model_params,
        )
    ]

    core_coeffs = [
        apply_and_restrict(model, params, model._apply_domain_restriction)
        for model, params in zip(
            self.transport_models,
            transport_runtime_params.transport_model_params,
        )
    ]

    # Combine the transport coefficients from core and pedestal models.
    def _combine_maybe_none_coeffs(*leaves):
      non_none_leaves = [leaf for leaf in leaves if leaf is not None]
      return sum(non_none_leaves) if non_none_leaves else None

    combined_transport_coeffs = jax.tree.map(
        _combine_maybe_none_coeffs,
        *pedestal_coeffs,
        *core_coeffs,
        # Needed to handle the case where some coefficients are None and others
        # are not.
        is_leaf=lambda x: x is None,
    )

    return combined_transport_coeffs

  def _apply_pedestal_domain_restriction(
      self,
      unused_transport_runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      transport_coeffs: transport_model_lib.TurbulentTransport,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    del unused_transport_runtime_params
    active_mask = geo.rho_face_norm > pedestal_model_output.rho_norm_ped_top

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
