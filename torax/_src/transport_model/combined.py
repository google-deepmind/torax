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
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  transport_model_params: Sequence[runtime_params_lib.DynamicRuntimeParams]
  pedestal_transport_model_params: Sequence[
      runtime_params_lib.DynamicRuntimeParams
  ]


class CombinedTransportModel(transport_model_lib.TransportModel):
  """Combines coefficients from a list of transport models."""

  def __init__(
      self,
      transport_models: Sequence[transport_model_lib.TransportModel],
      pedestal_transport_models: Sequence[transport_model_lib.TransportModel],
  ):
    super().__init__()
    self.transport_models = transport_models
    self.pedestal_transport_models = pedestal_transport_models
    self._frozen = True

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    if not getattr(self, "_frozen", False):
      raise RuntimeError(
          f"Subclass implementation {type(self)} forgot to "
          "freeze at the end of __init__."
      )

    transport_runtime_params = dynamic_runtime_params_slice.transport

    # Calculate the transport coefficients - includes contribution from pedestal
    # and core transport models.
    transport_coeffs = self._call_implementation(
        transport_runtime_params,
        dynamic_runtime_params_slice,
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
        dynamic_runtime_params_slice,
        geo,
        transport_coeffs,
        pedestal_model_output,
    )

  def _call_implementation(
      self,
      transport_dynamic_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    r"""Calculates transport coefficients using the Combined model.

    Args:
      transport_dynamic_runtime_params: Input runtime parameters for this
        transport model. Can change without triggering a JAX recompilation.
      dynamic_runtime_params_slice: Input runtime parameters for all components
        of the simulation that can change without triggering a JAX
        recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_output: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """
    # Required for pytype
    assert isinstance(transport_dynamic_runtime_params, DynamicRuntimeParams)

    def apply_and_restrict(
        component_model: transport_model_lib.TransportModel,
        component_params: runtime_params_lib.DynamicRuntimeParams,
        restriction_fn: Callable[
            [
                runtime_params_lib.DynamicRuntimeParams,
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
          dynamic_runtime_params_slice,
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
            transport_dynamic_runtime_params.pedestal_transport_model_params,
        )
    ]

    core_coeffs = [
        apply_and_restrict(model, params, model._apply_domain_restriction)
        for model, params in zip(
            self.transport_models,
            transport_dynamic_runtime_params.transport_model_params,
        )
    ]

    # Combine the transport coefficients from core and pedestal models.
    combined_transport_coeffs = jax.tree.map(
        lambda *leaves: sum(leaves),
        *pedestal_coeffs,
        *core_coeffs,
    )

    return combined_transport_coeffs

  def _apply_pedestal_domain_restriction(
      self,
      unused_transport_runtime_params: runtime_params_lib.DynamicRuntimeParams,
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

  def __hash__(self):
    return hash(
        tuple(self.transport_models) + tuple(self.pedestal_transport_models)
    )

  def __eq__(self, other):
    return (
        isinstance(other, CombinedTransportModel)
        and self.transport_models == other.transport_models
        and self.pedestal_transport_models == other.pedestal_transport_models
    )
