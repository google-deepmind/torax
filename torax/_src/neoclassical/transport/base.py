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

"""Base class for neoclassical transport models."""
import abc

import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.transport import runtime_params as transport_runtime_params
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib
import typing_extensions

# pylint: disable=invalid-name


class NeoclassicalTransportModel(abc.ABC):
  """Base class for neoclassical transport models."""

  def __call__(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    """Calculates neoclassical transport and applies clipping."""
    neoclassical_transport = self._call_implementation(
        runtime_params,
        geometry,
        core_profiles,
    )
    neoclassical_transport = self._apply_clipping(
        runtime_params,
        neoclassical_transport,
    )
    return neoclassical_transport

  def _apply_clipping(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      neoclassical_transport: transport_coeffs_lib.NeoclassicalTransport,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    """Applies min/max clipping to neoclassical transport coefficients."""
    chi_face_ion = jnp.clip(
        neoclassical_transport.chi_face_ion,
        runtime_params.neoclassical.transport.chi_min,
        runtime_params.neoclassical.transport.chi_max,
    )
    chi_face_el = jnp.clip(
        neoclassical_transport.chi_face_el,
        runtime_params.neoclassical.transport.chi_min,
        runtime_params.neoclassical.transport.chi_max,
    )
    d_face_el = jnp.clip(
        neoclassical_transport.d_face_el,
        runtime_params.neoclassical.transport.D_e_min,
        runtime_params.neoclassical.transport.D_e_max,
    )
    v_face_el = jnp.clip(
        neoclassical_transport.v_face_el,
        runtime_params.neoclassical.transport.V_e_min,
        runtime_params.neoclassical.transport.V_e_max,
    )
    return transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
        v_face_el_ware=neoclassical_transport.v_face_el_ware,
    )

  @abc.abstractmethod
  def _call_implementation(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> transport_coeffs_lib.NeoclassicalTransport:
    """Computes raw neoclassical transport coefficients.

    Note that the returned NeoclassicalTransport contains v_face_el (the total
    electron particle convection including the Ware pinch) and v_face_el_ware
    (the Ware pinch alone, included for debugging and diagnostics).
    """
    pass


class NeoclassicalTransportModelConfig(torax_pydantic.BaseModelFrozen, abc.ABC):
  """Base class for neoclassical transport model configs.

  Attributes:
   chi_min: Lower bound on heat conductivity.
   chi_max: Upper bound on heat conductivity.
   D_e_min: minimum electron density diffusivity.
   D_e_max: maximum electron density diffusivity.
   V_e_min: minimum electron density convection.
   V_e_max: maximum electron density convection.
  """

  chi_min: torax_pydantic.MeterSquaredPerSecond = 0.0
  chi_max: torax_pydantic.MeterSquaredPerSecond = 100
  D_e_min: torax_pydantic.MeterSquaredPerSecond = 0.0
  D_e_max: torax_pydantic.MeterSquaredPerSecond = 100.0
  V_e_min: torax_pydantic.MeterPerSecond = -50.0
  V_e_max: torax_pydantic.MeterPerSecond = 50.0

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.chi_min < self.chi_max:
      raise ValueError('chi_min must be less than chi_max.')
    if not self.D_e_min < self.D_e_max:
      raise ValueError('D_e_min must be less than D_e_max.')
    if not self.V_e_min < self.V_e_max:
      raise ValueError('V_e_min must be less than V_e_max.')
    return self

  def build_runtime_params(self) -> transport_runtime_params.RuntimeParams:
    return transport_runtime_params.RuntimeParams(
        chi_min=self.chi_min,
        chi_max=self.chi_max,
        D_e_min=self.D_e_min,
        D_e_max=self.D_e_max,
        V_e_min=self.V_e_min,
        V_e_max=self.V_e_max,
    )

  @abc.abstractmethod
  def build_model(self) -> NeoclassicalTransportModel:
    """Builds neoclassical transport model."""
