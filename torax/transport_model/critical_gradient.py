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

"""The CriticalGradientModel class."""

from __future__ import annotations

import dataclasses
from typing import Callable

import chex
from jax import numpy as jnp
from torax import array_typing
from torax import constants as constants_module
from torax import interpolated_param
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.geometry import geometry
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.torax_pydantic import torax_pydantic
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model


# pylint: disable=invalid-name
@chex.dataclass
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Extends the base runtime params with additional params for this model.

  See base class runtime_params.RuntimeParams docstring for more info.
  """

  # Exponent of chi power law: chi \propto (R/LTi - R/LTi_crit)^alpha
  alpha: float = 2.0
  # Stiffness parameter
  chistiff: float = 2.0
  # Ratio of electron to ion heat transport coefficient (ion higher for ITG)
  chiei_ratio: runtime_params_lib.TimeInterpolatedInput = 2.0
  # Ratio of electron particle to ion heat transport coefficient
  chi_D_ratio: runtime_params_lib.TimeInterpolatedInput = 5.0
  # Ratio of major radius * electron particle convection, to electron diffusion.
  # Sets the value of electron particle convection in the model.
  VR_D_ratio: runtime_params_lib.TimeInterpolatedInput = 0.0

  def make_provider(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    # TODO(b/360831279)
    return RuntimeParamsProvider(**self.get_provider_kwargs(torax_mesh))


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams
  chiei_ratio: interpolated_param.InterpolatedVarSingleAxis
  chi_D_ratio: interpolated_param.InterpolatedVarSingleAxis
  VR_D_ratio: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(**self.get_dynamic_params_kwargs(t))


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the CGM transport model."""

  alpha: float
  chistiff: float
  chiei_ratio: array_typing.ScalarFloat
  chi_D_ratio: array_typing.ScalarFloat
  VR_D_ratio: array_typing.ScalarFloat

  def sanity_check(self):
    runtime_params_lib.DynamicRuntimeParams.sanity_check(self)
    jax_utils.error_if_negative(self.chi_D_ratio, 'chi_D_ratio')

  def __post_init__(self):
    self.sanity_check()


class CriticalGradientTransportModel(transport_model.TransportModel):
  """Calculates various coefficients related to particle transport."""

  def __init__(
      self,
  ):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_outputs: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    r"""Calculates transport coefficients using the Critical Gradient Model.

    Uses critical normalized logarithmic ion temperature gradient
    :math:`R/L_{Ti}|_crit` from Guo Romanelli 1993:
    :math:`\chi_i = \chi_{GB} \chi_{stiff} H(R/L_{Ti} - R/L_{Ti})`
    where :math:`\chi_{GB}` is the GyroBohm diffusivity,
    :math:`\chi_{stiff}` is the stiffness parameter, and
    :math:`H` is the Heaviside function.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.
      pedestal_model_outputs: Output of the pedestal model.

    Returns:
      coeffs: The transport coefficients
    """

    # pylint: disable=invalid-name

    constants = constants_module.CONSTANTS
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    s = core_profiles.s_face
    q = core_profiles.q_face

    # very basic sawtooth model
    s = jnp.where(q < 1, 0, s)
    q = jnp.where(q < 1, 1, q)

    # define radial coordinate as midplane average r
    # (typical assumption for transport models developed in circular geo)
    rmid = (geo.Rout - geo.Rin) * 0.5

    temp_ion_face = core_profiles.temp_ion.face_value()
    temp_ion_face_grad = core_profiles.temp_ion.face_grad(rmid)
    temp_el_face = core_profiles.temp_el.face_value()

    # set critical gradient
    rlti_crit = (
        4.0
        / 3.0
        * (1.0 + temp_ion_face / temp_el_face)
        * (1.0 + 2.0 * jnp.abs(s) / q)
    )

    # gyrobohm diffusivity
    chiGB = (
        (
            dynamic_runtime_params_slice.plasma_composition.main_ion.avg_A
            * constants.mp
        )
        ** 0.5
        / (constants.qe * geo.B0) ** 2
        * (temp_ion_face * constants.keV2J) ** 1.5
        / geo.Rmaj
    )

    # R/LTi profile from current timestep temp_ion
    rlti = -geo.Rmaj * temp_ion_face_grad / temp_ion_face

    # build CGM model ion heat transport coefficient
    chi_face_ion = jnp.where(
        rlti >= rlti_crit,
        chiGB
        * dynamic_runtime_params_slice.transport.chistiff
        * (rlti - rlti_crit) ** dynamic_runtime_params_slice.transport.alpha,
        0.0,
    )

    # set electron heat transport coefficient to user-defined ratio of ion heat
    # transport coefficient
    chi_face_el = (
        chi_face_ion / dynamic_runtime_params_slice.transport.chiei_ratio
    )

    # set electron particle transport coefficient to user-defined ratio of ion
    # heat transport coefficient
    d_face_el = (
        chi_face_ion / dynamic_runtime_params_slice.transport.chi_D_ratio
    )

    # User-provided convection coefficient
    v_face_el = (
        d_face_el * dynamic_runtime_params_slice.transport.VR_D_ratio / geo.Rmaj
    )

    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self):
    # All CriticalGradientModels are equivalent and can hash the same
    return hash('CriticalGradientModel')

  def __eq__(self, other):
    return isinstance(other, CriticalGradientTransportModel)


def _default_cgm_builder() -> CriticalGradientTransportModel:
  return CriticalGradientTransportModel()


@dataclasses.dataclass(kw_only=True)
class CriticalGradientModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class CriticialGradientTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  builder: Callable[
      [],
      CriticalGradientTransportModel,
  ] = _default_cgm_builder

  def __call__(
      self,
  ) -> CriticalGradientTransportModel:
    return self.builder()
