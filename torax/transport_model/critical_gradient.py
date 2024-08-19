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
from torax import constants as constants_module
from torax import geometry
from torax import interpolated_param
from torax import jax_utils
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
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
  chiei_ratio: runtime_params_lib.TimeInterpolated = 2.0
  # Ratio of electron particle to ion heat transport coefficient
  chi_D_ratio: runtime_params_lib.TimeInterpolated = 5.0
  # Ratio of major radius * electron particle convection, to electron diffusion.
  # Sets the value of electron particle convection in the model.
  VR_D_ratio: runtime_params_lib.TimeInterpolated = 0.0

  def make_provider(
      self, torax_mesh: geometry.Grid1D | None = None
  ) -> RuntimeParamsProvider:
    # TODO(b/360831279)
    return RuntimeParamsProvider(
        runtime_params_config=self,
        apply_inner_patch=config_args.get_interpolated_var_single_axis(
            self.apply_inner_patch
        ),
        De_inner=config_args.get_interpolated_var_single_axis(self.De_inner),
        Ve_inner=config_args.get_interpolated_var_single_axis(self.Ve_inner),
        chii_inner=config_args.get_interpolated_var_single_axis(
            self.chii_inner
        ),
        chie_inner=config_args.get_interpolated_var_single_axis(
            self.chie_inner
        ),
        rho_inner=config_args.get_interpolated_var_single_axis(self.rho_inner),
        apply_outer_patch=config_args.get_interpolated_var_single_axis(
            self.apply_outer_patch
        ),
        De_outer=config_args.get_interpolated_var_single_axis(self.De_outer),
        Ve_outer=config_args.get_interpolated_var_single_axis(self.Ve_outer),
        chii_outer=config_args.get_interpolated_var_single_axis(
            self.chii_outer
        ),
        chie_outer=config_args.get_interpolated_var_single_axis(
            self.chie_outer
        ),
        rho_outer=config_args.get_interpolated_var_single_axis(self.rho_outer),
        chiei_ratio=config_args.get_interpolated_var_single_axis(
            self.chiei_ratio
        ),
        chi_D_ratio=config_args.get_interpolated_var_single_axis(
            self.chi_D_ratio
        ),
        VR_D_ratio=config_args.get_interpolated_var_single_axis(
            self.VR_D_ratio
        ),
    )


@chex.dataclass
class RuntimeParamsProvider(runtime_params_lib.RuntimeParamsProvider):
  """Provides a RuntimeParams to use during time t of the sim."""

  runtime_params_config: RuntimeParams
  chiei_ratio: interpolated_param.InterpolatedVarSingleAxis
  chi_D_ratio: interpolated_param.InterpolatedVarSingleAxis
  VR_D_ratio: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        chimin=self.runtime_params_config.chimin,
        chimax=self.runtime_params_config.chimax,
        Demin=self.runtime_params_config.Demin,
        Demax=self.runtime_params_config.Demax,
        Vemin=self.runtime_params_config.Vemin,
        Vemax=self.runtime_params_config.Vemax,
        apply_inner_patch=bool(self.apply_inner_patch.get_value(t)),
        De_inner=float(self.De_inner.get_value(t)),
        Ve_inner=float(self.Ve_inner.get_value(t)),
        chii_inner=float(self.chii_inner.get_value(t)),
        chie_inner=float(self.chie_inner.get_value(t)),
        rho_inner=float(self.rho_inner.get_value(t)),
        apply_outer_patch=bool(self.apply_outer_patch.get_value(t)),
        De_outer=float(self.De_outer.get_value(t)),
        Ve_outer=float(self.Ve_outer.get_value(t)),
        chii_outer=float(self.chii_outer.get_value(t)),
        chie_outer=float(self.chie_outer.get_value(t)),
        rho_outer=float(self.rho_outer.get_value(t)),
        smoothing_sigma=self.runtime_params_config.smoothing_sigma,
        smooth_everywhere=self.runtime_params_config.smooth_everywhere,
        chiei_ratio=float(self.chiei_ratio.get_value(t)),
        chi_D_ratio=float(self.chi_D_ratio.get_value(t)),
        VR_D_ratio=float(self.VR_D_ratio.get_value(t)),
        alpha=self.runtime_params_config.alpha,
        chistiff=self.runtime_params_config.chistiff,
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  """Dynamic runtime params for the CGM transport model."""

  alpha: float
  chistiff: float
  chiei_ratio: float
  chi_D_ratio: float
  VR_D_ratio: float

  def sanity_check(self):
    runtime_params_lib.DynamicRuntimeParams.sanity_check(self)
    # Using the object.__setattr__ call to get around the fact that this
    # dataclass is frozen.
    object.__setattr__(
        self,
        'chi_D_ratio',
        jax_utils.error_if_negative(self.chi_D_ratio, 'chi_D_ratio'),
    )

  def __post_init__(self):
    self.sanity_check()


class CriticalGradientModel(transport_model.TransportModel):
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
  ) -> state.CoreTransport:
    """Calculates transport coefficients using the Critical Gradient Model.

    Args:
      dynamic_runtime_params_slice: Input runtime parameters that can change
        without triggering a JAX recompilation.
      geo: Geometry of the torus.
      core_profiles: Core plasma profiles.

    Returns:
      coeffs: The transport coefficients
    """

    # Many variables throughout this function are capitalized based on physics
    # notational conventions rather than on Google Python style
    # pylint: disable=invalid-name

    # ITG critical gradient model. R/LTi_crit from Guo Romanelli 1993
    # chi_i = chiGB * chistiff * H(R/LTi -
    #  R/LTi_crit)*(R/LTi - R/LTi_crit)^alpha

    constants = constants_module.CONSTANTS
    assert isinstance(
        dynamic_runtime_params_slice.transport, DynamicRuntimeParams
    )

    # set typical values for now. Will include user-defined q and s later
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
        (dynamic_runtime_params_slice.plasma_composition.Ai * constants.mp)
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
    return hash(('CriticalGradientModel'))

  def __eq__(self, other):
    return isinstance(other, CriticalGradientModel)


def _default_cgm_builder() -> CriticalGradientModel:
  return CriticalGradientModel()


@dataclasses.dataclass(kw_only=True)
class CriticalGradientModelBuilder(transport_model.TransportModelBuilder):
  """Builds a class CriticialGradientTransportModel."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  builder: Callable[
      [],
      CriticalGradientModel,
  ] = _default_cgm_builder

  def __call__(
      self,
  ) -> CriticalGradientModel:
    return self.builder()
