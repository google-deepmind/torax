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
"""Ohmic heat source."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from torax import constants
from torax import jax_utils
from torax import physics
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import convection_terms
from torax.fvm import diffusion_terms
from torax.geometry import geometry
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_models as source_models_lib


@functools.partial(
    jax_utils.jit,
    static_argnames=[
        'source_models',
        'static_runtime_params_slice',
    ],
)
def calc_psidot(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  r"""Calculates psidot (loop voltage). Used for the Ohmic electron heat source.

  psidot is an interesting TORAX output, and is thus also saved in
  core_profiles.

  psidot = \partial psi / \partial t, and is derived from the same components
  that form the psi block in the coupled PDE equations. Thus, a similar
  (but abridged) formulation as in sim.calc_coeffs and fvm._calc_c is used here

  Args:
    static_runtime_params_slice: Simulation configuration that does not change
      from timestep to timestep.
    dynamic_runtime_params_slice: Simulation configuration at this timestep
    geo: Torus geometry
    core_profiles: Core plasma profiles.
    source_models: All TORAX source/sinks.

  Returns:
    psidot: on cell grid
  """
  consts = constants.CONSTANTS

  psi_sources, sigma, sigma_face = source_models_lib.calc_and_sum_sources_psi(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )
  # Calculate transient term
  toc_psi = (
      1.0
      / dynamic_runtime_params_slice.numerics.resistivity_mult
      * geo.rho_norm
      * sigma
      * consts.mu0
      * 16
      * jnp.pi**2
      * geo.Phib**2
      / geo.F**2
  )
  # Calculate diffusion term coefficient
  d_face_psi = geo.g2g3_over_rhon_face
  # Add phibdot terms to poloidal flux convection
  v_face_psi = (
      -8.0
      * jnp.pi**2
      * consts.mu0
      * geo.Phibdot
      * geo.Phib
      * sigma_face
      * geo.rho_face_norm**2
      / geo.F_face**2
  )

  # Add effective phibdot poloidal flux source term
  ddrnorm_sigma_rnorm2_over_f2 = jnp.gradient(
      sigma * geo.rho_norm**2 / geo.F**2, geo.rho_norm
  )

  psi_sources += (
      -8.0
      * jnp.pi**2
      * consts.mu0
      * geo.Phibdot
      * geo.Phib
      * ddrnorm_sigma_rnorm2_over_f2
  )

  diffusion_mat, diffusion_vec = diffusion_terms.make_diffusion_terms(
      d_face_psi, core_profiles.psi
  )

  # Set the psi convection term for psidot used in ohmic power, always with
  # the default 'ghost' mode. Impact of different modes would mildly impact
  # Ohmic power at the LCFS which has negligible impact on simulations.
  # Allowing it to be configurable introduces more complexity in the code by
  # needing to pass in the mode from the static_runtime_params across multiple
  # functions.
  conv_mat, conv_vec = convection_terms.make_convection_terms(
      v_face_psi,
      d_face_psi,
      core_profiles.psi,
  )

  c_mat = diffusion_mat + conv_mat
  c = diffusion_vec + conv_vec

  c += psi_sources

  psidot = (jnp.dot(c_mat, core_profiles.psi.value) + c) / toc_psi

  return psidot


def ohmic_model_func(
    static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    source_name: str,
    core_profiles: state.CoreProfiles,
    source_models: source_models_lib.SourceModels,
) -> jax.Array:
  """Returns the Ohmic source for electron heat equation."""
  del source_name  # Unused.
  if source_models is None:
    raise TypeError('source_models is a required argument for ohmic_model_func')

  jtot, _, _ = physics.calc_jtot_from_psi(
      geo,
      core_profiles.psi,
  )

  psidot = calc_psidot(
      static_runtime_params_slice,
      dynamic_runtime_params_slice,
      geo,
      core_profiles,
      source_models,
  )

  pohm = jtot * psidot / (2 * jnp.pi * geo.Rmaj)
  return pohm


@dataclasses.dataclass
class OhmicRuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime params for OhmicHeatSource."""

  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED


# OhmicHeatSource is a special case and defined here to avoid circular
# dependencies, since it depends on the psi sources
@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class OhmicHeatSource(source_lib.Source):
  """Ohmic heat source for electron heat equation.

  Pohm = jtor * psidot /(2*pi*Rmaj), related to electric power formula P = IV.
  """

  SOURCE_NAME: ClassVar[str] = 'ohmic_heat_source'
  DEFAULT_MODEL_FUNCTION_NAME: ClassVar[str] = 'ohmic_model_func'
  model_func: source_lib.SourceProfileFunction = ohmic_model_func
  # Users must pass in a pointer to the complete set of sources to this object.
  source_models: source_models_lib.SourceModels

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (source_lib.AffectedCoreProfile.TEMP_EL,)
