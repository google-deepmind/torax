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
from typing import ClassVar, Literal

import chex
import jax
import jax.numpy as jnp
from torax import constants
from torax import state
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.fvm import convection_terms
from torax.fvm import diffusion_terms
from torax.geometry import geometry
from torax.physics import psi_calculations
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_operations
from torax.sources import source_profiles as source_profiles_lib


def calculate_psidot_from_psi_sources(
    *,
    source_profiles: source_profiles_lib.SourceProfiles,
    resistivity_multiplier: float,
    psi: cell_variable.CellVariable,
    geo: geometry.Geometry,
) -> jax.Array:
  """Calculates psidot (loop voltage) from precalculated sources."""
  psi_sources = source_operations.sum_sources_psi(geo, source_profiles)
  sigma = source_profiles.j_bootstrap.sigma
  sigma_face = source_profiles.j_bootstrap.sigma_face

  # Calculate transient term
  consts = constants.CONSTANTS
  toc_psi = (
      1.0
      / resistivity_multiplier
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
      d_face_psi, psi
  )

  # Set the psi convection term for psidot used in ohmic power, always with
  # the default 'ghost' mode. Impact of different modes would mildly impact
  # Ohmic power at the LCFS which has negligible impact on simulations.
  # Allowing it to be configurable introduces more complexity in the code by
  # needing to pass in the mode from the static_runtime_params across multiple
  # functions.
  conv_mat, conv_vec = convection_terms.make_convection_terms(
      v_face_psi, d_face_psi, psi
  )

  c_mat = diffusion_mat + conv_mat
  c = diffusion_vec + conv_vec

  c += psi_sources

  psidot = (jnp.dot(c_mat, psi.value) + c) / toc_psi

  return psidot


def ohmic_model_func(
    unused_static_runtime_params_slice: runtime_params_slice.StaticRuntimeParamsSlice,
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    geo: geometry.Geometry,
    unused_source_name: str,
    core_profiles: state.CoreProfiles,
    calculated_source_profiles: source_profiles_lib.SourceProfiles | None,
) -> tuple[chex.Array, ...]:
  """Returns the Ohmic source for electron heat equation."""
  if calculated_source_profiles is None:
    raise ValueError(
        'calculated_source_profiles is a required argument for'
        ' ohmic_model_func. This can occur if this source function is used in'
        ' an explicit source.'
    )

  jtot, _, _ = psi_calculations.calc_jtot(
      geo,
      core_profiles.psi,
  )
  psidot = calculate_psidot_from_psi_sources(
      source_profiles=calculated_source_profiles,
      resistivity_multiplier=dynamic_runtime_params_slice.numerics.resistivity_mult,
      psi=core_profiles.psi,
      geo=geo,
  )
  pohm = jtot * psidot / (2 * jnp.pi * geo.Rmaj)
  return (pohm,)


class OhmicHeatSourceConfig(runtime_params_lib.SourceModelBase):
  """Configuration for the OhmicHeatSource."""
  source_name: Literal['ohmic_heat_source'] = 'ohmic_heat_source'
  mode: runtime_params_lib.Mode = runtime_params_lib.Mode.MODEL_BASED


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

  @property
  def source_name(self) -> str:
    return self.SOURCE_NAME

  @property
  def affected_core_profiles(
      self,
  ) -> tuple[source_lib.AffectedCoreProfile, ...]:
    return (source_lib.AffectedCoreProfile.TEMP_EL,)
