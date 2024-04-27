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

"""Generic heat source for both ion and electron heat."""

from __future__ import annotations

import dataclasses

import chex
import jax
from jax import numpy as jnp
from torax import geometry
from torax import state
from torax.config import config_args
from torax.config import runtime_params_slice
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source


# Many variables throughout this function are capitalized based on physics
# notational conventions rather than on Google Python style
# pylint: disable=invalid-name


@dataclasses.dataclass(kw_only=True)
class RuntimeParams(runtime_params_lib.RuntimeParams):
  """Runtime parameters for the generic heat source."""

  # external heat source parameters
  # Gaussian width in normalized radial coordinate
  w: runtime_params_lib.TimeDependentField = 0.25
  # Source Gaussian central location (in normalized r)
  rsource: runtime_params_lib.TimeDependentField = 0.0
  # total heating
  Ptot: runtime_params_lib.TimeDependentField = 120e6
  # electron heating fraction
  el_heat_fraction: runtime_params_lib.TimeDependentField = 0.66666

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(
        **config_args.get_init_kwargs(
            input_config=self,
            output_type=DynamicRuntimeParams,
            t=t,
        )
    )


@chex.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params_lib.DynamicRuntimeParams):
  w: float
  rsource: float
  Ptot: float
  el_heat_fraction: float


def calc_generic_heat_source(
    geo: geometry.Geometry,
    rsource: float,
    w: float,
    Ptot: float,
    el_heat_fraction: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes ion/electron heat source terms.

  Flexible prescribed heat source term.

  Args:
    geo: Geometry describing the torus.
    rsource: Source Gaussian central location
    w: Gaussian width
    Ptot: total heating
    el_heat_fraction: fraction of heating deposited on electrons

  Returns:
    source_ion: source term for ions.
    source_el: source term for electrons.
  """

  # calculate heat profile (face grid)
  Q = jnp.exp(-((geo.r_norm - rsource) ** 2) / (2 * w**2))
  Q_face = jnp.exp(-((geo.r_face_norm - rsource) ** 2) / (2 * w**2))
  # calculate constant prefactor
  C = Ptot / jax.scipy.integrate.trapezoid(geo.vpr_face * Q_face, geo.r_face)

  source_ion = C * Q * (1 - el_heat_fraction)
  source_el = C * Q * el_heat_fraction

  return source_ion, source_el


def _default_formula(
    dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
    dynamic_source_runtime_params: runtime_params_lib.DynamicRuntimeParams,
    geo: geometry.Geometry,
    core_profiles: state.CoreProfiles,
) -> jnp.ndarray:
  """Returns the default formula-based ion/electron heat source profile."""
  del dynamic_runtime_params_slice, core_profiles  # Unused.
  assert isinstance(dynamic_source_runtime_params, DynamicRuntimeParams)
  ion, el = calc_generic_heat_source(
      geo,
      dynamic_source_runtime_params.rsource,
      dynamic_source_runtime_params.w,
      dynamic_source_runtime_params.Ptot,
      dynamic_source_runtime_params.el_heat_fraction,
  )
  return jnp.stack([ion, el])


# pylint: enable=invalid-name


@dataclasses.dataclass(kw_only=True)
class GenericIonElectronHeatSource(source.IonElectronSource):
  """Generic heat source for both ion and electron heat."""

  runtime_params: RuntimeParams = dataclasses.field(
      default_factory=RuntimeParams
  )

  formula: source.SourceProfileFunction = _default_formula
