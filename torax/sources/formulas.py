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

"""Prescribed formulas for computing source profiles."""

import dataclasses
import jax
from jax import numpy as jnp
from torax import geometry
from torax import jax_utils
from torax import state
from torax.config import runtime_params_slice
from torax.sources import formula_config
from torax.sources import runtime_params


# Many variables throughout this function are capitalized based on physics
# notational conventions rather than on Google Python style
# pylint: disable=invalid-name


def exponential_profile(
    geo: geometry.Geometry,
    c1: float,
    c2: float,
    total: float,
    use_normalized_r: bool = False,
) -> jnp.ndarray:
  """Returns an exponential profile on the cell grid.

  The profile is parameterized by (c1, c2, c3) like so:

    | cell = exp(-(c1 - r) / c2)
    | face = exp(-(c1 - r_face) / c2)
    | C = total / trapz(vpr_face * face, r_face)
    | profile = C * cell

  The formula can use the normalized r and r_face if specified.

  Args:
    geo: Geometry constants of torus.
    c1: Constant. See description above.
    c2: Constant. See description above.
    total: Constant. See description above.
    use_normalized_r: If True, uses r_norm and r_face_norm to calculate the
      profile.

  Returns:
    Exponential profile on the cell grid.
  """
  r = jax_utils.select(use_normalized_r, geo.r_norm, geo.r)
  r_face = jax_utils.select(use_normalized_r, geo.r_face_norm, geo.r_face)
  S = jnp.exp(-(c1 - r) / c2)
  S_face = jnp.exp(-(c1 - r_face) / c2)
  # calculate constant prefactor
  C = total / jax.scipy.integrate.trapezoid(geo.vpr_face * S_face, geo.r_face)
  return C * S


def gaussian_profile(
    geo: geometry.Geometry,
    c1: float,
    c2: float,
    total: float,
    use_normalized_r: bool = False,
) -> jnp.ndarray:
  """Returns a gaussian profile on the cell grid.

  The profile is parameterized by (c1, c2, c3) like so:

    | cell = exp(-( (r - c1)**2 / (2 * c2**2) ))
    | face = exp(-( (r_face - c1)**2 / (2 * c2**2) ))
    | C = total / trazp(vpr_face * face, r_face)
    | profile = C * cell

  The formula can use the normalized r and r_face if specified.

  Args:
    geo: Geometry constants of torus.
    c1: Constant. See description above.
    c2: Constant. See description above.
    total: Constant. See description above.
    use_normalized_r: If True, uses r_norm and r_face_norm to calculate the
      profile.

  Returns:
    Gaussian profile on the cell grid.
  """
  r = jax_utils.select(use_normalized_r, geo.r_norm, geo.r)
  r_face = jax_utils.select(use_normalized_r, geo.r_face_norm, geo.r_face)
  S = jnp.exp(-((r - c1) ** 2) / (2 * c2**2))
  S_face = jnp.exp(-((r_face - c1) ** 2) / (2 * c2**2))
  # calculate constant prefactor
  C = total / jax.scipy.integrate.trapezoid(geo.vpr_face * S_face, geo.r_face)
  return C * S


# pylint: enable=invalid-name


# Callable classes used as arguments for Source formulas.


@dataclasses.dataclass(frozen=True)
class Exponential:
  """Callable class providing an exponential profile."""

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_source_runtime_params: runtime_params.DynamicRuntimeParams,
      geo: geometry.Geometry,
      unused_state: state.CoreProfiles | None,
  ) -> jnp.ndarray:
    exp_config = dynamic_source_runtime_params.formula
    assert isinstance(exp_config, formula_config.DynamicExponential)
    return exponential_profile(
        c1=exp_config.c1,
        c2=exp_config.c2,
        total=exp_config.total,
        geo=geo,
        use_normalized_r=exp_config.use_normalized_r,
    )


@dataclasses.dataclass(frozen=True)
class Gaussian:
  """Callable class providing a gaussian profile."""

  def __call__(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      dynamic_source_runtime_params: runtime_params.DynamicRuntimeParams,
      geo: geometry.Geometry,
      unused_state: state.CoreProfiles | None,
  ) -> jnp.ndarray:
    gaussian_config = dynamic_source_runtime_params.formula
    assert isinstance(gaussian_config, formula_config.DynamicGaussian)
    return gaussian_profile(
        c1=gaussian_config.c1,
        c2=gaussian_config.c2,
        total=gaussian_config.total,
        geo=geo,
        use_normalized_r=gaussian_config.use_normalized_r,
    )
