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
"""Sauter model for bootstrap current."""
import dataclasses
from typing import Annotated, Literal

import chex
import jax
import jax.numpy as jnp
from torax._src import jax_utils
from torax._src import state
from torax._src.config import runtime_params_slice
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import formulas
from torax._src.neoclassical.bootstrap_current import base
from torax._src.neoclassical.bootstrap_current import runtime_params
from torax._src.physics import collisions
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicRuntimeParams(runtime_params.DynamicRuntimeParams):
  """Dynamic runtime params for the Sauter model."""

  bootstrap_multiplier: float


class SauterModel(base.BootstrapCurrentModel):
  """Sauter model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> base.BootstrapCurrent:
    """Calculates bootstrap current."""
    bootstrap_params = (
        dynamic_runtime_params_slice.neoclassical.bootstrap_current
    )
    assert isinstance(bootstrap_params, DynamicRuntimeParams)
    result = _calculate_bootstrap_current(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        Z_eff_face=core_profiles.Z_eff_face,
        Z_i_face=core_profiles.Z_i_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        psi=core_profiles.psi,
        q_face=core_profiles.q_face,
        geo=geometry,
    )
    return base.BootstrapCurrent(
        j_bootstrap=result.j_bootstrap,
        j_bootstrap_face=result.j_bootstrap_face,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(base.BootstrapCurrentModelConfig):
  """Config for the Sauter model implementation of bootstrap current.

  Attributes:
    bootstrap_multiplier: Multiplication factor for bootstrap current.
  """

  model_name: Annotated[Literal['sauter'], torax_pydantic.JAX_STATIC] = 'sauter'
  bootstrap_multiplier: float = 1.0

  def build_dynamic_params(self) -> DynamicRuntimeParams:
    return DynamicRuntimeParams(bootstrap_multiplier=self.bootstrap_multiplier)

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax_utils.jit
def _calculate_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    Z_eff_face: chex.Array,
    Z_i_face: chex.Array,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    q_face: chex.Array,
    geo: geometry_lib.Geometry,
) -> base.BootstrapCurrent:
  """Calculates j_bootstrap and j_bootstrap_face using the Sauter model."""
  # pylint: disable=invalid-name

  # Formulas from Sauter PoP 1999. Future work can include Redl PoP 2021
  # corrections.

  # Effective trapped particle fraction
  f_trap = formulas.calculate_f_trap(geo)

  # Spitzer conductivity
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), Z_i_face
  )
  nu_e_star = formulas.calculate_nu_e_star(
      q=q_face,
      geo=geo,
      n_e=n_e.face_value(),
      T_e=T_e.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = formulas.calculate_nu_i_star(
      q=q_face,
      geo=geo,
      n_i=n_i.face_value(),
      T_i=T_i.face_value(),
      Z_eff=Z_eff_face,
      log_lambda_ii=log_lambda_ii,
  )

  # Calculate terms needed for bootstrap current
  L31 = formulas.calculate_L31(f_trap, nu_e_star, Z_eff_face)
  L32 = formulas.calculate_L32(f_trap, nu_e_star, Z_eff_face)
  L34 = _calculate_L34(f_trap, nu_e_star, Z_eff_face)
  alpha = _calculate_alpha(f_trap, nu_i_star)

  # calculate bootstrap current
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0

  pe = n_e.face_value() * T_e.face_value() * 1e3 * 1.6e-19
  pi = n_i.face_value() * T_i.face_value() * 1e3 * 1.6e-19

  dpsi_drnorm = psi.face_grad()
  dlnne_drnorm = n_e.face_grad() / n_e.face_value()
  dlnni_drnorm = n_i.face_grad() / n_i.face_value()
  dlnte_drnorm = T_e.face_grad() / T_e.face_value()
  dlnti_drnorm = T_i.face_grad() / T_i.face_value()

  global_coeff = prefactor[1:] / dpsi_drnorm[1:]
  global_coeff = jnp.concatenate([jnp.zeros(1), global_coeff])

  necoeff = L31 * pe
  nicoeff = L31 * pi
  tecoeff = (L31 + L32) * pe
  ticoeff = (L31 + alpha * L34) * pi

  j_bootstrap_face = global_coeff * (
      necoeff * dlnne_drnorm
      + nicoeff * dlnni_drnorm
      + tecoeff * dlnte_drnorm
      + ticoeff * dlnti_drnorm
  )
  j_bootstrap = geometry_lib.face_to_cell(j_bootstrap_face)

  return base.BootstrapCurrent(
      j_bootstrap=j_bootstrap,
      j_bootstrap_face=j_bootstrap_face,
  )


def _calculate_L34(
    f_trap: chex.Array,
    nu_e_star: chex.Array,
    Z_eff: chex.Array,
) -> chex.Array:
  """Calculates the L34 coefficient: Sauter PoP 1999 Eqs. 16a+b."""
  ft34 = f_trap / (
      1.0
      + (1 - 0.1 * f_trap) * jnp.sqrt(nu_e_star)
      + 0.5 * (1.0 - 0.5 * f_trap) * nu_e_star / Z_eff
  )
  return (
      (1 + 1.4 / (Z_eff + 1)) * ft34
      - 1.9 / (Z_eff + 1) * ft34**2
      + 0.3 / (Z_eff + 1) * ft34**3
      + 0.2 / (Z_eff + 1) * ft34**4
  )


def _calculate_alpha(
    f_trap: chex.Array,
    nu_i_star: chex.Array,
) -> chex.Array:
  """Calculates the alpha coefficient: Sauter PoP 1999 Eqs. 17a+b."""
  alpha0 = -1.17 * (1 - f_trap) / (1 - 0.22 * f_trap - 0.19 * f_trap**2)
  alpha = (
      (alpha0 + 0.25 * (1 - f_trap**2) * jnp.sqrt(nu_i_star))
      / (1 + 0.5 * jnp.sqrt(nu_i_star))
      + 0.315 * nu_i_star**2 * f_trap**6
  ) / (1 + 0.15 * nu_i_star**2 * f_trap**6)
  return alpha
