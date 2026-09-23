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

from typing import Annotated, Literal

import jax
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical import runtime_params as neoclassical_runtime_params
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.neoclassical.bootstrap_current import runtime_params as bootstrap_current_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import sauter as sauter_formulas
from torax._src.torax_pydantic import torax_pydantic


class SauterModel(bootstrap_current_base.BootstrapCurrentModel):
  """Sauter model for bootstrap current."""

  def calculate_bootstrap_current(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: state.CoreProfiles,
      analytical_cache: formulas.AnalyticalCache | None = None,
  ) -> bootstrap_current_base.BootstrapCurrent:
    """Calculates bootstrap current according to the Sauter model."""
    assert isinstance(
        runtime_params.neoclassical,
        neoclassical_runtime_params.AnalyticalRuntimeParams,
    )
    bootstrap_params = runtime_params.neoclassical.bootstrap_current
    assert isinstance(
        bootstrap_params, bootstrap_current_runtime_params.RuntimeParams
    )
    if analytical_cache is None:
      analytical_cache = formulas.compute_analytical_cache(
          geometry, core_profiles
      )
    return _calculate_bootstrap_current(
        bootstrap_multiplier=bootstrap_params.bootstrap_multiplier,
        Z_eff_face=core_profiles.Z_eff_face,
        n_e=core_profiles.n_e,
        n_i=core_profiles.n_i,
        T_e=core_profiles.T_e,
        T_i=core_profiles.T_i,
        p_e=core_profiles.pressure_thermal_e,
        p_i=core_profiles.pressure_thermal_i,
        psi=core_profiles.psi,
        geo=geometry,
        f_trap=analytical_cache.f_trap,
        nu_e_star=analytical_cache.nu_e_star,
        nu_i_star=analytical_cache.nu_i_star,
    )

  def __eq__(self, other) -> bool:
    return isinstance(other, self.__class__)

  def __hash__(self) -> int:
    return hash(self.__class__)


class SauterModelConfig(bootstrap_current_base.BootstrapCurrentModelConfig):
  """Config for the Sauter model implementation of bootstrap current.

  Attributes:
    bootstrap_multiplier: Multiplication factor for bootstrap current.
  """

  model_name: Annotated[Literal['sauter'], torax_pydantic.JAX_STATIC] = 'sauter'

  def build_runtime_params(
      self,
  ) -> bootstrap_current_runtime_params.RuntimeParams:
    return bootstrap_current_runtime_params.RuntimeParams(
        bootstrap_multiplier=self.bootstrap_multiplier
    )

  def build_model(self) -> SauterModel:
    return SauterModel()


@jax.jit
def _calculate_bootstrap_current(
    *,
    bootstrap_multiplier: float,
    Z_eff_face: array_typing.FloatVectorFace,
    n_e: cell_variable.CellVariable,
    n_i: cell_variable.CellVariable,
    T_e: cell_variable.CellVariable,
    T_i: cell_variable.CellVariable,
    p_e: cell_variable.CellVariable,
    p_i: cell_variable.CellVariable,
    psi: cell_variable.CellVariable,
    geo: geometry_lib.Geometry,
    f_trap: array_typing.FloatVectorFace,
    nu_e_star: array_typing.FloatVectorFace,
    nu_i_star: array_typing.FloatVectorFace,
) -> bootstrap_current_base.BootstrapCurrent:
  """Calculates j_parallel_bootstrap using the Sauter model."""
  # pylint: disable=invalid-name

  # Terms for analytical fit
  L31 = sauter_formulas.calculate_L31(
      f_trap, nu_e_star, Z_eff_face
  )
  L32 = sauter_formulas.calculate_L32(
      f_trap, nu_e_star, Z_eff_face
  )
  L34 = sauter_formulas.calculate_L34(
      f_trap, nu_e_star, Z_eff_face
  )
  alpha = sauter_formulas.calculate_alpha(f_trap, nu_i_star)

  return bootstrap_current_base.calculate_analytic_bootstrap_current(
      bootstrap_multiplier=bootstrap_multiplier,
      n_e=n_e,
      n_i=n_i,
      T_e=T_e,
      T_i=T_i,
      p_e=p_e,
      p_i=p_i,
      psi=psi,
      geo=geo,
      L31=L31,
      L32=L32,
      L34=L34,
      alpha=alpha,
  )
