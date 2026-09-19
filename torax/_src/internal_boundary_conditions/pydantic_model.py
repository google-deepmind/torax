# Copyright 2026 DeepMind Technologies Limited
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

"""Pydantic config for internal boundary conditions."""

from typing import Annotated, Literal

import pydantic
from torax._src import array_typing
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import beta_poloidal_prime as beta_poloidal_prime_lib
from torax._src.internal_boundary_conditions import no_ibc
from torax._src.internal_boundary_conditions import prescribed
from torax._src.internal_boundary_conditions import runtime_params as ibc_runtime_params
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class BetaPoloidalPrimeIBC(torax_pydantic.BaseModelFrozen):
  """Pydantic model for beta_poloidal_prime internal boundary conditions."""

  model_name: Annotated[
      Literal['beta_poloidal_prime'], torax_pydantic.JAX_STATIC
  ] = 'beta_poloidal_prime'
  rho_norm_edge: torax_pydantic.TimeVaryingScalar
  n_e_edge: torax_pydantic.PositiveTimeVaryingScalar
  beta_poloidal_prime: torax_pydantic.PositiveTimeVaryingScalar
  Ti_Te_ratio: torax_pydantic.PositiveTimeVaryingScalar
  n_e_is_fGW: Annotated[bool, torax_pydantic.JAX_STATIC] = False

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return beta_poloidal_prime_lib.BetaPoloidalPrimeIBCModel()

  def build_runtime_params(
      self, t: array_typing.FloatScalar
  ) -> beta_poloidal_prime_lib.RuntimeParams:
    """Builds the runtime params for the beta_poloidal_prime IBC model."""
    return beta_poloidal_prime_lib.RuntimeParams(
        rho_norm_edge=self.rho_norm_edge.get_value(t),
        n_e_edge=self.n_e_edge.get_value(t),
        beta_poloidal_prime=self.beta_poloidal_prime.get_value(t),
        Ti_Te_ratio=self.Ti_Te_ratio.get_value(t),
        n_e_is_fGW=self.n_e_is_fGW,
    )


class NoIBC(torax_pydantic.BaseModelFrozen):
  """Pydantic model for when there are no internal boundary conditions."""

  model_name: Annotated[Literal['no_ibc'], torax_pydantic.JAX_STATIC] = 'no_ibc'

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return no_ibc.NoIBCModel()

  def build_runtime_params(
      self, t: array_typing.FloatScalar
  ) -> ibc_runtime_params.RuntimeParams:
    del t
    return ibc_runtime_params.RuntimeParams()


class PrescribedIBC(torax_pydantic.BaseModelFrozen):
  """Pydantic model for prescribed internal boundary conditions."""

  model_name: Annotated[Literal['prescribed'], torax_pydantic.JAX_STATIC] = (
      'prescribed'
  )
  T_i: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  T_e: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )
  n_e: interpolated_param_2d.SparseTimeVaryingArray = (
      torax_pydantic.ValidatedDefault(0.0)
  )

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return prescribed.PrescribedIBCModel()

  def build_runtime_params(
      self, t: array_typing.FloatScalar
  ) -> prescribed.RuntimeParams:
    """Builds the runtime params for the internal boundary conditions."""
    return prescribed.RuntimeParams(
        T_i=self.T_i.get_value(t),
        T_e=self.T_e.get_value(t),
        n_e=self.n_e.get_value(t),
    )


InternalBoundaryConditionsConfig = Annotated[
    PrescribedIBC | BetaPoloidalPrimeIBC | NoIBC,
    pydantic.Field(discriminator='model_name'),
]
