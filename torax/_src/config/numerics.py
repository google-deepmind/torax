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

"""Numerics parameters used throughout TORAX simulations."""

import dataclasses
import functools
from typing import Annotated, Self

import chex
import jax
import pydantic
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass
class RuntimeParams:
  """Generic numeric parameters for the simulation.

  For definitions see `Numerics`.
  """

  t_initial: float
  t_final: float
  current_diffusion_time_reduction_factor: array_typing.FloatScalar
  dW_dt_smoothing_time_scale: float
  min_rho_norm: float
  evolve_ion_heat: bool = jax.tree.static()
  evolve_electron_heat: bool = jax.tree.static()
  evolve_current: bool = jax.tree.static()
  evolve_density: bool = jax.tree.static()
  exact_t_final: bool = jax.tree.static()
  enable_fast_ions: bool = jax.tree.static()

  @functools.cached_property
  def evolving_names(self) -> tuple[str, ...]:
    """The names of core_profiles variables that are evolved by the solver."""
    evolving_names = []
    if self.evolve_ion_heat:
      evolving_names.append('T_i')
    if self.evolve_electron_heat:
      evolving_names.append('T_e')
    if self.evolve_current:
      evolving_names.append('psi')
    if self.evolve_density:
      evolving_names.append('n_e')
    return tuple(evolving_names)


class Numerics(torax_pydantic.BaseModelFrozen):
  """Generic numeric parameters for the simulation.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#numerics.

  Attributes:
    t_initial: Simulation start time, in units of seconds.
    t_final: Simulation end time, in units of seconds.
    exact_t_final: If True, ensures that the simulation end time is exactly
      `t_final`, by adapting the final `dt` to match.
    evolve_ion_heat: Solve the ion heat equation (ion temperature evolves over
      time).
    evolve_electron_heat: Solve the electron heat equation (electron temperature
      evolves over time)
    evolve_current: Solve the current equation (current evolves over time).
    evolve_density: Solve the density equation (n evolves over time).
    current_diffusion_time_reduction_factor: Factor that divides the transient
      term in the poloidal flux (psi) PDE to shorten the current diffusion
      timescale without modifying physical conductivity, loop voltage, or
      heating.
    dW_dt_smoothing_time_scale: Time scale [s] for the exponential moving
      average smoothing of dW/dt terms used in P_SOL and confinement time
      calculations. If 0.0, no smoothing is applied and raw dW/dt is used.
    min_rho_norm: Minimum rho_norm value below which current profile values are
      extrapolated to the axis in psi calculations, to avoid numerical artifacts
      near rho=0.
  """

  t_initial: torax_pydantic.Second = 0.0
  t_final: torax_pydantic.Second = 5.0
  exact_t_final: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  evolve_ion_heat: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  evolve_electron_heat: Annotated[bool, torax_pydantic.JAX_STATIC] = True
  evolve_current: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  evolve_density: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  enable_fast_ions: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  current_diffusion_time_reduction_factor: (
      torax_pydantic.PositiveTimeVaryingScalar
  ) = torax_pydantic.ValidatedDefault(1.0)
  dW_dt_smoothing_time_scale: pydantic.NonNegativeFloat = 0.3
  min_rho_norm: torax_pydantic.UnitInterval = 0.015

  T_min_eV: pydantic.PositiveFloat = 5.0

  @pydantic.model_validator(mode='after')
  def model_validation(self) -> Self:
    if self.t_initial > self.t_final:
      raise ValueError(
          't_initial must be less than or equal to t_final. '
          f't_initial: {self.t_initial}, t_final: {self.t_final}'
      )
    return self

  @property
  def evolving_names(self) -> tuple[str, ...]:
    """The names of core_profiles variables that are evolved by the solver."""
    evolving_names = []
    if self.evolve_ion_heat:
      evolving_names.append('T_i')
    if self.evolve_electron_heat:
      evolving_names.append('T_e')
    if self.evolve_current:
      evolving_names.append('psi')
    if self.evolve_density:
      evolving_names.append('n_e')
    return tuple(evolving_names)

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds a RuntimeParams object for time t."""
    return RuntimeParams(
        t_initial=self.t_initial,
        t_final=self.t_final,
        current_diffusion_time_reduction_factor=(
            self.current_diffusion_time_reduction_factor.get_value(t)
        ),
        dW_dt_smoothing_time_scale=self.dW_dt_smoothing_time_scale,
        min_rho_norm=self.min_rho_norm,
        evolve_ion_heat=self.evolve_ion_heat,
        evolve_electron_heat=self.evolve_electron_heat,
        evolve_current=self.evolve_current,
        evolve_density=self.evolve_density,
        exact_t_final=self.exact_t_final,
        enable_fast_ions=self.enable_fast_ions,
    )
