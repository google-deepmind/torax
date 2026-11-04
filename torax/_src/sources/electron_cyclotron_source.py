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

"""Electron cyclotron heating and current drive."""

import dataclasses
from typing import Annotated, ClassVar, Literal, Self

import chex
import jax
import jax.numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import solver_profiles as solver_profiles_lib
from torax._src.geometry import geometry
from torax._src.physics import collisions
from torax._src.sources import base
from torax._src.sources import core_sources
from torax._src.sources import formulas
from torax._src.sources import runtime_params as sources_runtime_params_lib
from torax._src.sources import source
from torax._src.sources import source_state as source_state_lib
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(sources_runtime_params_lib.RuntimeParams):
  """Runtime parameters for the electron-cyclotron source."""

  current_drive_efficiency: array_typing.FloatVector
  extra_prescribed_power_density: array_typing.FloatVector | None
  gaussian_width: array_typing.FloatScalar | None
  gaussian_location: array_typing.FloatScalar | None
  P_total: array_typing.FloatScalar | None


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class ElectronCyclotronSource(source.Source):
  """Electron cyclotron source for the T_e and Psi equations."""

  SOURCE_ID: ClassVar[str] = "ecrh"
  POPULATED_SOURCE_OUTPUTS: ClassVar[
      tuple[core_sources.PopulatedSourceOutput, ...]
  ] = (
      core_sources.PopulatedSourceOutput.POWER_DENSITY_E,
      core_sources.PopulatedSourceOutput.J_PARALLEL,
  )

  def _get_model_value(
      self,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      solver_profiles: solver_profiles_lib.SolverProfiles,
      source_state: source_state_lib.SourceState,
  ) -> tuple[core_sources.SourceOutput, source_state_lib.SourceState]:
    """Calculates external electron heat source and current drive from EC.

    Based on Lin-Liu, Y. R., Chan, V. S., & Prater, R. (2003). See
    https://torax.readthedocs.io/en/latest/physics_models/electron-cyclotron-derivation.html
    for more details.

    Args:
      runtime_params: Global runtime parameters.
      geo: Magnetic geometry.
      solver_profiles: Base core plasma profiles.
      source_state: Source state carried over from the previous timestep. Unused
        by this model and passed through unchanged.

    Returns:
      A tuple containing:
        - SourceOutput, with the power_density_e and j_parallel profiles
          populated.
        - SourceState for this source.
    """
    source_params = runtime_params.sources[self.SOURCE_ID]
    # Helps linter understand the type of source_params.
    assert isinstance(source_params, RuntimeParams)

    # Build the EC power deposition profile
    ec_power_density = jnp.zeros_like(solver_profiles.kinetic.T_e.value)
    if source_params.extra_prescribed_power_density is not None:
      ec_power_density = (
          ec_power_density + source_params.extra_prescribed_power_density
      )
    if source_params.P_total is not None:
      if source_params.gaussian_width is None:
        raise ValueError(
            "ElectronCyclotronSource requires `gaussian_width` to be set when"
            " `P_total` is provided."
        )
      if source_params.gaussian_location is None:
        raise ValueError(
            "ElectronCyclotronSource requires `gaussian_location` to be set"
            " when `P_total` is provided."
        )
      ec_power_density = ec_power_density + formulas.gaussian_profile(
          center=source_params.gaussian_location,
          width=source_params.gaussian_width,
          total=source_params.P_total,
          geo=geo,
      )

    j_tor_ec = (
        jnp.exp(
            jnp.log(16.0)
            + jnp.log(jnp.pi)
            + 2 * jnp.log(constants.CONSTANTS.epsilon_0)
            + jnp.log(solver_profiles.kinetic.T_e.value * 1e3)
            - (
                2 * jnp.log(constants.CONSTANTS.q_e)
                + jnp.log(
                    collisions.calculate_log_lambda_ee(
                        solver_profiles.kinetic.T_e.value,
                        solver_profiles.kinetic.n_e.value,
                    )
                )
                + jnp.log(solver_profiles.kinetic.n_e.value)
            )
        )
        * source_params.current_drive_efficiency
        * ec_power_density
    )

    # < j.B >
    q_cell = geometry.face_to_cell(solver_profiles.poloidal_flux.q_face)
    fsa_j_dot_B = (
        geo.F
        * geo.gm9
        * (1 + geo.g2 * geo.g3 / (16 * jnp.pi**4 * q_cell**2))
        * j_tor_ec
    )
    j_parallel_ec = fsa_j_dot_B / geo.B_0

    return (
        core_sources.SourceOutput(
            power_density_e=ec_power_density, j_parallel=j_parallel_ec
        ),
        source_state,
    )


class ElectronCyclotronSourceConfig(base.SourceConfigBase):
  r"""Config for the electron-cyclotron source.

  Attributes:
    current_drive_efficiency: Dimensionless current drive efficiency profile
      defined on the cell grid, :math:`\zeta = \frac{e^2\ln\Lambda_{ee}}{8
      \varepsilon_0^2} \frac{1}{\left\langle\frac{1}{R}\right\rangle} \frac{n_e
      [\mathrm{m^{-3}}]}{T_e [\mathrm{eV}]} \frac{I_{ec} [\mathrm{A}]}{P_{ec}
      [\mathrm{W}]}`. See
      https://torax.readthedocs.io/en/latest/physics_models/electron-cyclotron-derivation.html
        for more details.
    extra_prescribed_power_density: Manual EC power density profile on the rho
      grid. At least one of P_total or extra_prescribed_power_density must be
      provided when mode is 'model_based'.
    gaussian_width: Gaussian EC power density profile width. Required when mode
      is 'model_based' and P_total is provided.
    gaussian_location: Gaussian EC power density profile location. Required when
      mode is 'model_based' and P_total is provided.
    P_total: Gaussian EC total power. At least one of P_total or
      extra_prescribed_power_density must be provided when mode is
      'model_based'.
  """

  model_name: Annotated[
      Literal["gaussian_lin_liu"], torax_pydantic.JAX_STATIC
  ] = "gaussian_lin_liu"
  current_drive_efficiency: torax_pydantic.TimeVaryingArray = (
      torax_pydantic.ValidatedDefault({0.0: {0.0: 0.25, 1.0: 0.25}})
  )
  extra_prescribed_power_density: torax_pydantic.TimeVaryingArray | None = None
  gaussian_width: torax_pydantic.PositiveTimeVaryingScalar | None = None
  gaussian_location: torax_pydantic.UnitIntervalTimeVaryingScalar | None = None
  P_total: torax_pydantic.TimeVaryingScalar | None = None

  @pydantic.model_validator(mode="after")
  def _check_fields(self) -> Self:
    if self.mode == sources_runtime_params_lib.Mode.MODEL_BASED:
      if self.P_total is None and self.extra_prescribed_power_density is None:
        raise ValueError(
            "At least one of P_total or extra_prescribed_power_density must be"
            " provided when mode is 'model_based'."
        )
      if self.P_total is not None:
        if self.gaussian_width is None:
          raise ValueError(
              "gaussian_width must be provided when mode is 'model_based' and"
              " P_total is provided."
          )
        if self.gaussian_location is None:
          raise ValueError(
              "gaussian_location must be provided when mode is 'model_based'"
              " and P_total is provided."
          )
    return self

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        current_drive_efficiency=self.current_drive_efficiency.get_value(t),
        extra_prescribed_power_density=(
            None
            if self.extra_prescribed_power_density is None
            else self.extra_prescribed_power_density.get_value(t)
        ),
        gaussian_width=(
            None
            if self.gaussian_width is None
            else self.gaussian_width.get_value(t)
        ),
        gaussian_location=(
            None
            if self.gaussian_location is None
            else self.gaussian_location.get_value(t)
        ),
        P_total=(None if self.P_total is None else self.P_total.get_value(t)),
    )

  def build_source(self) -> ElectronCyclotronSource:
    return ElectronCyclotronSource()
