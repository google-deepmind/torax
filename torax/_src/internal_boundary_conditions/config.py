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
import chex
import numpy as np
import pydantic
from torax._src.internal_boundary_conditions import base_model
from torax._src.internal_boundary_conditions import beta_poloidal_prime as beta_poloidal_prime_lib
from torax._src.internal_boundary_conditions import no_ibc
from torax._src.internal_boundary_conditions import prescribed
from torax._src.internal_boundary_conditions import runtime_params as ibc_runtime_params
from torax._src.torax_pydantic import interpolated_param_2d
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Self

# pylint: disable=invalid-name


class BetaPoloidalPrimeIBCConfig(torax_pydantic.ConfigBase):
  """Dynamically computes L-mode edge kinetic profiles ($T_i$, $T_e$, $n_e$).

  Constrains the edge region $rho_norm >= rho_norm_edge$ using a prescribed
  gradient of the local poloidal beta with respect to normalized poloidal flux,
  $beta_pol'$ ($-partial(beta_pol_local) / partial(psi_N)$), motivated by
  empirical observations of critical edge pressure gradients in L-mode
  plasmas |labombard2008|.

  Attributes:
    model_name: Internal boundary condition model name, set to
      `'beta_poloidal_prime'`.
    rho_norm_edge: Edge normalized toroidal flux coordinate $rho_norm_edge$
      bounding the constrained edge region.
    beta_poloidal_prime: Prescribed normalized poloidal beta gradient
      $beta_pol' > 0$ [dimensionless] (serves as the high-power asymptotic value
      when `mode = 'power_dependent'`).
    T_i_T_e_ratio: Prescribed ratio $T_i / T_e$ in the edge region
      [dimensionless].
    n_e_edge: Prescribed electron density at the edge boundary $rho_norm_edge$
      [m^-3 if `n_e_is_f_GW = False`, Greenwald fraction if `n_e_is_f_GW =
      True`]. Mutually exclusive with `n_e_edge_multiplier` (exactly one must be
      provided).
    n_e_edge_multiplier: Multiplier applied to the right boundary condition
      density (`n_e_right_bc`) to set the electron density at $rho_norm_edge$
      [dimensionless]. Mutually exclusive with `n_e_edge` (exactly one must be
      provided).
    n_e_is_f_GW: If `True`, `n_e_edge` is interpreted as a Greenwald fraction
      [dimensionless] instead of absolute density [m^-3].
    mode: Controls how the effective edge poloidal beta gradient is
      determined:

      - `'constant'`: Uses `beta_poloidal_prime` directly.
      - `'power_dependent'`: Scales the effective poloidal beta gradient
        smoothly with the net heating power crossing the separatrix $P_SOL$:

        $$
        beta_pol_eff' = beta_pol_min' +
        (beta_pol' - beta_pol_min') * y(x)
        $$

        where $x = max(P_SOL, 0) / P_SOL_scaling$,
        $z_1 = (x^2 + 0.3 * x) / 1.3$, and $y = (z_1^alpha + 1)^(1 / alpha)$
        with $alpha = -15$.
    beta_poloidal_prime_min: Minimum normalized poloidal beta gradient
      $beta_pol_min' >= 0$ at $P_SOL <= 0$ when `mode = 'power_dependent'`
      [dimensionless].
    P_SOL_scaling: Characteristic power crossing the separatrix
      $P_SOL_scaling > 0$ [W] for the power transition when `mode =
      'power_dependent'` (required when `mode = 'power_dependent'`). Sets the
      knee of the power scaling curve so that a scaled $P_SOL$ of `1` is
      approximately where `beta_poloidal_prime` saturates.
  """  # fmt: skip

  model_name: Annotated[
      Literal['beta_poloidal_prime'], torax_pydantic.JAX_STATIC
  ] = 'beta_poloidal_prime'
  rho_norm_edge: torax_pydantic.TimeVaryingScalar
  beta_poloidal_prime: torax_pydantic.PositiveTimeVaryingScalar
  T_i_T_e_ratio: torax_pydantic.PositiveTimeVaryingScalar
  n_e_edge: torax_pydantic.PositiveTimeVaryingScalar | None = None
  n_e_edge_multiplier: torax_pydantic.PositiveTimeVaryingScalar | None = None
  n_e_is_f_GW: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  mode: Annotated[
      beta_poloidal_prime_lib.Mode, torax_pydantic.JAX_STATIC
  ] = beta_poloidal_prime_lib.Mode.CONSTANT
  beta_poloidal_prime_min: torax_pydantic.NonNegativeTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(0.2)
  )
  P_SOL_scaling: torax_pydantic.PositiveTimeVaryingScalar | None = None

  @pydantic.model_validator(mode='after')
  def _validate_params(self) -> Self:
    if (self.n_e_edge is None) == (self.n_e_edge_multiplier is None):
      raise ValueError(
          "Exactly one of 'n_e_edge' or 'n_e_edge_multiplier' must be provided."
      )
    if (
        self.mode == beta_poloidal_prime_lib.Mode.POWER_DEPENDENT
        and self.P_SOL_scaling is None
    ):
      raise ValueError(
          'P_SOL_scaling must be provided when mode is'
          f' {beta_poloidal_prime_lib.Mode.POWER_DEPENDENT!r}.'
      )
    return self

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return beta_poloidal_prime_lib.BetaPoloidalPrimeIBCModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> beta_poloidal_prime_lib.RuntimeParams:
    """Builds the runtime params for the beta_poloidal_prime IBC model."""
    return beta_poloidal_prime_lib.RuntimeParams(
        rho_norm_edge=self.rho_norm_edge.get_value(t),
        beta_poloidal_prime=self.beta_poloidal_prime.get_value(t),
        T_i_T_e_ratio=self.T_i_T_e_ratio.get_value(t),
        n_e_edge=(
            self.n_e_edge.get_value(t) if self.n_e_edge is not None else None
        ),
        n_e_edge_multiplier=(
            self.n_e_edge_multiplier.get_value(t)
            if self.n_e_edge_multiplier is not None
            else None
        ),
        n_e_is_f_GW=self.n_e_is_f_GW,
        mode=self.mode,
        beta_poloidal_prime_min=self.beta_poloidal_prime_min.get_value(t),
        P_SOL_scaling=(
            self.P_SOL_scaling.get_value(t)
            if self.P_SOL_scaling is not None
            else None
        ),
    )


class NoIBCConfig(torax_pydantic.ConfigBase):
  """Pydantic config for when there are no internal boundary conditions.

  Attributes:
    model_name: Internal boundary condition model name, set to `'no_ibc'`.
  """

  model_name: Annotated[Literal['no_ibc'], torax_pydantic.JAX_STATIC] = 'no_ibc'

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return no_ibc.NoIBCModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> ibc_runtime_params.RuntimeParams:
    del t
    return ibc_runtime_params.RuntimeParams()


def _active_locations_at_time(
    channel_ibc: interpolated_param_2d.SparseTimeVaryingArray,
    t: float,
) -> set[float | tuple[float, float]]:
  """Returns radial location keys where `channel_ibc` is nonzero at time `t`."""
  active: set[float | tuple[float, float]] = set()
  for location, tva in channel_ibc.values:
    # Without a grid we cannot call `get_value(t)`, so mark each time knot as
    # 1.0 (nonzero) or 0.0 (all zero) and linearly interpolate those flags.
    # This is > 0 at time `t` whenever the interpolated profile is nonzero.
    is_nonzero = [float(np.any(vals != 0.0)) for _, vals in tva.value.values()]
    if np.interp(t, list(tva.value), is_nonzero) > 0.0:
      active.add(location)
  return active


class PrescribedIBCConfig(torax_pydantic.ConfigBase):
  """Explicitly prescribes time-varying internal boundary conditions.

  Configures internal boundary conditions for `T_i`, `T_e`, and `n_e` using
  `SparseTimeVaryingArray` types. Any values of `0.0` are treated as no internal
  boundary condition at that location.

  All provided profiles must use the same radial locations at all times. If any
  profile has a nonzero internal boundary condition, one must be provided for
  every evolving profile.

  Attributes:
    model_name: Internal boundary condition model name, set to `'prescribed'`.
    T_i: Prescribed ion temperature internal boundary condition values [keV] at
      sparse radial locations or ranges. Values of `0.0` indicate no internal
      boundary condition at that location.
    T_e: Prescribed electron temperature internal boundary condition values
      [keV] at sparse radial locations or ranges. Values of `0.0` indicate no
      internal boundary condition at that location.
    n_e: Prescribed electron density internal boundary condition values [m^-3]
      at sparse radial locations or ranges. Values of `0.0` indicate no internal
      boundary condition at that location.
  """

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

  def _active_channel_ibcs(
      self,
  ) -> dict[str, interpolated_param_2d.SparseTimeVaryingArray]:
    """Returns channel -> IBC mapping for channels with any nonzero value."""
    active: dict[str, interpolated_param_2d.SparseTimeVaryingArray] = {}
    for channel, channel_ibc in (
        ('T_i', self.T_i),
        ('T_e', self.T_e),
        ('n_e', self.n_e),
    ):
      for _, tva in channel_ibc.values:
        if any(np.any(vals != 0.0) for _, vals in tva.value.values()):
          active[channel] = channel_ibc
          break
    return active

  @property
  def active_channels(self) -> frozenset[str]:
    """Channels with at least one nonzero prescribed IBC value."""
    return frozenset(self._active_channel_ibcs())

  # TODO(b/590388748): In V3, move radial locations to top-level fields on
  # PrescribedIBCConfig so channel alignment is enforced directly by the schema
  # instead of this after-validator.
  @pydantic.model_validator(mode='after')
  def _validate_matching_channel_locations(self) -> Self:
    """Checks that all active IBC channels share the same radial locations."""
    active_by_channel = self._active_channel_ibcs()
    if len(active_by_channel) <= 1:
      return self

    # Collect every time knot where any active channel's IBC can change.
    # SparseTimeVaryingArray.values holds (location, TimeVaryingArray) pairs,
    # where TimeVaryingArray.value is a dict keyed by time [s].
    time_knots: set[float] = set()
    for channel_ibc in active_by_channel.values():
      for _, tva in channel_ibc.values:
        time_knots.update(tva.value)

    ref_channel, *other_channels = active_by_channel
    for t in sorted(time_knots):
      ref_locs = _active_locations_at_time(active_by_channel[ref_channel], t)
      for channel in other_channels:
        locs = _active_locations_at_time(active_by_channel[channel], t)
        if locs != ref_locs:
          raise ValueError(
              'Prescribed internal boundary conditions must use the same'
              f' radial locations for all active channels. At t={t},'
              f' {ref_channel} locations {sorted(ref_locs, key=str)} differ'
              f' from {channel} locations {sorted(locs, key=str)}.'
          )
    return self

  def build_model(self) -> base_model.InternalBoundaryConditionModel:
    """Builds the internal boundary condition model."""
    return prescribed.PrescribedIBCModel()

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> prescribed.RuntimeParams:
    """Builds the runtime params for the internal boundary conditions."""
    return prescribed.RuntimeParams(
        T_i=self.T_i.get_value(t),
        T_e=self.T_e.get_value(t),
        n_e=self.n_e.get_value(t),
    )


InternalBoundaryConditionsConfig = Annotated[
    PrescribedIBCConfig | BetaPoloidalPrimeIBCConfig | NoIBCConfig,
    pydantic.Field(discriminator='model_name'),
]
