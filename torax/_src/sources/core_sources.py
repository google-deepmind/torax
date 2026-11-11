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
"""Core source/sink profiles for all the sources in TORAX."""

from collections.abc import Iterator, Mapping
import dataclasses
from typing import Any, Literal, Self

import jax
import jax.numpy as jnp
from torax._src import array_typing
from torax._src import constants
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import runtime_params

# pylint: disable=invalid-name

PopulatedSourceOutput = runtime_params.PopulatedSourceOutput
SourceOutput = runtime_params.SourceOutput


def j_parallel_to_psi_source(
    j_parallel: array_typing.Array,
    geo: geometry.Geometry,
) -> array_typing.Array:
  """Converts parallel current <j.B>/B_0 to the psi PDE source term."""
  total_j_dot_B = j_parallel * geo.B_0
  prefactor = (
      8 * geo.vpr * jnp.pi**2 * constants.CONSTANTS.mu_0 * geo.Phi_b / geo.F**2
  )
  return -total_j_dot_B * prefactor


def _asarray_source_output(output: SourceOutput) -> SourceOutput:
  """Converts non-None profile arrays in SourceOutput to JAX arrays."""
  return SourceOutput(
      power_density_i=(
          jnp.asarray(output.power_density_i)
          if output.power_density_i is not None
          else None
      ),
      power_density_e=(
          jnp.asarray(output.power_density_e)
          if output.power_density_e is not None
          else None
      ),
      j_parallel=(
          jnp.asarray(output.j_parallel)
          if output.j_parallel is not None
          else None
      ),
      particle_source_e=(
          jnp.asarray(output.particle_source_e)
          if output.particle_source_e is not None
          else None
      ),
      fast_ions=output.fast_ions,
  )


def extract_fast_ions(
    source_outputs: Mapping[str, SourceOutput],
) -> tuple[fast_ion_lib.FastIon, ...]:
  """Returns all fast ions produced by source outputs."""
  fast_ions = []
  for out in source_outputs.values():
    fast_ions.extend(out.fast_ions)
  return tuple(fast_ions)


def extract_psi_sources(
    source_outputs: Mapping[str, SourceOutput],
    source_models: Any,
) -> dict[str, jax.Array]:
  """Extracts psi source profiles from SourceOutput mapping."""
  return {
      name: jnp.asarray(out.j_parallel)
      for name, out in source_outputs.items()
      if name in source_models.standard_sources and out.j_parallel is not None
  }


def sum_total_psi_sources(
    basic_source_outputs: Mapping[str, SourceOutput],
    source_models: Any,
    bootstrap_current: bootstrap_current_base.BootstrapCurrent,
    geo: geometry.Geometry,
) -> array_typing.Array:
  """Extracts and sums total psi PDE source term from basic sources and bootstrap."""
  psi_sources = extract_psi_sources(basic_source_outputs, source_models)
  total_j_parallel = bootstrap_current.j_parallel_bootstrap + sum(
      psi_sources.values(), start=jnp.zeros_like(geo.rho)
  )
  return j_parallel_to_psi_source(total_j_parallel, geo)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class QeiInfo:
  """Represents the source values coming from a QeiSource."""

  implicit_ii: array_typing.Array
  explicit_i: array_typing.Array
  implicit_ee: array_typing.Array
  explicit_e: array_typing.Array
  implicit_ie: array_typing.Array
  implicit_ei: array_typing.Array
  p_ei: array_typing.Array

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> Self:
    return cls(
        implicit_ii=jnp.zeros_like(geo.rho),
        explicit_i=jnp.zeros_like(geo.rho),
        implicit_ee=jnp.zeros_like(geo.rho),
        explicit_e=jnp.zeros_like(geo.rho),
        implicit_ie=jnp.zeros_like(geo.rho),
        implicit_ei=jnp.zeros_like(geo.rho),
        p_ei=jnp.zeros_like(geo.rho),
    )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Converts QeiInfo into an OutputVar mapping."""
    p_ei_i_key = output_keys.OutputKey(
        output_keys.p_source_i_key('ei_exchange'),
        units=output_keys.Units.WATT_PER_CUBIC_METER,
        grid_type=output_keys.GridType.CELL,
    )
    p_ei_e_key = output_keys.OutputKey(
        output_keys.p_source_e_key('ei_exchange'),
        units=output_keys.Units.WATT_PER_CUBIC_METER,
        grid_type=output_keys.GridType.CELL,
    )
    return {
        p_ei_i_key: context.pack(p_ei_i_key, self.p_ei),
        p_ei_e_key: context.pack(p_ei_e_key, -self.p_ei),
    }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CoreSources:
  """Collection of profiles for all sources in TORAX.

  Standard source outputs are stored in `sources`, keyed by source name, while
  special-case profiles `bootstrap_current` and `qei` are pulled out into their
  own attributes as these sources need to be treated differently (though they
  could still be set to zero using appropriate runtime params).

  This dataclass is inspired by the IMAS `core_sources` IDS. It is not a 1:1
  mapping to that schema, but it contains similar profiles as you'd expect in
  that IDS.
  """

  # Special-case profiles.
  bootstrap_current: bootstrap_current_base.BootstrapCurrent
  qei: QeiInfo
  # Outputs of standard sources keyed by source name.
  sources: dict[str, SourceOutput] = dataclasses.field(default_factory=dict)

  def j_parallel(self, geo: geometry.Geometry) -> array_typing.FloatVectorCell:
    """Returns the summed parallel current density across all sources."""
    return jnp.asarray(
        sum(
            (
                output.j_parallel
                for output in self.sources.values()
                if output.j_parallel is not None
            ),
            start=jnp.zeros_like(geo.rho),
        )
    )

  @classmethod
  def zeros(cls, geo: geometry.Geometry) -> Self:
    """Returns a CoreSources object with all zero profiles."""
    return cls(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=QeiInfo.zeros(geo),
    )

  def total_psi_sources(self, geo: geometry.Geometry) -> array_typing.Array:
    """Returns the total poloidal flux source terms on the cell grid.

    All `psi` sources (`bootstrap_current.j_parallel_bootstrap` and
    `self.j_parallel(geo)`) are assumed to follow the `<j.B> / B_0` convention
    (flux-surface-averaged `j.B` normalized by the on-axis vacuum toroidal
    field `B_0`).

    Args:
      geo: The magnetic geometry of the tokamak.

    Returns:
      Array of total poloidal flux source terms on the cell grid.
    """
    total_j_parallel = (
        self.bootstrap_current.j_parallel_bootstrap + self.j_parallel(geo)
    )
    return j_parallel_to_psi_source(total_j_parallel, geo)

  def total_sources(
      self,
      source_type: Literal[
          'particle_source_e', 'power_density_i', 'power_density_e'
      ],
      geo: geometry.Geometry,
  ) -> array_typing.Array:
    """Returns the total volume-weighted sources for a given profile channel.

    To compute `psi` sources, use `total_psi_sources` instead.

    Args:
      source_type: The source output channel ('particle_source_e',
        'power_density_i', or 'power_density_e').
      geo: The magnetic geometry of the tokamak.

    Returns:
      Array of total source terms weighted by `geo.vpr` on the cell grid.
    """
    profiles = [
        getattr(output, source_type)
        for output in self.sources.values()
        if getattr(output, source_type) is not None
    ]
    if not profiles:
      return jnp.zeros_like(geo.vpr)
    return sum(profiles) * geo.vpr

  @classmethod
  def from_source_outputs(
      cls,
      source_outputs: Mapping[str, SourceOutput],
      standard_sources: Mapping[str, Any],
      bootstrap_current: bootstrap_current_base.BootstrapCurrent,
      qei: QeiInfo,
  ) -> Self:
    """Constructs CoreSources from evaluated source outputs."""
    sources = {
        name: _asarray_source_output(output)
        for name, output in source_outputs.items()
        if name in standard_sources
    }
    return cls(
        bootstrap_current=bootstrap_current,
        qei=qei,
        sources=sources,
    )

  def update_sources(
      self,
      source_outputs: Mapping[str, SourceOutput],
      source_models: Any,
  ) -> Self:
    """Updates sources with source_outputs, returning a new CoreSources."""
    sources = dict(self.sources)
    for name, output in source_outputs.items():
      if name in source_models.standard_sources:
        sources[name] = _asarray_source_output(output)
    return dataclasses.replace(
        self,
        sources=sources,
    )

  def _iterate_channels_for_output(
      self,
      *,
      renames: Mapping[str, str],
  ) -> Iterator[tuple[output_keys.OutputKey, array_typing.Array]]:
    """Yields (OutputKey, data) across all active channels for output.

    Args:
      renames: Mapping from internal profile names to output naming convention.

    Yields:
      Tuples of (OutputKey, data) for all active source profiles.
    """
    for profile, output in self.sources.items():
      name = renames.get(profile, profile)
      if output.power_density_i is not None:
        yield (
            output_keys.OutputKey(
                output_keys.p_source_i_key(name),
                units=output_keys.Units.WATT_PER_CUBIC_METER,
                grid_type=output_keys.GridType.CELL,
            ),
            output.power_density_i,
        )
      if output.power_density_e is not None:
        yield (
            output_keys.OutputKey(
                output_keys.p_source_e_key(name),
                units=output_keys.Units.WATT_PER_CUBIC_METER,
                grid_type=output_keys.GridType.CELL,
            ),
            output.power_density_e,
        )
      if output.j_parallel is not None:
        yield (
            output_keys.OutputKey(
                output_keys.j_parallel_source_key(name),
                units=output_keys.Units.AMPERE_PER_SQUARE_METER,
                grid_type=output_keys.GridType.CELL,
            ),
            output.j_parallel,
        )
      if output.particle_source_e is not None:
        yield (
            output_keys.OutputKey(
                output_keys.s_source_key(name),
                units=output_keys.Units.INVERSE_CUBIC_METER_PER_SECOND,
                grid_type=output_keys.GridType.CELL,
            ),
            output.particle_source_e,
        )

  def to_output_dict(
      self,
      context: output_grid_context.OutputGridContext,
  ) -> dict[str, output_grid_context.OutputVar]:
    """Packs time-stacked source profiles into xarray-ready variables.

    Args:
      context: Grid context providing time and radial coordinates for shape
        validation and `(dims, data, attrs)` packing.

    Returns:
      Mapping from source output variable names to `(dims, data, attrs)`
      `OutputVar` tuples:
        - `j_parallel_bootstrap`: `('time', 'rho_norm')` with shape
          `(n_time, n_rho_cell + 2)`.
        - `p_ei_exchange_i`, `p_ei_exchange_e`, and per-source
          heating/current/particle channels (`p_*_i`, `p_*_e`, `j_parallel_*`,
          `s_*`): `('time', 'rho_cell_norm')` with shape `(n_time, n_rho_cell)`.
    """
    out_dict = self.bootstrap_current.to_output_dict(context)
    out_dict.update(self.qei.to_output_dict(context))
    for key, data in self._iterate_channels_for_output(
        renames=output_keys.SOURCE_NAME_RENAMES
    ):
      out_dict[key] = context.pack(key, data)
    return out_dict
