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
"""Base model for source pydantic configs."""

import abc
from typing import Annotated, Self

import chex
import jax.numpy as jnp
import pydantic
from torax._src import jax_utils
from torax._src.fvm import cell_variable
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import core_sources
from torax._src.sources import runtime_params
from torax._src.sources import source as source_lib
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


class FastIonPrescribedValues(torax_pydantic.ConfigBase):
  """Prescribed density and temperature profiles for a single fast-ion species.

  Attributes:
    n: Prescribed fast ion density profile [m^-3].
    T: Prescribed fast ion effective temperature profile [keV].
  """

  n: torax_pydantic.NonNegativeTimeVaryingArray
  T: torax_pydantic.NonNegativeTimeVaryingArray

  def build_fast_ion(
      self,
      t: chex.Numeric,
      *,
      species: str,
      source_id: str,
  ) -> fast_ion_lib.FastIon:
    """Evaluates prescribed profiles at time `t` into a `FastIon`."""
    if self.n.grid is None:
      raise RuntimeError('grid must be set before calling build_fast_ion.')
    face_centers = jnp.asarray(
        self.n.grid.face_centers, dtype=jax_utils.get_dtype()
    )
    return fast_ion_lib.FastIon(
        species=species,
        source=source_id,
        n=cell_variable.CellVariable(
            value=self.n.get_value(t),
            face_centers=face_centers,
            right_face_grad_constraint=None,
            right_face_constraint=self.n.get_value(t, grid_type='face_right'),
        ),
        T=cell_variable.CellVariable(
            value=self.T.get_value(t),
            face_centers=face_centers,
            right_face_grad_constraint=None,
            right_face_constraint=self.T.get_value(t, grid_type='face_right'),
        ),
    )


class PrescribedValues(torax_pydantic.ConfigBase):
  """Prescribed profile values for a source in `Mode.PRESCRIBED`.

  Each attribute is optional (`None` by default). Only the profile fields
  matching the source's `POPULATED_SOURCE_OUTPUTS` should be set when
  `mode == Mode.PRESCRIBED`. Sources with `PRODUCES_FAST_IONS = True` may
  additionally prescribe fast-ion species distributions via `fast_ions`.

  Attributes:
    power_density_i: Ion heating power density [W/m^3].
    power_density_e: Electron heating power density [W/m^3].
    j_parallel: Parallel current density `<j.B>/B_0` [A/m^2].
    particle_source_e: Electron particle source density [m^-3 s^-1].
    fast_ions: Mapping from fast-ion species symbol (`'H'`, `'D'`, `'T'`,
      `'He3'`, or `'He4'`) to prescribed `n` and `T` profiles
      (`FastIonPrescribedValues`). Only supported on sources that produce fast
      ions (`fusion` and `icrh`).
  """

  power_density_i: torax_pydantic.TimeVaryingArray | None = None
  power_density_e: torax_pydantic.TimeVaryingArray | None = None
  j_parallel: torax_pydantic.TimeVaryingArray | None = None
  particle_source_e: torax_pydantic.TimeVaryingArray | None = None
  fast_ions: dict[str, FastIonPrescribedValues] | None = None

  @pydantic.field_validator('fast_ions')
  @classmethod
  def _validate_fast_ion_species(
      cls, v: dict[str, FastIonPrescribedValues] | None
  ) -> dict[str, FastIonPrescribedValues] | None:
    if v is not None:
      for species in v:
        fast_ion_lib.validate_fast_ion_species(species, 'prescribed_values')
    return v

  def build_source_output(
      self,
      t: chex.Numeric,
      source_id: str = '',
  ) -> core_sources.SourceOutput:
    """Evaluates prescribed time-varying arrays at time `t`."""
    if self.fast_ions:
      fast_ions = tuple(
          fast_ion.build_fast_ion(t, species=species, source_id=source_id)
          for species, fast_ion in sorted(self.fast_ions.items())
      )
    else:
      fast_ions = ()
    return core_sources.SourceOutput(
        power_density_i=(
            self.power_density_i.get_value(t)
            if self.power_density_i is not None
            else None
        ),
        power_density_e=(
            self.power_density_e.get_value(t)
            if self.power_density_e is not None
            else None
        ),
        j_parallel=(
            self.j_parallel.get_value(t)
            if self.j_parallel is not None
            else None
        ),
        particle_source_e=(
            self.particle_source_e.get_value(t)
            if self.particle_source_e is not None
            else None
        ),
        fast_ions=fast_ions,
    )


class SourceConfigBase(torax_pydantic.ConfigBase):
  """Base model holding parameters common to all source model configs.

  Subclasses should define the `model_name` attribute as a `Literal`
  string. This is used as an identifier for the model in the source config
  for Pydantic to "discriminate" against. This should be given a unique value
  for each source model implementation.

  Subclasses must override `build_source` to return the `Source` object for this
  config, and may override `build_runtime_params` if the source uses custom
  runtime parameters.

  Attributes:
    mode: Defines how the source values are computed. Available options are:

      - `'zero'`: Source is set to zero.
      - `'model_based'`: Source values come from a model in code. When more
        than one model implementation is available for a source, select it
        via `model_name`.
      - `'prescribed'`: Source values are prescribed by the user via
        `prescribed_values`.
    is_explicit: Defines whether the source is evaluated explicitly or
      implicitly. Explicit sources are calculated based on the simulation state
      at the beginning of a time step (or have no state dependence). Implicit
      sources depend on updated states as the iterative solvers evolve the state
      through a time step. If a source model is complex but evolves over slow
      timescales compared to the state, it may be beneficial to set it as
      explicit. Not all source types support both modes; an error is raised at
      runtime if an unsupported combination is configured.
    prescribed_values: Prescribed profile values for the source, keyed by output
      channel name (`power_density_i`, `power_density_e`, `j_parallel`,
      `particle_source_e`, and optional `fast_ions` for fast-ion sources). Only
      channels matching the source's `POPULATED_SOURCE_OUTPUTS` (and `fast_ions`
      when `PRODUCES_FAST_IONS` is `True`) should be set. Used only when
      `mode == 'prescribed'`. Each channel accepts any
      :ref:`time-varying array <Time-varying arrays>` data structure and is
      evaluated independently on the cell grid.
  """  # fmt: skip

  mode: Annotated[runtime_params.Mode, torax_pydantic.JAX_STATIC] = (
      runtime_params.Mode.MODEL_BASED
  )
  is_explicit: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  prescribed_values: PrescribedValues = torax_pydantic.ValidatedDefault({})

  @pydantic.model_validator(mode='after')
  def _validate_prescribed_fast_ions(self) -> Self:
    if self.prescribed_values.fast_ions:
      source = self.build_source()
      if not source.PRODUCES_FAST_IONS:
        raise ValueError(
            f'{type(source).__name__} does not support fast_ions in'
            ' prescribed_values.'
        )
    return self

  @abc.abstractmethod
  def build_source(
      self,
  ) -> source_lib.SourceBase:
    """Builds a source object from the model config."""

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> runtime_params.RuntimeParams:
    """Builds the runtime parameters for the source at time `t`."""
    source_id = (
        self.build_source().SOURCE_ID
        if self.prescribed_values.fast_ions
        else ''
    )
    return runtime_params.RuntimeParams(
        prescribed_values=self.prescribed_values.build_source_output(
            t, source_id=source_id
        ),
        mode=self.mode,
        is_explicit=self.is_explicit,
    )
