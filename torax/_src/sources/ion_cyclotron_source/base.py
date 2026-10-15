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
"""Base infrastructure for ion-cyclotron resonance heating (ICRH) sources."""

import dataclasses
from typing import Annotated, ClassVar

import chex
import jax
from jax import numpy as jnp
from torax._src import array_typing
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import base as source_base
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources import source
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams(source_runtime_params_lib.RuntimeParams):
  """Base runtime parameters for ICRH models."""

  P_total: array_typing.FloatScalar
  absorption_fraction: array_typing.FloatScalar
  minority_species: str | None = dataclasses.field(metadata={'static': True})


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class IonCyclotronSource(source.Source):
  """Ion cyclotron source."""

  SOURCE_ID: ClassVar[str] = 'icrh'
  AFFECTED_CORE_PROFILES: ClassVar[tuple[source.AffectedCoreProfile, ...]] = (
      source.AffectedCoreProfile.TEMP_ION,
      source.AffectedCoreProfile.TEMP_EL,
      source.AffectedCoreProfile.FAST_IONS,
  )

  @classmethod
  def zero_fast_ions(
      cls,
      geo: geometry.Geometry,
  ) -> tuple[fast_ion_lib.FastIon, ...]:
    zeros = jnp.zeros_like(geo.rho)
    return tuple(
        fast_ion_lib.FastIon(
            species=species,
            source=cls.SOURCE_ID,
            n=cell_variable.CellVariable(
                value=zeros,
                face_centers=geo.rho_face_norm,
                right_face_grad_constraint=None,
                right_face_constraint=jnp.zeros(()),
            ),
            T=cell_variable.CellVariable(
                value=zeros,
                face_centers=geo.rho_face_norm,
                right_face_grad_constraint=None,
                right_face_constraint=jnp.zeros(()),
            ),
        )
        for species in fast_ion_lib.FAST_ION_SPECIES
    )


class IonCyclotronSourceConfig(source_base.SourceConfigBase):
  """Base configuration for IonCyclotronSource.

  This base class contains fields common to all ICRH model implementations.
  Subclasses implement the specific model logic,and must override `model_name`
  with a `Literal` to serve as discriminator.

  Attributes:
    model_name: Discriminator field for Pydantic. Subclasses must override with
      a `Literal` value.
    P_total: Total heating power [W].
    absorption_fraction: Fraction of absorbed power.
    mode: Defines how the source values are computed.
    minority_species: Optional symbol of the minority species (e.g., 'He3').
      When specified, the minority concentration is extracted from
      plasma_composition. The species can be either a main ion or an impurity.
  """

  model_name: Annotated[str, torax_pydantic.JAX_STATIC] = ''
  # TODO(b/434175938): Remove default source amplitudes in V2.
  P_total: torax_pydantic.TimeVaryingScalar = torax_pydantic.ValidatedDefault(
      10e6
  )
  absorption_fraction: torax_pydantic.PositiveTimeVaryingScalar = (
      torax_pydantic.ValidatedDefault(1.0)
  )
  # TODO(b/434175938): Make minority_species a required field in V2.
  minority_species: Annotated[str | None, torax_pydantic.JAX_STATIC] = None

  def build_source(self) -> IonCyclotronSource:
    return IonCyclotronSource(model_func=self.model_func)

  def build_runtime_params(
      self,
      t: chex.Numeric,
  ) -> RuntimeParams:
    base_params = super().build_runtime_params(t)
    return RuntimeParams(
        **vars(base_params),
        P_total=self.P_total.get_value(t),
        absorption_fraction=self.absorption_fraction.get_value(t),
        minority_species=self.minority_species,
    )
