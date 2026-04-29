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

from collections.abc import Sequence
import dataclasses
from typing import Annotated, ClassVar

from jax import numpy as jnp
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import base as source_base
from torax._src.sources import runtime_params as source_runtime_params_lib
from torax._src.sources import source
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

# Default value for the model function to be used for the ion cyclotron
# source. This is also used as an identifier for the model function in
# the default source config for Pydantic to "discriminate" against.
DEFAULT_MODEL_FUNCTION_NAME: str = 'toric_nn'


def build_fast_ions(
    source_name: str,
    geo: geometry.Geometry,
    fast_ions: Sequence[fast_ion_lib.FastIon] = (),
) -> tuple[fast_ion_lib.FastIon, ...]:
  """Builds a complete FastIon tuple for all supported species.

  Takes a list of computed FastIon objects (for a subset of species) and
  produces a full tuple covering all species in
  ``fast_ion_lib.FAST_ION_SPECIES``.
  Species not present in the input list are filled with zero density and
  temperature.

  Args:
    source_name: The name of the source.
    geo: Geometry.
    fast_ions: Computed FastIon objects for a subset of species.

  Returns:
    Tuple of FastIon objects, one per species in
    ``fast_ion_lib.FAST_ION_SPECIES``, in order.
  """
  computed = {fi.species: fi for fi in fast_ions}
  zeros = jnp.zeros_like(geo.rho)
  result = []
  for species in fast_ion_lib.FAST_ION_SPECIES:
    if species in computed:
      result.append(computed[species])
    else:
      result.append(
          fast_ion_lib.FastIon(
              species=species,
              source=source_name,
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
      )
  return tuple(result)


@dataclasses.dataclass(kw_only=True, frozen=True, eq=False)
class IonCyclotronSource(source.Source):
  """Ion cyclotron source."""

  SOURCE_NAME: ClassVar[str] = 'icrh'
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
    return build_fast_ions(source_name=cls.SOURCE_NAME, geo=geo)


class IonCyclotronSourceConfig(source_base.SourceModelBase):
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
  mode: Annotated[source_runtime_params_lib.Mode, torax_pydantic.JAX_STATIC] = (
      source_runtime_params_lib.Mode.MODEL_BASED
  )
  # TODO(b/434175938): Make minority_species a required field in V2.
  minority_species: Annotated[str | None, torax_pydantic.JAX_STATIC] = None

  def build_source(self) -> IonCyclotronSource:
    return IonCyclotronSource(model_func=self.model_func)
