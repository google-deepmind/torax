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

"""Ion mixture model and impurity fractions model for plasma composition."""
import dataclasses
from typing import Annotated, Any, Literal
import chex
import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_validation_utils
from torax._src.torax_pydantic import torax_pydantic
from typing_extensions import Final

# pylint: disable=invalid-name
_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Represents a fixed mixture of ion species at a specific time.

  Information on ion names are not stored here, but rather as static attributes,
  to simplify JAX logic and performance in source functions for fusion power and
  radiation which are species-dependent.

  Attributes:
    fractions: Ion fractions for a time slice. Can be 1D (n_species,) for
      radially constant fractions, or 2D (n_species, n_grid) for radially
      varying fractions.
    A_avg: Average A of the mixture. Can be a scalar or 1D array (n_grid,).
    Z_override: Typically, the average Z is calculated according to the
      temperature dependent charge-state-distribution, or for low-Z cases by the
      atomic numbers of the ions assuming full ionization. If Z_override is
      provided, it is used instead for the average Z.
  """

  fractions: array_typing.FloatVector
  A_avg: array_typing.FloatScalar | array_typing.FloatVectorCell
  Z_override: array_typing.FloatScalar | None = None


class IonMixture(torax_pydantic.BaseModelFrozen):
  """Represents a mixture of ion species. The mixture can depend on time.

  Main use cases:
  1. Represent a bundled mixture of hydrogenic main ions (e.g. D and T)
  2. Represent a bundled impurity species where the avg charge state, mass,
    and radiation is consistent with each fractional concentration, and these
    quantities are then averaged over the mixture to represent a single impurity
    species in the transport equations for efficiency.

  Attributes:
    species: A dict mapping ion symbols (from ION_SYMBOLS) to their fractional
      concentration in the mixture. The fractions must sum to 1.
    Z_override: An optional override for the average charge (Z) of the mixture.
    A_override: An optional override for the average mass (A) of the mixture.
  """

  species: runtime_validation_utils.IonMapping
  Z_override: torax_pydantic.TimeVaryingScalar | None = None
  A_override: torax_pydantic.TimeVaryingScalar | None = None

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    """Builds a RuntimeParams object at a given time."""
    ions = self.species.keys()
    fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)

    if not self.A_override:
      As = jnp.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
      A_avg = jnp.sum(As * fractions)
    else:
      A_avg = self.A_override.get_value(t)

    return RuntimeParams(
        fractions=fractions,
        A_avg=A_avg,
        Z_override=Z_override,
    )


class ImpurityFractions(IonMixture):
  """Impurity content defined by fractional abundances."""

  impurity_mode: Annotated[Literal['fractions'], torax_pydantic.JAX_STATIC] = (
      'fractions'
  )
  # Default impurity setting. Parent class has species without a default.
  species: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault({'Ne': 1.0})
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_impurity_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Ensures backward compatibility if infered that data in legacy format."""

    # Maps legacy inputs to the new API format.
    # TODO(b/434175938): Remove this once V1 API is deprecated.
    if 'species' not in data and 'impurity_mode' not in data:
      return {'species': data, 'impurity_mode': _IMPURITY_MODE_FRACTIONS}
    return data
