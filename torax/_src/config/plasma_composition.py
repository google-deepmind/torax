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

"""Plasma composition parameters used throughout TORAX simulations."""
import functools

import chex
import numpy as np
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_validation_utils
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class DynamicIonMixture:
  """Represents a fixed mixture of ion species at a specific time.

  Information on ion names are not stored here, but rather in
  StaticRuntimeParamsSlice, to simplify JAX logic and performance in source
  functions for fusion power and radiation which are species-dependent.

  Attributes:
    fractions: Ion fractions for a time slice.
    avg_A: Average A of the mixture.
    Z_override: Typically, the average Z is calculated according to the
      temperature dependent charge-state-distribution, or for low-Z cases by the
      atomic numbers of the ions assuming full ionization. If Z_override is
      provided, it is used instead for the average Z.
  """

  fractions: array_typing.ArrayFloat
  avg_A: array_typing.ScalarFloat
  Z_override: array_typing.ScalarFloat | None = None


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

  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicIonMixture:
    """Creates a DynamicIonMixture object at a given time.

    Optional overrides for Z and A can be provided.

    Args:
      t: The time at which to build the DynamicIonMixture.

    Returns:
      A DynamicIonMixture object.
    """
    ions = self.species.keys()
    fractions = np.array([self.species[ion].get_value(t) for ion in ions])
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)

    if not self.A_override:
      As = np.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
      avg_A = np.sum(As * fractions)
    else:
      avg_A = self.A_override.get_value(t)

    return DynamicIonMixture(
        fractions=fractions,
        avg_A=avg_A,
        Z_override=Z_override,
    )


@chex.dataclass
class DynamicPlasmaComposition:
  main_ion: DynamicIonMixture
  impurity: DynamicIonMixture
  Z_eff: array_typing.ArrayFloat
  Z_eff_face: array_typing.ArrayFloat


class PlasmaComposition(torax_pydantic.BaseModelFrozen):
  """Configuration for the plasma composition.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration.html#plasma-composition.

  Attributes:
    main_ion: Main ion species. Can be single ion or a mixture of ions (e.g. D
      and T). Either a single ion, and constant mixture, or a time-dependent
      mixture. For single ions the input is one of the allowed strings in
      `ION_SYMBOLS`. For mixtures the input is an IonMixture object, constructed
      from a dict mapping ion symbols to their fractional concentration in the
      mixture.
    impurity: Impurity ion species. Same format as main_ion.
    Z_eff: Constraint for impurity densities.
    Z_i_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    A_i_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    Z_impurity_override: Optional arbitrary masses and charges which can
  """

  main_ion: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault({'D': 0.5, 'T': 0.5})
  )
  impurity: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault('Ne')
  )
  Z_eff: (
      runtime_validation_utils.TimeVaryingArrayDefinedAtRightBoundaryAndBounded
  ) = torax_pydantic.ValidatedDefault(1.0)
  Z_i_override: torax_pydantic.TimeVaryingScalar | None = None
  A_i_override: torax_pydantic.TimeVaryingScalar | None = None
  Z_impurity_override: torax_pydantic.TimeVaryingScalar | None = None
  A_impurity_override: torax_pydantic.TimeVaryingScalar | None = None

  # Generate the IonMixture objects from the input for either a mixture (dict)
  # or the shortcut for a single ion (string). IonMixture objects with a
  # single key and fraction=1.0 is used also for the single ion case to reduce
  # code duplication.

  @functools.cached_property
  def main_ion_mixture(self) -> IonMixture:
    """Returns the IonMixture object for the main ions."""
    # Use `model_construct` as no validation required.
    return IonMixture.model_construct(
        species=self.main_ion,
        Z_override=self.Z_i_override,
        A_override=self.A_i_override,
    )

  @functools.cached_property
  def impurity_mixture(self) -> IonMixture:
    """Returns the IonMixture object for the impurity ions."""
    # Use `model_construct` as no validation required.
    return IonMixture.model_construct(
        species=self.impurity,
        Z_override=self.Z_impurity_override,
        A_override=self.A_impurity_override,
    )

  def get_main_ion_names(self) -> tuple[str, ...]:
    """Returns the main ion symbol strings from the input."""
    return tuple(self.main_ion_mixture.species.keys())

  def get_impurity_names(self) -> tuple[str, ...]:
    """Returns the impurity symbol strings from the input."""
    return tuple(self.impurity_mixture.species.keys())

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(
        main_ion=self.main_ion_mixture.build_dynamic_params(t),
        impurity=self.impurity_mixture.build_dynamic_params(t),
        Z_eff=self.Z_eff.get_value(t),
        Z_eff_face=self.Z_eff.get_value(t, grid_type='face'),
    )
