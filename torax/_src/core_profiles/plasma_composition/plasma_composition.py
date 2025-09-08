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
import copy
import dataclasses
import functools
import logging
from typing import Annotated, Any, Literal, Mapping
import chex
import jax
from jax import numpy as jnp
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_validation_utils
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions
from typing_extensions import Final

# pylint: disable=invalid-name

# Constants for impurity modes.
_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'
_IMPURITY_MODE_NE_RATIOS: Final[str] = 'n_e_ratios'
_IMPURITY_MODE_NE_RATIOS_ZEFF: Final[str] = 'n_e_ratios_Z_eff'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicIonMixture:
  """Represents a fixed mixture of ion species at a specific time.

  Information on ion names are not stored here, but rather in
  StaticRuntimeParamsSlice, to simplify JAX logic and performance in source
  functions for fusion power and radiation which are species-dependent.

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class DynamicNeRatiosZeff:
  """Analogous to DynamicImpurityFractions but for n_e_ratio_Z_eff inputs."""

  n_e_ratios: Mapping[str, array_typing.FloatScalar | None]
  unknown_species: str = dataclasses.field(metadata={'static': True})
  Z_override: array_typing.FloatScalar | None = None
  A_override: array_typing.FloatScalar | None = None


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
    fractions = jnp.array([self.species[ion].get_value(t) for ion in ions])
    Z_override = None if not self.Z_override else self.Z_override.get_value(t)

    if not self.A_override:
      As = jnp.array([constants.ION_PROPERTIES_DICT[ion].A for ion in ions])
      A_avg = jnp.sum(As * fractions)
    else:
      A_avg = self.A_override.get_value(t)

    return DynamicIonMixture(
        fractions=fractions,
        A_avg=A_avg,
        Z_override=Z_override,
    )


class ImpurityFractionsModel(IonMixture):
  """Impurity content defined by fractional abundances."""

  impurity_mode: Annotated[Literal['fractions'], torax_pydantic.JAX_STATIC] = (
      'fractions'
  )
  # Default impurity setting. Parent class has species without a default.
  species: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault({'Ne': 1.0})
  )

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicIonMixture:
    # Call the parent IonMixture's builder
    dynamic_impurity_mixture = super().build_dynamic_params(t)
    # Use the result to construct the specialized DynamicFractions dataclass
    return DynamicIonMixture(
        fractions=dynamic_impurity_mixture.fractions,
        A_avg=dynamic_impurity_mixture.A_avg,
        Z_override=dynamic_impurity_mixture.Z_override,
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


class NeRatiosZeffModel(torax_pydantic.BaseModelFrozen):
  """Impurity content defined by ratios, with one species constrained by Z_eff."""

  # Exactly one species must have a None ratio to be constrained by Z_eff.
  species: Mapping[str, torax_pydantic.NonNegativeTimeVaryingScalar | None]
  Z_override: torax_pydantic.TimeVaryingScalar | None = None
  A_override: torax_pydantic.TimeVaryingScalar | None = None
  impurity_mode: Annotated[
      Literal['n_e_ratios_Z_eff'], torax_pydantic.JAX_STATIC
  ] = 'n_e_ratios_Z_eff'

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicNeRatiosZeff:
    unknown_species = next(
        (symbol for symbol, ratio in self.species.items() if ratio is None),
        None,
    )
    # The validator ensures unknown_species is not None but add an extra check.
    assert unknown_species is not None
    return DynamicNeRatiosZeff(
        n_e_ratios={
            symbol: ratio.get_value(t) if ratio is not None else None
            for symbol, ratio in self.species.items()
        },
        unknown_species=unknown_species,
        Z_override=self.Z_override.get_value(t) if self.Z_override else None,
        A_override=self.A_override.get_value(t) if self.A_override else None,
    )

  @pydantic.model_validator(mode='after')
  def _validate_one_none(self) -> typing_extensions.Self:
    if not self.species:
      raise ValueError('The species dictionary cannot be empty.')
    none_count = sum(v is None for v in self.species.values())
    if none_count != 1:
      raise ValueError(
          'Exactly one impurity must have a `None` ratio to be'
          ' constrained by Z_eff.'
      )
    return self


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DynamicPlasmaComposition:
  main_ion_names: tuple[str, ...] = dataclasses.field(metadata={'static': True})
  impurity_names: tuple[str, ...] = dataclasses.field(metadata={'static': True})
  main_ion: DynamicIonMixture
  impurity: (
      DynamicIonMixture
      | electron_density_ratios.RuntimeParams
      | DynamicNeRatiosZeff
  )
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace


# TODO(b/440667088): Consider validation against duplicate species.
@jax.tree_util.register_pytree_node_class
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
    impurity: Dictionary configuring plasma impurities with the following keys:
      `impurity_mode`: Sets how the impurity species are defined. `species`: A
      dictionary mapping ion symbols (e.g., 'Ne', 'W') to their respective
      values. The interpretation of these values depends on the `impurity_mode`
      as follows. * `'fractions'`: relative fractional abundances. *
      `'n_e_ratios'`: ratios of impurity density to electron density. Z_eff is
      ignored in this mode. * `'n_e_ratios_Z_eff'`: ratios of impurity density
      to electron density. A single value must provided as None, and Z_eff is
      used to then constrain this value dynamically during runtime.
      `Z_override`: Optional. Overrides the calculated impurity average charge
      `A_override`: Optional. Overrides the calculated average impurity mass
      Backwards compatibility is provided for legacy inputs to `'impurity'`,
      e.g. string or dict inputs similar to `main_ion`, such as `'Ar'` or
      `{'Ar': 0.6, 'Ne': 0.4}`.
    Z_eff: Constraint for impurity densities. If the impurity_mode is
      `'n_e_ratios'`, and the input is not None, then any input value will be
      ignored and a warning provided to the user.
    Z_i_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    A_i_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    Z_impurity_override: DEPRECATED. As Z_i_override, but for the impurities.
    A_impurity_override: DEPRECATED. As A_i_override, but for the impurities.
  """

  impurity: Annotated[
      ImpurityFractionsModel
      | electron_density_ratios.ELectronDensityRatios
      | NeRatiosZeffModel,
      pydantic.Field(discriminator='impurity_mode'),
  ]
  main_ion: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault({'D': 0.5, 'T': 0.5})
  )
  Z_eff: (
      runtime_validation_utils.TimeVaryingArrayDefinedAtRightBoundaryAndBounded
  ) = torax_pydantic.ValidatedDefault(1.0)
  Z_i_override: torax_pydantic.TimeVaryingScalar | None = None
  A_i_override: torax_pydantic.TimeVaryingScalar | None = None
  Z_impurity_override: torax_pydantic.TimeVaryingScalar | None = None
  A_impurity_override: torax_pydantic.TimeVaryingScalar | None = None

  # For main_ions, IonMixture objects are generated by either a fractional
  # mixture (dict[str, TimeVaryingScalar]) or the shortcut for a single constant
  # ion (string).
  # For impurities, this input is legacy but still supported. A new API is also
  # available with different impurity_modes, e.g. fractions or n_e_ratios.
  # A pydantic before validator infers the API format and handles conversions.

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_impurity_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    """Sets defaults and ensures backward compatibility for impurity inputs."""
    configurable_data = copy.deepcopy(data)

    Z_impurity_override = configurable_data.get('Z_impurity_override')
    A_impurity_override = configurable_data.get('A_impurity_override')

    # Set defaults for impurity if not specified. To maintain same default
    # behaviour as before, the top-level Z_impurity_override and
    # A_impurity_override are used as overrides for the impurity fractions.
    # TODO(b/434175938): Remove this once V1 API is deprecated and the top-level
    # overrides are removed, and set default directly in class attribute.
    if 'impurity' not in configurable_data:
      configurable_data['impurity'] = {
          'impurity_mode': _IMPURITY_MODE_FRACTIONS,
          'Z_override': Z_impurity_override,
          'A_override': A_impurity_override,
      }
      return configurable_data

    impurity_data = configurable_data['impurity']

    # New API format: impurity_mode is specified.
    if isinstance(impurity_data, dict) and 'impurity_mode' in impurity_data:
      if Z_impurity_override is not None or A_impurity_override is not None:
        logging.warning(
            'Z_impurity_override and/or A_impurity_override are set at the'
            ' plasma_composition level, but the new impurity API is being used'
            ' (impurity_mode is set). These top-level overrides are deprecated'
            ' and will be ignored. Use Z_override and A_override within the'
            ' impurity dictionary instead.'
        )
      return configurable_data

    # Legacy format from here on.
    # This handles conformant V1 inputs like 'Ne' or {'Ne': 0.8, 'Ar': 0.2}.
    # Non-conformant inputs are caught by ImpurityFractionsModel validation.
    # TODO(b/434175938): Remove this once V1 API is deprecated.
    configurable_data['impurity'] = {
        'impurity_mode': _IMPURITY_MODE_FRACTIONS,
        'species': impurity_data,
        'Z_override': Z_impurity_override,
        'A_override': A_impurity_override,
    }
    return configurable_data

  @pydantic.model_validator(mode='after')
  def _check_zeff_usage(self) -> typing_extensions.Self:
    """Warns user if Z_eff is provided but will be ignored."""
    if (
        isinstance(self.impurity, electron_density_ratios.ELectronDensityRatios)
        and self.Z_eff.value != 1.0  # default value if input Z_eff is None
    ):
      logging.warning(
          "Z_eff is provided but impurity_mode is '%s'. Z_eff will be an"
          ' emergent quantity and the input value will be ignored.',
          _IMPURITY_MODE_NE_RATIOS,
      )
    return self

  def tree_flatten(self):
    # Override the default tree_flatten to also save out the cached
    # main_ion_mixture and impurity_mixture objects.
    children = (
        self.main_ion,
        self.impurity,
        self.Z_eff,
        self.Z_i_override,
        self.A_i_override,
        self.Z_impurity_override,
        self.A_impurity_override,
        self._main_ion_mixture,
    )
    aux_data = ()
    return children, aux_data

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    obj = cls.model_construct(
        main_ion=children[0],
        impurity=children[1],
        Z_eff=children[2],
        Z_i_override=children[3],
        A_i_override=children[4],
        Z_impurity_override=children[5],
        A_impurity_override=children[6],
    )
    obj._main_ion_mixture = children[7]  # pylint: disable=protected-access
    return obj

  @functools.cached_property
  def _main_ion_mixture(self) -> IonMixture:
    """Returns the IonMixture object for the main ions."""
    # Use `model_construct` as no validation required.
    return IonMixture.model_construct(
        species=self.main_ion,
        Z_override=self.Z_i_override,
        A_override=self.A_i_override,
    )

  def get_main_ion_names(self) -> tuple[str, ...]:
    """Returns the main ion symbol strings from the input."""
    return tuple(self._main_ion_mixture.species.keys())

  def get_impurity_names(self) -> tuple[str, ...]:
    """Returns the impurity symbol strings from the input."""
    return tuple(self.impurity.species.keys())

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(
        main_ion_names=self.get_main_ion_names(),
        impurity_names=self.get_impurity_names(),
        main_ion=self._main_ion_mixture.build_dynamic_params(t),
        impurity=self.impurity.build_dynamic_params(t),
        Z_eff=self.Z_eff.get_value(t),
        Z_eff_face=self.Z_eff.get_value(t, grid_type='face'),
    )
