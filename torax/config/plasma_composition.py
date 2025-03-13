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

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import logging

import chex
import numpy as np
import pydantic
from torax import array_typing
from torax import constants
from torax import interpolated_param
from torax.config import base
from torax.config import config_args
from torax.torax_pydantic import torax_pydantic
from typing_extensions import Self

# pylint: disable=invalid-name


class PlasmaCompositionPydantic(torax_pydantic.BaseModelFrozen):
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
    Zeff: Constraint for impurity densities.
    Zi_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    Ai_override: Optional arbitrary masses and charges which can be used to
      override the data for the average Z and A of each IonMixture for main_ions
      or impurities. Useful for testing or testing physical sensitivities,
      outside the constraint of allowed impurity species.
    Zimp_override: Optional arbitrary masses and charges which can
  """

  main_ion: str | Mapping[str, torax_pydantic.TimeVaryingScalar] = (
      pydantic.Field(
          default_factory=lambda: {'D': 0.5, 'T': 0.5}, validate_default=True
      )
  )
  impurity: str | Mapping[str, torax_pydantic.TimeVaryingScalar] = 'Ne'
  Zeff: torax_pydantic.TimeVaryingArray = torax_pydantic.ValidatedDefault(1.0)
  Zi_override: torax_pydantic.TimeVaryingScalar | None = None
  Ai_override: torax_pydantic.TimeVaryingScalar | None = None
  Zimp_override: torax_pydantic.TimeVaryingScalar | None = None
  Aimp_override: torax_pydantic.TimeVaryingScalar | None = None

  @pydantic.model_validator(mode='after')
  def after_validator(self) -> Self:
    if not self.Zeff.right_boundary_conditions_defined:
      logging.debug("""
          Config input Zeff not directly defined at rhonorm=1.0.
          Zeff_face at rhonorm=1.0 set from constant values or constant extrapolation.
          """)
    return self


@chex.dataclass(frozen=True)
class IonMixture:
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
    tolerance: The tolerance used to check if the fractions sum to 1
    Z_override: An optional override for the average charge (Z) of the mixture.
    A_override: An optional override for the average mass (A) of the mixture.
  """

  species: Mapping[str, interpolated_param.TimeInterpolatedInput]
  tolerance: float = 1e-6
  Z_override: interpolated_param.TimeInterpolatedInput | None = None
  A_override: interpolated_param.TimeInterpolatedInput | None = None

  def make_provider(self) -> IonMixtureProvider:
    """Creates an IonMixtureProvider."""
    ion_fractions = {
        k: config_args.get_interpolated_var_single_axis(v)
        for k, v in self.species.items()
    }
    if self.Z_override:
      Z_override = config_args.get_interpolated_var_single_axis(self.Z_override)
    else:
      Z_override = None
    if self.A_override:
      A_override = config_args.get_interpolated_var_single_axis(self.A_override)
    else:
      A_override = None
    return IonMixtureProvider(
        ion_fractions=ion_fractions,
        Z_override=Z_override,
        A_override=A_override,
    )

  @classmethod
  def from_config(
      cls,
      species_input: (
          str | Mapping[str, interpolated_param.TimeInterpolatedInput]
      ),
      Z_override: interpolated_param.TimeInterpolatedInput | None = None,
      A_override: interpolated_param.TimeInterpolatedInput | None = None,
  ) -> IonMixture:
    """Constructs an IonMixture instance.

    Args:
      species_input: The input for the IonMixture. Either a string (shortcut for
        a mixture with a single ion) or a Mapping for a mixture of ions with
        different fractional concentrations.
      Z_override: Optional override for the average charge (Z) of the mixture.
      A_override: Optional override for the average mass (A) of the mixture.

    Returns:
      An IonMixture instance.
    """
    if isinstance(species_input, str):
      return cls(
          species={species_input: 1.0},
          Z_override=Z_override,
          A_override=A_override,
      )
    elif isinstance(species_input, Mapping):
      return cls(
          species=species_input, Z_override=Z_override, A_override=A_override
      )
    else:
      raise TypeError(
          'Expected a string (for a single ion) or a Mapping for IonMixture,'
          f' got: {type(species_input)}'
      )

  def __post_init__(self):

    if not self.species:
      raise ValueError(self.__class__.__name__ + ' species cannot be empty.')

    if not isinstance(self.species, Mapping):
      raise ValueError('species must be a Mapping')

    # Check if all species keys are in the allowed list.
    invalid_ion_symbols = set(self.species.keys()) - constants.ION_SYMBOLS
    if invalid_ion_symbols:
      raise ValueError(
          f'Invalid ion symbols: {invalid_ion_symbols}. Allowed symbols are:'
          f' {constants.ION_SYMBOLS}'
      )

    time_arrays = []
    fraction_arrays = []

    for value in self.species.values():
      time_array, fraction_array, _, _ = (
          interpolated_param.convert_input_to_xs_ys(value)
      )
      time_arrays.append(time_array)
      fraction_arrays.append(fraction_array)

    # Check if all time arrays are equal
    if not all(np.array_equal(time_arrays[0], x) for x in time_arrays[1:]):
      raise ValueError(
          'All time indexes for '
          + self.__class__.__name__
          + ' fractions must be equal.'
      )

    # Check if the ion fractions sum to 1 at all times
    fraction_sum = np.sum(fraction_arrays, axis=0)
    if not np.allclose(fraction_sum, 1.0, rtol=self.tolerance):
      raise ValueError(
          'Fractional concentrations in an IonMixture must sum to 1 at all'
          ' times.'
      )


@chex.dataclass
class IonMixtureProvider:
  """Creates DynamicIonMixture objects at a given time."""

  ion_fractions: Mapping[str, interpolated_param.InterpolatedVarSingleAxis]
  Z_override: interpolated_param.InterpolatedVarSingleAxis | None = None
  A_override: interpolated_param.InterpolatedVarSingleAxis | None = None

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
    ions = self.ion_fractions.keys()
    fractions = np.array([self.ion_fractions[ion].get_value(t) for ion in ions])

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


@chex.dataclass
class PlasmaComposition(
    base.RuntimeParametersConfig['PlasmaCompositionProvider']
):
  """Configuration for the plasma composition.

  List of allowed ion species is found in constants.ION_SYMBOLS.
  """

  # Main ion species. Can be single ion or a mixture of ions (e.g. D and T)
  # Either a single ion, and constant mixture, or a time-dependent mixture.
  # For single ions the input is one of the allowed strings in ION_SYMBOLS.
  # For mixtures the input is an IonMixture object, constructed from a dict
  # mapping ion symbols to their fractional concentration in the mixture.
  main_ion: str | Mapping[str, interpolated_param.TimeInterpolatedInput] = (
      dataclasses.field(default_factory=lambda: {'D': 0.5, 'T': 0.5})
  )

  # Impurity ion species. Same format as main_ion.
  impurity: str | Mapping[str, interpolated_param.TimeInterpolatedInput] = (
      dataclasses.field(default_factory=lambda: 'Ne')
  )

  # Constraint for impurity densities.
  Zeff: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: 1.0
  )

  # Optional arbitrary masses and charges which can be used to override the data
  # for the average Z and A of each IonMixture for main_ions or impurities.
  # Useful for testing or testing physical sensitivities, outside the constraint
  # of allowed impurity species.

  Zi_override: interpolated_param.TimeInterpolatedInput | None = (
      dataclasses.field(default_factory=lambda: None)
  )
  Ai_override: interpolated_param.TimeInterpolatedInput | None = (
      dataclasses.field(default_factory=lambda: None)
  )

  Zimp_override: interpolated_param.TimeInterpolatedInput | None = (
      dataclasses.field(default_factory=lambda: None)
  )
  Aimp_override: interpolated_param.TimeInterpolatedInput | None = (
      dataclasses.field(default_factory=lambda: None)
  )

  # IonMixture instances created in __post_init__ from input.
  main_ion_mixture: IonMixture = dataclasses.field(init=False)
  impurity_mixture: IonMixture = dataclasses.field(init=False)

  def make_provider(
      self,
      torax_mesh: torax_pydantic.Grid1D | None = None,
  ) -> PlasmaCompositionProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required to make a PlasmaCompositionProvider'
      )

    return PlasmaCompositionProvider(
        runtime_params_config=self,
        main_ion_provider=self.main_ion_mixture.make_provider(),
        impurity_provider=self.impurity_mixture.make_provider(),
        Zeff=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.cell_centers,
        ),
        Zeff_face=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.face_centers,
        ),
    )

  def get_main_ion_names(self) -> tuple[str, ...]:
    """Returns the main ion symbol strings from the input."""
    return tuple(self.main_ion_mixture.species.keys())

  def get_impurity_names(self) -> tuple[str, ...]:
    """Returns the impurity symbol strings from the input."""
    return tuple(self.impurity_mixture.species.keys())

  def __post_init__(self):
    # Generate the IonMixture objects from the input for either a mixture (dict)
    # or the shortcut for a single ion (string). IonMixture objects with a
    # single key and fraction=1.0 is used also for the single ion case to reduce
    # code duplication.
    self.main_ion_mixture = IonMixture.from_config(
        self.main_ion, self.Zi_override, self.Ai_override
    )
    self.impurity_mixture = IonMixture.from_config(
        self.impurity, self.Zimp_override, self.Aimp_override
    )

    if not interpolated_param.rhonorm1_defined_in_timerhoinput(self.Zeff):
      logging.debug("""
          Config input Zeff not directly defined at rhonorm=1.0.
          Zeff_face at rhonorm=1.0 set from constant values or constant extrapolation.
          """)


@chex.dataclass
class PlasmaCompositionProvider(
    base.RuntimeParametersProvider['DynamicPlasmaComposition']
):
  """Prepared plasma composition."""

  runtime_params_config: PlasmaComposition
  main_ion_provider: IonMixtureProvider
  impurity_provider: IonMixtureProvider
  Zeff: interpolated_param.InterpolatedVarTimeRho
  Zeff_face: interpolated_param.InterpolatedVarTimeRho

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(
        main_ion=self.main_ion_provider.build_dynamic_params(t),
        impurity=self.impurity_provider.build_dynamic_params(t),
        Zeff=self.Zeff.get_value(t),
        Zeff_face=self.Zeff_face.get_value(t),
    )


@chex.dataclass
class DynamicPlasmaComposition:
  main_ion: DynamicIonMixture
  impurity: DynamicIonMixture
  Zeff: array_typing.ArrayFloat
  Zeff_face: array_typing.ArrayFloat
