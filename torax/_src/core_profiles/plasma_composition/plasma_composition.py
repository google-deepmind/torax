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

import dataclasses
import functools
import logging
from typing import Annotated, Final, Self
import chex
import jax
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import constants
from torax._src.config import runtime_validation_utils
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.core_profiles.plasma_composition import electron_density_ratios_zeff
from torax._src.core_profiles.plasma_composition import impurity_fractions
from torax._src.core_profiles.plasma_composition import ion_mixture
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name

# Constants for impurity modes.
_IMPURITY_MODE_FRACTIONS: Final[str] = 'fractions'
_IMPURITY_MODE_NE_RATIOS: Final[str] = 'n_e_ratios'
_IMPURITY_MODE_NE_RATIOS_ZEFF: Final[str] = 'n_e_ratios_Z_eff'


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class RuntimeParams:
  """Runtime parameters for the plasma composition."""

  main_ion_names: tuple[str, ...] = jax.tree.static()
  impurity_names: tuple[str, ...] = jax.tree.static()
  main_ion: ion_mixture.RuntimeParams
  impurity: (
      ion_mixture.RuntimeParams
      | electron_density_ratios.RuntimeParams
      | electron_density_ratios_zeff.RuntimeParams
  )
  Z_eff: array_typing.FloatVectorCell
  Z_eff_face: array_typing.FloatVectorFace


# TODO(b/440667088): Consider validation against duplicate species.
@jax.tree_util.register_pytree_node_class
class PlasmaComposition(torax_pydantic.ConfigBase):
  """Configuration for the plasma composition.

  The `from_dict(...)` method can accept a dictionary defined by
  https://torax.readthedocs.io/en/latest/configuration/plasma_composition.html.

  Attributes:
    main_ion: Specifies the main ion species:

      - If a string, it represents a single ion species (e.g., `'D'` for
        deuterium, `'T'` for tritium, `'H'` for hydrogen).
      - If a dict, it represents a mixture of ion species with given
        fractional concentrations (e.g., `{'D': 0.5, 'T': 0.5}`), which must
        sum to `1` within a tolerance of `1e-6`. The effective mass and charge
        of the mixture are the weighted averages of the species masses and
        charges. The fractions can be time-dependent (**time-varying-scalar**),
        supporting time-varying isotope ratios.
    impurity: Specifies the impurity species. How impurities are defined is set
      by the `impurity_mode` field within this dictionary (`'fractions'`,
      `'n_e_ratios'`, or `'n_e_ratios_Z_eff'`). Defaults to `{'impurity_mode':
      'fractions', 'species': {'Ne': 1.0}}`.
    Z_eff: Plasma effective charge, defined as $Z_eff = sum(n_i * Z_i^2, i) /
      n_e$. Required input when using the `'fractions'` or
      `'n_e_ratios_Z_eff'` impurity modes. When using the `'n_e_ratios'` mode,
      `Z_eff` is an emergent calculated quantity, and any user-provided `Z_eff`
      value is ignored (a warning is issued).
    Z_i_override: Optional override for the main ion's charge ($Z$) or average
      charge of an ion mixture. If provided, this value is used instead of the
      $Z$ calculated from `main_ion`.
    A_i_override: Optional override for the main ion's mass ($A$) [amu] or
      average mass of an ion mixture. If provided, this value is used instead of
      the $A$ calculated from `main_ion`.
  """  # fmt: skip

  impurity: Annotated[
      impurity_fractions.ImpurityFractions
      | electron_density_ratios.ElectronDensityRatios
      | electron_density_ratios_zeff.ElectronDensityRatiosZeff,
      pydantic.Field(discriminator='impurity_mode'),
  ] = torax_pydantic.ValidatedDefault(impurity_fractions.ImpurityFractions())
  main_ion: runtime_validation_utils.IonMapping = (
      torax_pydantic.ValidatedDefault({'D': 0.5, 'T': 0.5})
  )
  Z_eff: (
      runtime_validation_utils.TimeVaryingArrayDefinedAtRightBoundaryAndBounded
  ) = torax_pydantic.ValidatedDefault(1.0)
  Z_i_override: torax_pydantic.TimeVaryingScalar | None = None
  A_i_override: torax_pydantic.TimeVaryingScalar | None = None

  @pydantic.model_validator(mode='after')
  def _check_zeff_usage(self) -> Self:
    """Warns user if Z_eff is provided but will be ignored."""
    is_default_zeff = all(
        np.allclose(val, 1.0) for _, (_, val) in self.Z_eff.value.items()
    )
    if (
        isinstance(self.impurity, electron_density_ratios.ElectronDensityRatios)
        and not is_default_zeff
    ):
      logging.warning(
          "Z_eff is provided but impurity_mode is '%s'. Z_eff will be an"
          ' emergent quantity and the input value will be ignored.',
          _IMPURITY_MODE_NE_RATIOS,
      )
    if (
        isinstance(
            self.impurity,
            electron_density_ratios_zeff.ElectronDensityRatiosZeff,
        )
        and self.Z_i_override is None
        and self.impurity.Z_override is None
    ):
      unknown_species = next(
          symbol
          for symbol, ratio in self.impurity.species.items()
          if ratio is None
      )
      main_ion_symbols = set(self.main_ion.keys())
      all_hydrogenic = all(
          constants.ION_PROPERTIES_DICT[s].Z == 1
          for s in {unknown_species, *main_ion_symbols}
      )
      if main_ion_symbols == {unknown_species} or all_hydrogenic:
        raise ValueError(
            f"Impurity unknown species '{unknown_species}' in mode"
            f" '{_IMPURITY_MODE_NE_RATIOS_ZEFF}' cannot have the same atomic"
            f' charge Z as the main ion species {tuple(self.main_ion.keys())}:'
            ' quasineutrality and Z_eff equations become linearly dependent.'
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
    )
    obj._main_ion_mixture = children[5]  # pylint: disable=protected-access
    return obj

  @functools.cached_property
  def _main_ion_mixture(self) -> ion_mixture.IonMixture:
    """Returns the IonMixture object for the main ions."""
    # Use `model_construct` as no validation required.
    return ion_mixture.IonMixture.model_construct(
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

  def build_runtime_params(self, t: chex.Numeric) -> RuntimeParams:
    return RuntimeParams(
        main_ion_names=self.get_main_ion_names(),
        impurity_names=self.get_impurity_names(),
        main_ion=self._main_ion_mixture.build_runtime_params(t),
        impurity=self.impurity.build_runtime_params(t),  # pyrefly: ignore[bad-argument-type]
        Z_eff=self.Z_eff.get_value(t),
        Z_eff_face=self.Z_eff.get_value(t, grid_type='face'),
    )
