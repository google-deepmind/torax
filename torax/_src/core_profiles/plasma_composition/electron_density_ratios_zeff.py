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

"""Impurity content defined by ratios, with one species constrained by Z_eff."""
import dataclasses
from typing import Annotated, Literal, Mapping
import chex
import jax
import pydantic
from torax._src import array_typing
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RuntimeParams:
  """Runtime parameters for ElectronDensityRatiosZeff."""

  n_e_ratios: Mapping[str, array_typing.FloatScalar | None]
  unknown_species: str = dataclasses.field(metadata={'static': True})
  Z_override: array_typing.FloatScalar | None = None
  A_override: array_typing.FloatScalar | None = None


class ElectronDensityRatiosZeff(torax_pydantic.BaseModelFrozen):
  """Impurity content defined by ratios, with one species constrained by Z_eff."""

  # Exactly one species must have a None ratio to be constrained by Z_eff.
  species: Mapping[str, torax_pydantic.NonNegativeTimeVaryingScalar | None]
  Z_override: torax_pydantic.TimeVaryingScalar | None = None
  A_override: torax_pydantic.TimeVaryingScalar | None = None
  impurity_mode: Annotated[
      Literal['n_e_ratios_Z_eff'], torax_pydantic.JAX_STATIC
  ] = 'n_e_ratios_Z_eff'

  def build_dynamic_params(self, t: chex.Numeric) -> RuntimeParams:
    unknown_species = next(
        (symbol for symbol, ratio in self.species.items() if ratio is None),
        None,
    )
    # The validator ensures unknown_species is not None but add an extra check.
    assert unknown_species is not None
    return RuntimeParams(
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
