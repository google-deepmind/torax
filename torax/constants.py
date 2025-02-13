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

"""Physics constants.

This module saves immutable constants used in various calculations.
"""

from typing import Final, Mapping
import chex
import immutabledict
from jax import numpy as jnp

# pylint: disable=invalid-name


@chex.dataclass(frozen=True)
class IonProperties:
  """Properties of an ion.

  Attributes:
    symbol: The ion's symbol.
    name: The ion's full name.
    A: The ion's atomic mass unit (amu).
    Z: The ion's atomic number.
  """

  symbol: str
  name: str
  A: float
  Z: float


@chex.dataclass(frozen=True)
class Constants:
  keV2J: chex.Numeric  # pylint: disable=invalid-name
  mp: chex.Numeric
  qe: chex.Numeric
  me: chex.Numeric
  epsilon0: chex.Numeric
  mu0: chex.Numeric
  eps: chex.Numeric
  c: chex.Numeric


CONSTANTS: Final[Constants] = Constants(
    keV2J=1e3 * 1.6e-19,
    mp=1.67e-27,
    qe=1.6e-19,
    me=9.11e-31,
    epsilon0=8.854e-12,
    mu0=4 * jnp.pi * 1e-7,
    eps=1e-7,
    c=2.99792458e8,
)

# Taken from
# https://www.nist.gov/pml/periodic-table-elements and https://ciaaw.org.
ION_PROPERTIES: Final[tuple[IonProperties, ...]] = (
    IonProperties(symbol='H', name='Hydrogen', A=1.008, Z=1.0),
    IonProperties(symbol='D', name='Deuterium', A=2.0141, Z=1.0),
    IonProperties(symbol='T', name='Tritium', A=3.0160, Z=1.0),
    IonProperties(symbol='He3', name='Helium-3', A=3.0160, Z=2.0),
    IonProperties(symbol='He4', name='Helium-4', A=4.0026, Z=2.0),
    IonProperties(symbol='Li', name='Lithium', A=5.3917, Z=3.0),
    IonProperties(symbol='Be', name='Beryllium', A=9.0122, Z=4.0),
    IonProperties(symbol='C', name='Carbon', A=12.011, Z=6.0),
    IonProperties(symbol='N', name='Nitrogen', A=14.007, Z=7.0),
    IonProperties(symbol='O', name='Oxygen', A=15.999, Z=8.0),
    IonProperties(symbol='Ne', name='Neon', A=20.180, Z=10.0),
    IonProperties(symbol='Ar', name='Argon', A=39.95, Z=18.0),
    IonProperties(symbol='Kr', name='Krypton', A=83.798, Z=36.0),
    IonProperties(symbol='Xe', name='Xenon', A=131.29, Z=54.0),
    IonProperties(symbol='W', name='Tungsten', A=183.84, Z=74.0),
)

ION_PROPERTIES_DICT: Final[Mapping[str, IonProperties]] = (
    immutabledict.immutabledict({v.symbol: v for v in ION_PROPERTIES})
)

ION_SYMBOLS = frozenset(ION_PROPERTIES_DICT.keys())
