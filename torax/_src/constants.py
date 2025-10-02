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
import dataclasses
from typing import Final, Mapping

import chex
import immutabledict
import jax
from jax import numpy as jnp

# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Constants:
  """Physical constants.

  Attributes:
    keV_to_J: Conversion factor from keV to J.
    eV_to_J: Conversion factor from eV to J.
    m_amu: Atomic mass unit in kg, defined as 1/12 the mass of a C12 nucleus.
    q_e: Elementary charge in Coulombs.
    m_e: Electron mass in kg.
    epsilon_0: Vacuum permittivity in Henry per meter (H/m).
    mu_0: Vacuum permeability in N/A^2.
    k_B: Boltzman constant in J/K.
    eps: A small epsilon value used for numerical stability.
  """
  keV_to_J: chex.Numeric
  eV_to_J: chex.Numeric
  m_amu: chex.Numeric
  q_e: chex.Numeric
  m_e: chex.Numeric
  epsilon_0: chex.Numeric
  mu_0: chex.Numeric
  k_B: chex.Numeric
  eps: chex.Numeric


CONSTANTS: Final[Constants] = Constants(
    keV_to_J=1e3 * 1.602176634e-19,
    eV_to_J=1.602176634e-19,
    m_amu=1.6605390666e-27,
    q_e=1.602176634e-19,
    m_e=9.1093837e-31,
    epsilon_0=8.85418782e-12,
    mu_0=4 * jnp.pi * 1e-7,
    k_B=1.380649e-23,
    eps=1e-7,
)

# In amu. Taken from
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
