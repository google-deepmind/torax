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
from torax import array_typing
from torax import constants
from torax import interpolated_param
from torax.config import base
from torax.config import config_args
from torax.geometry import geometry


# pylint: disable=invalid-name
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
    tolerance: The tolerance used to check if the fractions sum to 1.
  """

  species: Mapping[
      constants.ION_SYMBOLS, interpolated_param.TimeInterpolatedInput
  ]
  tolerance: float = 1e-6

  def __post_init__(self):

    if not self.species:
      raise ValueError(self.__class__.__name__ + ' species cannot be empty.')

    if not isinstance(self.species, Mapping):
      raise ValueError('species must be a Mapping')

    time_arrays = []
    fraction_arrays = []

    for value in self.species.values():
      time_array, fraction_array, _, _ = (
          interpolated_param.convert_input_to_xs_ys(value)
      )
      time_arrays.append(time_array)
      fraction_arrays.append(fraction_array)

    # Check if all time arrays are equal
    # Note that if the TimeInterpolatedInput is a constant fraction (float) then
    # convert_input_to_xs_ys returns a single-element array for t with value=0
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
class PlasmaComposition(
    base.RuntimeParametersConfig['PlasmaCompositionProvider']
):
  """Configuration for the plasma composition."""

  # amu of main ion (if multiple isotope, make average)
  Ai: float = 2.51505
  # charge of main ion
  Zi: float = 1.0
  # needed for qlknn and fusion power
  Zeff: interpolated_param.InterpolatedVarTimeRhoInput = dataclasses.field(
      default_factory=lambda: 1.0
  )
  Zimp: interpolated_param.TimeInterpolatedInput = (
      10.0  # impurity charge state assumed for dilution
  )
  Aimp: float = 20.18  # impurity mass in amu

  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> PlasmaCompositionProvider:
    if torax_mesh is None:
      raise ValueError(
          'torax_mesh is required to make a PlasmaCompositionProvider'
      )
    return PlasmaCompositionProvider(
        runtime_params_config=self,
        Zeff=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.cell_centers,
        ),
        Zeff_face=config_args.get_interpolated_var_2d(
            self.Zeff,
            torax_mesh.face_centers,
        ),
        Zimp=config_args.get_interpolated_var_single_axis(self.Zimp),
    )

  def __post_init__(self):
    if not interpolated_param.rhonorm1_defined_in_timerhoinput(self.Zeff):
      logging.info("""
          Config input Zeff not directly defined at rhonorm=1.0.
          Zeff_face at rhonorm=1.0 set from constant values or constant extrapolation.
          """)


@chex.dataclass
class PlasmaCompositionProvider(
    base.RuntimeParametersProvider['DynamicPlasmaComposition']
):
  """Prepared plasma composition."""

  runtime_params_config: PlasmaComposition
  Zeff: interpolated_param.InterpolatedVarTimeRho
  Zeff_face: interpolated_param.InterpolatedVarTimeRho
  Zimp: interpolated_param.InterpolatedVarSingleAxis

  def build_dynamic_params(self, t: chex.Numeric) -> DynamicPlasmaComposition:
    return DynamicPlasmaComposition(**self.get_dynamic_params_kwargs(t))


@chex.dataclass
class DynamicPlasmaComposition:
  Ai: float
  Zi: float
  Zeff: array_typing.ArrayFloat
  Zeff_face: array_typing.ArrayFloat
  Zimp: array_typing.ScalarFloat
  Aimp: float
