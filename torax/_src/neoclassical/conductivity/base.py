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

"""Base class for conductivity models."""

from __future__ import annotations

import abc
from typing import Annotated

import pydantic
from torax._src import state
from torax._src.core_profiles import solver_profiles
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.conductivity import runtime_params as conductivity_runtime_params
from torax._src.neoclassical.formulas import formulas
from torax._src.torax_pydantic import torax_pydantic


class ConductivityModel(abc.ABC):
  """Base class for conductivity models."""

  @abc.abstractmethod
  def calculate_conductivity(
      self,
      runtime_params: conductivity_runtime_params.RuntimeParams,
      geometry: geometry_lib.Geometry,
      core_profiles: solver_profiles.SolverProfiles,
      neoclassical_intermediates: formulas.AnalyticalNeoclassicalIntermediates,
  ) -> state.Conductivity:
    """Calculates conductivity."""


class ConductivityModelConfig(torax_pydantic.ConfigBase):
  """Base class for conductivity model configs.

  Attributes:
    conductivity_multiplier: Multiplication factor for the conductivity.
    neoclassical_correction: If True, applies the neoclassical trapped-particle
      correction to the Spitzer conductivity. If False, the Spitzer conductivity
      is used.
  """

  conductivity_multiplier: pydantic.PositiveFloat = 1.0
  neoclassical_correction: Annotated[bool, torax_pydantic.JAX_STATIC] = True

  def build_runtime_params(self) -> conductivity_runtime_params.RuntimeParams:
    """Builds runtime params."""
    return conductivity_runtime_params.RuntimeParams(
        conductivity_multiplier=self.conductivity_multiplier,
        neoclassical_correction=self.neoclassical_correction,
    )

  @abc.abstractmethod
  def build_model(self) -> ConductivityModel:
    """Builds conductivity model."""
