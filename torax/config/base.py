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

"""Base classes for runtime parameter configs."""

from __future__ import annotations

import abc
import dataclasses
from typing import Generic, TypeVar

import chex
from torax import geometry

DynamicT = TypeVar('DynamicT')
ProviderT = TypeVar('ProviderT', bound='RuntimeParametersProvider')


@dataclasses.dataclass
class RuntimeParametersConfig(Generic[ProviderT], metaclass=abc.ABCMeta):
  """Base class for all runtime parameter configs.

  The purpose of this config class is to be a container for all the runtime
  parameters that are defined in a config file including arguments needed to
  construct interpolated params
  """

  @abc.abstractmethod
  def make_provider(
      self,
      torax_mesh: geometry.Grid1D | None = None,
  ) -> ProviderT:
    """Builds a RuntimeParamsProvider object from this config.

    Args:
      torax_mesh: Some of the interpolated parameters might be both radially and
        temporally interpolated. In this case as the TORAX mesh is known upfront
        we can compute the radial interpolation at construction time. In cases
        where none of the parameters will be radially interpolated, this can be
        left as None.

    The RuntimeParamsProvider object is intended to be ready for retrieving
    dynamic parameters at each time step during the simulation and is intended
    to contain the constructed interpolated variables.

    Returns:
      A RuntimeParamsProvider object.
    """


@dataclasses.dataclass
class RuntimeParametersProvider(Generic[DynamicT], metaclass=abc.ABCMeta):
  """Base class for all prepared runtime parameter configs.

  A lot of the variables used in the TORAX simulation loop are interpolated
  variables of type `interpolated_param.InterpolatedParamBase`.
  This class is intended to be ready for retrieving dynamic parameters
  at each time step during the simulation and is intended to contain the
  constructed interpolated variables.

  This class contains:
  - the config of all non-interpolated runtime parameters and constructor
  arguments for interpolated variables, used to get any static parameters.
  - any constructed interpolated variables which will already be spatially
  interpolated and could vary in time.
  """
  runtime_params_config: RuntimeParametersConfig

  @abc.abstractmethod
  def build_dynamic_params(
      self,
      t: chex.Numeric,
  ) -> DynamicT:
    """Builds dynamic parameters for this provider.

    This method is intended to be called at each time step during the simulation
    to get the dynamic parameters for this provider. For interpolated variables
    this method will interpolate the variables at the given time.

    Args:
      t: The time to interpolate the dynamic parameters at.

    Returns:
      The dynamic parameters at time t.
    """
