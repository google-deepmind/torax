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
import enum
from typing import Any, Generic, TypeVar

import chex
from torax import interpolated_param
from torax.config import config_args
from torax.torax_pydantic import torax_pydantic

DynamicT = TypeVar('DynamicT')
ProviderT = TypeVar('ProviderT', bound='RuntimeParametersProvider')


class GridType(enum.Enum):
  """Describes where interpolated values are defined on."""

  CELL = enum.auto()
  FACE = enum.auto()

  def get_mesh(self, torax_mesh: torax_pydantic.Grid1D) -> chex.Array:
    match self:
      case GridType.CELL:
        return torax_mesh.cell_centers
      case GridType.FACE:
        return torax_mesh.face_centers


@dataclasses.dataclass
class RuntimeParametersConfig(Generic[ProviderT], metaclass=abc.ABCMeta):
  """Base class for all runtime parameter configs.

  The purpose of this config class is to be a container for all the runtime
  parameters that are defined in a config file including arguments needed to
  construct interpolated params
  """

  @property
  def grid_type(self) -> GridType:
    """The grid values any values interpolated in rhon will be defined on.

    For most cases quantities interpolated in rhon will be interpolated onto
    the cell grid. In case you want to override this behaviour for other params
    then override this property.
    """
    return GridType.CELL

  def get_provider_kwargs(
      self, torax_mesh: torax_pydantic.Grid1D | None = None
  ) -> dict[str, Any]:
    """Returns the kwargs to be passed to the provider constructor.

    We adopt a pattern where for any fields in a `RuntimeParametersConfig` that
    are interpolated variables, we have an identically named field in the
    `RuntimeParametersProvider` that is the constructed interpolated variable.
    We also assume for any fields in a `RuntimeParametersConfig` that are
    `RuntimeParametersConfig` themselves, we have an identically named field in
    the `RuntimeParametersProvider` that is the constructed provider.

    Args:
      torax_mesh: Required if any of the interpolated variables are both
        temporally and radially interpolated.

    Returns:
      A dict of kwargs to be passed to the provider constructor.
    """
    provider_kwargs = {'runtime_params_config': self}
    input_config_fields_to_types = {
        f.name: f.type for f in dataclasses.fields(self)
    }
    for field in dataclasses.fields(self):
      field_value = getattr(self, field.name)
      if config_args.input_is_an_interpolated_var_single_axis(
          field.name, input_config_fields_to_types
      ):
        # Some fields are a union including `None` values. In this case we
        # should not try to construct an interpolated var.
        if field_value is None:
          provider_kwargs[field.name] = None
        else:
          provider_kwargs[field.name] = (
              config_args.get_interpolated_var_single_axis(
                  getattr(self, field.name)
              )
          )
      elif config_args.input_is_an_interpolated_var_time_rho(
          field.name, input_config_fields_to_types
      ):
        # Some fields are a union including `None` values. In this case we
        # should not try to construct an interpolated var.
        if field_value is None:
          provider_kwargs[field.name] = None
        else:
          if torax_mesh is None:
            raise ValueError(
                f'torax_mesh is required for {self.__class__.__name__} as it'
                ' contains an interpolated var with time and rho.'
            )
          provider_kwargs[field.name] = config_args.get_interpolated_var_2d(
              getattr(self, field.name),
              self.grid_type.get_mesh(torax_mesh),
          )
      # If the field is itself a config, recurse.
      elif isinstance(field_value, RuntimeParametersConfig):
        provider_kwargs[field.name] = field_value.make_provider(torax_mesh)

    return provider_kwargs

  @abc.abstractmethod
  def make_provider(
      self,
      torax_mesh: torax_pydantic.Grid1D | None = None,
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

  def get_dynamic_params_kwargs(
      self,
      t: chex.Numeric,
  ) -> dict[str, Any]:
    """Returns the kwargs to be passed to the dynamic params constructor.

    This method is intended to be called at each time step during the simulation
    to get the dynamic parameters for this provider. For interpolated variables
    this method will interpolate the variables at the given time. For any fields
    that are themselves `RuntimeParametersProvider` this method will build the
    dynamic params for those fields as well.

    Args:
      t: The time to interpolate the dynamic parameters at.

    Returns:
      A dict of kwargs to be passed to the dynamic params constructor.
    """
    dynamic_params_kwargs = dataclasses.asdict(self.runtime_params_config)
    # Convert any Enums to their values.
    dynamic_params_kwargs = {
        k: v.value if isinstance(v, enum.Enum) else v
        for k, v in dynamic_params_kwargs.items()
    }
    for field in dataclasses.fields(self):
      field_value = getattr(self, field.name)
      match field_value:
        case RuntimeParametersConfig():
          continue
        case RuntimeParametersProvider():
          dynamic_params_kwargs[field.name] = field_value.build_dynamic_params(
              t
          )
        case interpolated_param.InterpolatedParamBase():
          dynamic_params_kwargs[field.name] = field_value.get_value(t)
        # Some interpolated param fields can also be None.
        case None:
          dynamic_params_kwargs[field.name] = None
        case _:
          raise ValueError(
              f'Unable to construct dynamic param for param {field.name} with'
              f' type {type(field_value)}.'
          )

    return dynamic_params_kwargs

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
