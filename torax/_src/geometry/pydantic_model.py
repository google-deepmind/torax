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
"""Pydantic model for geometry."""
import functools
from typing import Annotated, Any, Literal, TypeVar

import pydantic
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry
from torax._src.geometry import eqdsk
from torax._src.geometry import fbt
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import imas
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

T = TypeVar('T')


# pylint: disable=invalid-name
class CircularConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the circular geometry config.

  Attributes:
    geometry_type: Always set to 'circular'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
    elongation_LCFS: Sets the plasma elongation used for volume, area and
      q-profile corrections.
  """

  geometry_type: Annotated[
      Literal['circular'], torax_pydantic.TIME_INVARIANT
  ] = 'circular'
  n_rho: Annotated[pydantic.PositiveInt, torax_pydantic.TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3
  elongation_LCFS: pydantic.PositiveFloat = 1.72

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def build_geometry(self) -> geometry.Geometry:
    return circular_geometry.build_circular_geometry(
        n_rho=self.n_rho,
        elongation_LCFS=self.elongation_LCFS,
        R_major=self.R_major,
        a_minor=self.a_minor,
        B_0=self.B_0,
        hires_factor=self.hires_factor,
    )


class GeometryConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a single geometry config."""

  config: (
      CircularConfig
      | chease.CheaseConfig
      | fbt.FBTConfig
      | eqdsk.EQDSKConfig
      | imas.IMASConfig
  ) = pydantic.Field(discriminator='geometry_type')


class Geometry(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a geometry.

  This object can be constructed via `Geometry.from_dict(config)`, where
  `config` is a dict described in
  https://torax.readthedocs.io/en/latest/configuration.html#geometry.

  Attributes:
    geometry_type: A `geometry.GeometryType` enum.
    geometry_configs: Either a single `GeometryConfig` or a dict of
      `GeometryConfig` objects, where the keys are times in seconds.
    calcphibdot: Whether to calculate Phibdot in the geometry dataclasses. This
      is used in calc_coeffs to calculate terms related to time-dependent
      geometry. We can set this to False to zero out Phibdot in the geometry
      dataclasses to look at its effect on the simulation.
  """

  geometry_type: geometry.GeometryType
  geometry_configs: GeometryConfig | dict[torax_pydantic.Second, GeometryConfig]
  calcphibdot: Annotated[bool, torax_pydantic.JAX_STATIC] = True

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:

    if 'geometry_type' not in data:
      raise ValueError('geometry_type must be set in the input config.')

    geometry_type = data['geometry_type']
    # The geometry type can be an int if loading from JSON.
    if isinstance(geometry_type, geometry.GeometryType | int):
      return data
    # Parse the user config dict.
    elif isinstance(geometry_type, str):
      return _conform_user_data(data)
    else:
      raise ValueError(f'Invalid value for geometry: {geometry_type}')

  @functools.cached_property
  def build_provider(self) -> geometry_provider.GeometryProvider:
    # TODO(b/398191165): Remove this branch once the FBT bundle logic is
    # redesigned.
    if self.geometry_type == geometry.GeometryType.FBT:
      if not isinstance(self.geometry_configs, dict):
        assert isinstance(self.geometry_configs.config, fbt.FBTConfig)
        fbt_config = self.geometry_configs.config
        if fbt_config.LY_bundle_object is not None:
          return fbt_config.build_fbt_geometry_provider_from_bundle(
              self.calcphibdot
          )

    if isinstance(self.geometry_configs, dict):
      geometries = {
          time: config.config.build_geometry()
          for time, config in self.geometry_configs.items()
      }
      provider = (
          geometry_provider.TimeDependentGeometryProvider.create_provider
          if self.geometry_type == geometry.GeometryType.CIRCULAR
          else standard_geometry.StandardGeometryProvider.create_provider
      )
      return provider(geometries, calcphibdot=self.calcphibdot)
    else:
      geometries = self.geometry_configs.config.build_geometry()
      provider = geometry_provider.ConstantGeometryProvider
      return provider(geometries)


def _conform_user_data(data: dict[str, Any]) -> dict[str, Any]:
  """Conform the user geometry dict to the pydantic model."""

  if 'LY_bundle_object' in data and 'geometry_configs' in data:
    raise ValueError(
        'Cannot use both `LY_bundle_object` and `geometry_configs` together.'
    )

  data_copy = data.copy()
  # Useful to avoid failing if users mistakenly give the wrong case.
  data_copy['geometry_type'] = data['geometry_type'].lower()
  geometry_type = getattr(geometry.GeometryType, data['geometry_type'].upper())
  constructor_args = {'geometry_type': geometry_type}
  configs_time_dependent = data_copy.pop('geometry_configs', None)

  if 'calcphibdot' in data_copy:
    calcphibdot = data_copy.pop('calcphibdot')
    constructor_args['calcphibdot'] = calcphibdot

  if configs_time_dependent:
    # geometry config has sequence of standalone geometry files.
    if not isinstance(data['geometry_configs'], dict):
      raise ValueError('geometry_configs must be a dict.')
    constructor_args['geometry_configs'] = {}
    for time, c_time_dependent in configs_time_dependent.items():
      gc = GeometryConfig.from_dict({'config': c_time_dependent | data_copy})
      constructor_args['geometry_configs'][time] = gc
      if x := set(gc.config.time_invariant_fields()).intersection(
          c_time_dependent.keys()
      ):
        raise ValueError(
            'The following parameters cannot be set per geometry_config:'
            f' {", ".join(x)}'
        )
  else:
    constructor_args['geometry_configs'] = {'config': data_copy}

  return constructor_args
