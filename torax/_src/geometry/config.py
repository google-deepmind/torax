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
"""Pydantic config for geometry."""

# pylint: disable=invalid-name

import functools
from typing import Annotated, Any, TypeAlias

import numpy as np
import pydantic
from torax._src.geometry import chease
from torax._src.geometry import circular_geometry
from torax._src.geometry import eqdsk
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import imas
from torax._src.geometry import meq
from torax._src.geometry import standard_geometry
from torax._src.geometry import tokamaker
from torax._src.torax_pydantic import torax_pydantic

SupportedGeometryConfig: TypeAlias = Annotated[
    circular_geometry.CircularConfig
    | chease.CheaseConfig
    | meq.MEQConfig
    | eqdsk.EQDSKConfig
    | imas.IMASConfig
    | tokamaker.TokaMakerConfig,
    pydantic.Field(discriminator='geometry_type'),
]

_GEOMETRY_CONFIG_ADAPTER = pydantic.TypeAdapter(SupportedGeometryConfig)


class GeometryConfig(torax_pydantic.ConfigBase):
  """Pydantic config for a geometry.

  This object can be constructed via `GeometryConfig.from_dict(config)`, where
  `config` is a dict described in
  https://torax.readthedocs.io/en/latest/configuration.html#geometry.

  Attributes:
    geometry_type: A `geometry.GeometryType` enum.
    geometry_configs: Either a single `SupportedGeometryConfig` or a dict of
      `SupportedGeometryConfig` objects, where the keys are times in seconds.
    calc_Phi_b_dot: Whether to calculate Phi_b_dot in the geometry dataclasses.
      This is used in calc_coeffs to calculate terms related to time-dependent
      geometry. We can set this to False to zero out Phi_b_dot in the geometry
      dataclasses to look at its effect on the simulation.
  """

  geometry_type: geometry.GeometryType
  geometry_configs: (
      SupportedGeometryConfig
      | dict[torax_pydantic.Second, SupportedGeometryConfig]
  )
  calc_Phi_b_dot: Annotated[bool, torax_pydantic.JAX_STATIC] = True

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
    """Builds and caches a `GeometryProvider` from the configuration."""
    # Time-varying geometries from MEQ and IMAS can be loaded directly and we
    # special-case both paths here.
    # TODO(b/398191165): Remove these branches once the Geometry configuration
    # is redesigned.
    if self.geometry_type == geometry.GeometryType.MEQ:
      if not isinstance(self.geometry_configs, dict):
        assert isinstance(self.geometry_configs, meq.MEQConfig)
        if self.geometry_configs.LY_bundle_object is not None:
          return self.geometry_configs.build_meq_geometry_provider_from_bundle(
              self.calc_Phi_b_dot
          )
    if self.geometry_type == geometry.GeometryType.IMAS:
      if not isinstance(self.geometry_configs, dict):
        assert isinstance(self.geometry_configs, imas.IMASConfig)
        if self.geometry_configs.slice_time is None:
          return self.geometry_configs.build_geometry_provider(
              self.calc_Phi_b_dot
          )

    if isinstance(self.geometry_configs, dict):
      geometries = {
          time: config.build_geometry()
          for time, config in self.geometry_configs.items()
      }
      provider = (
          geometry_provider.TimeDependentGeometryProvider.create_provider
          if self.geometry_type == geometry.GeometryType.CIRCULAR
          else standard_geometry.StandardGeometryProvider.create_provider
      )
      return provider(geometries, calc_Phi_b_dot=self.calc_Phi_b_dot)
    else:
      geometries = self.geometry_configs.build_geometry()
      provider = geometry_provider.ConstantGeometryProvider
      return provider(geometries)

  def get_face_centers(self) -> np.ndarray:
    """Returns the face centers of the geometry.

    Returns:
      1D NumPy array of normalized radial face center coordinates.
    """
    if isinstance(self.geometry_configs, dict):
      # We check that the grid is the same for all geometries so can return the
      # face centers from any of them here.
      return next(iter(self.geometry_configs.values())).get_face_centers()
    else:
      return self.geometry_configs.get_face_centers()


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

  if 'calc_Phi_b_dot' in data_copy:
    calc_Phi_b_dot = data_copy.pop('calc_Phi_b_dot')
    constructor_args['calc_Phi_b_dot'] = calc_Phi_b_dot

  if configs_time_dependent:
    # geometry config has sequence of standalone geometry files.
    if not isinstance(data['geometry_configs'], dict):
      raise ValueError('geometry_configs must be a dict.')
    constructor_args['geometry_configs'] = {}
    for time, c_time_dependent in configs_time_dependent.items():
      gc = _GEOMETRY_CONFIG_ADAPTER.validate_python(
          c_time_dependent | data_copy
      )
      constructor_args['geometry_configs'][time] = gc
      if x := set(gc.time_invariant_fields()).intersection(
          c_time_dependent.keys()
      ):
        raise ValueError(
            'The following parameters cannot be set per geometry_config:'
            f' {", ".join(x)}'
        )
  else:
    constructor_args['geometry_configs'] = data_copy

  return constructor_args
