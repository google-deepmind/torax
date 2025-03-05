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

from collections.abc import Callable, Mapping
import functools
import inspect
from typing import Annotated, Any, Literal, TypeAlias, TypeVar
import pydantic
from torax.geometry import circular_geometry
from torax.geometry import geometry
from torax.geometry import geometry_provider
from torax.geometry import standard_geometry
from torax.torax_pydantic import torax_pydantic
import typing_extensions
# Using invalid-name because we are using the same naming convention as the
# external physics implementations
# pylint: disable=invalid-name
T = TypeVar('T')

LY_OBJECT_TYPE: TypeAlias = (
    str | Mapping[str, torax_pydantic.NumpyArray | float]
)

TIME_INVARIANT = torax_pydantic.TIME_INVARIANT


class CircularConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the circular geometry config.

  Attributes:
    geometry_type: Always set to 'circular'.
    n_rho: Number of radial grid points.
    hires_fac: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    Rmaj: Major radius (R) in meters.
    Rmin: Minor radius (a) in meters.
    B0: Vacuum toroidal magnetic field on axis [T].
    elongation_LCFS: Sets the plasma elongation used for volume, area and
      q-profile corrections.
  """

  geometry_type: Annotated[Literal['circular'], TIME_INVARIANT] = 'circular'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_fac: pydantic.PositiveInt = 4
  Rmaj: torax_pydantic.Meter = 6.2
  Rmin: torax_pydantic.Meter = 2.0
  B0: torax_pydantic.Tesla = 5.3
  elongation_LCFS: pydantic.PositiveFloat = 1.72

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.Rmaj >= self.Rmin:
      raise ValueError('Rmin must be less than or equal to Rmaj.')
    return self

  def build_geometry(self) -> geometry.Geometry:
    return circular_geometry.build_circular_geometry(
        n_rho=self.n_rho,
        elongation_LCFS=self.elongation_LCFS,
        Rmaj=self.Rmaj,
        Rmin=self.Rmin,
        B0=self.B0,
        hires_fac=self.hires_fac,
    )


class CheaseConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the CHEASE geometry.

  Attributes:
    geometry_type: Always set to 'chease'.
    n_rho: Number of radial grid points.
    hires_fac: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_dir: Optionally overrides the `TORAX_GEOMETRY_DIR` environment
      variable.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    Rmaj: Major radius (R) in meters.
    Rmin: Minor radius (a) in meters.
    B0: Vacuum toroidal magnetic field on axis [T].
  """

  geometry_type: Annotated[Literal['chease'], TIME_INVARIANT] = 'chease'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_fac: pydantic.PositiveInt = 4
  geometry_dir: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  geometry_file: str = 'ITER_hybrid_citrin_equil_cheasedata.mat2cols'
  Rmaj: torax_pydantic.Meter = 6.2
  Rmin: torax_pydantic.Meter = 2.0
  B0: torax_pydantic.Tesla = 5.3

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.Rmaj >= self.Rmin:
      raise ValueError('Rmin must be less than or equal to Rmaj.')
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:

    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_chease,
            self.__dict__,
        )
    )


class FBTConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the FBT geometry.

  Attributes:
    geometry_type: Always set to 'fbt'.
    n_rho: Number of radial grid points.
    hires_fac: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_dir: Optionally overrides the `TORAX_GEOMETRY_DIR` environment
      variable.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    hires_fac: Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    LY_object: Sets a single-slice FBT LY geometry file to be loaded, or
      alternatively a dict directly containing a single time slice of LY data.
    LY_bundle_object: Sets the FBT LY bundle file to be loaded, corresponding to
      multiple time-slices, or alternatively a dict directly containing all
      time-slices of LY data.
    LY_to_torax_times: Sets the TORAX simulation times corresponding to the
      individual slices in the FBT LY bundle file. If not provided, then the
      times are taken from the LY_bundle_file itself. The length of the array
      must match the number of slices in the bundle.
    L_object: Sets the FBT L geometry file loaded, or alternatively a dict
      directly containing the L data.
  """

  geometry_type: Annotated[Literal['fbt'], TIME_INVARIANT] = 'fbt'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_fac: pydantic.PositiveInt = 4
  geometry_dir: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  LY_object: LY_OBJECT_TYPE | None = None
  LY_bundle_object: LY_OBJECT_TYPE | None = None
  LY_to_torax_times: torax_pydantic.NumpyArray | None = None
  L_object: LY_OBJECT_TYPE | None = None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: dict[str, Any]) -> dict[str, Any]:
    # Remove unused fields from the data dict that come from file loading.
    for obj in ('L_object', 'LY_object'):
      if obj in data and isinstance(data[obj], dict):
        for k in ('__header__', '__version__', '__globals__', 'shot'):
          data[obj].pop(k, None)
    return data

  @pydantic.model_validator(mode='after')
  def _validate_model(self) -> typing_extensions.Self:
    if self.LY_bundle_object is not None and self.LY_object is not None:
      raise ValueError(
          "Cannot use 'LY_object' together with a bundled FBT file"
      )
    if self.LY_to_torax_times is not None and self.LY_bundle_object is None:
      raise ValueError(
          'LY_bundle_object must be set when using LY_to_torax_times.'
      )
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:

    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_fbt_single_slice,
            self.__dict__,
        )
    )

  # TODO(b/398191165): Remove this branch once the FBT bundle logic is
  # redesigned.
  def build_fbt_geometry_provider_from_bundle(
      self,
  ) -> geometry_provider.GeometryProvider:
    """Builds a `GeometryProvider` from the input config."""
    intermediates = _apply_relevant_kwargs(
        standard_geometry.StandardGeometryIntermediates.from_fbt_bundle,
        self.__dict__,
    )
    geometries = {
        t: standard_geometry.build_standard_geometry(intermediates[t])
        for t in intermediates
    }
    return standard_geometry.StandardGeometryProvider.create_provider(
        geometries
    )


class EQDSKConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the EQDSK geometry.

  Attributes:
    geometry_type: Always set to 'eqdsk'.
    n_rho: Number of radial grid points.
    hires_fac: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_dir: Optionally overrides the `TORAX_GEOMETRY_DIR` environment
      variable.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    n_surfaces: Number of surfaces for which flux surface averages are
      calculated.
    last_surface_factor: Multiplication factor of the boundary poloidal flux,
      used for the contour defining geometry terms at the LCFS on the TORAX
      grid. Needed to avoid divergent integrations in diverted geometries.
  """

  geometry_type: Annotated[Literal['eqdsk'], TIME_INVARIANT] = 'eqdsk'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_fac: pydantic.PositiveInt = 4
  geometry_dir: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  geometry_file: str = 'EQDSK_ITERhybrid_COCOS02.eqdsk'
  n_surfaces: pydantic.PositiveInt = 100
  last_surface_factor: torax_pydantic.OpenUnitInterval = 0.99

  def build_geometry(self) -> standard_geometry.StandardGeometry:
    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_eqdsk,
            self.__dict__,
        )
    )


class GeometryConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a single geometry config."""

  config: CircularConfig | CheaseConfig | FBTConfig | EQDSKConfig = (
      pydantic.Field(discriminator='geometry_type')
  )


class Geometry(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a geometry.

  This object can be constructed via `Geometry.from_dict(config)`, where
  `config` is a dict described in
  https://torax.readthedocs.io/en/latest/configuration.html#geometry.

  Attributes:
    geometry_type: A `geometry.GeometryType` enum.
    geometry_configs: Either a single `GeometryConfig` or a dict of
      `GeometryConfig` objects, where the keys are times in seconds.
  """

  geometry_type: geometry.GeometryType
  geometry_configs: GeometryConfig | dict[torax_pydantic.Second, GeometryConfig]

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
        assert isinstance(self.geometry_configs.config, FBTConfig)
        if self.geometry_configs.config.LY_bundle_object is not None:
          return (
              self.geometry_configs.config.build_fbt_geometry_provider_from_bundle()
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
    else:
      geometries = self.geometry_configs.config.build_geometry()
      provider = geometry_provider.ConstantGeometryProvider

    return provider(geometries)  # pytype: disable=attribute-error


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


def _apply_relevant_kwargs(f: Callable[..., T], kwargs: Mapping[str, Any]) -> T:
  """Apply only the kwargs actually used by the function."""
  relevant_kwargs = [i.name for i in inspect.signature(f).parameters.values()]
  kwargs = {k: kwargs[k] for k in relevant_kwargs}
  return f(**kwargs)
