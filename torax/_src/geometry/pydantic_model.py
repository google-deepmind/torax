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

from collections.abc import Callable
from collections.abc import Mapping
import functools
import inspect
import logging
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from imas import ids_toplevel
import pydantic
from torax._src.geometry import circular_geometry
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
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
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
    elongation_LCFS: Sets the plasma elongation used for volume, area and
      q-profile corrections.
  """

  geometry_type: Annotated[Literal['circular'], TIME_INVARIANT] = 'circular'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
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


class CheaseConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the CHEASE geometry.

  Attributes:
    geometry_type: Always set to 'chease'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    R_major: Major radius (R) in meters.
    a_minor: Minor radius (a) in meters.
    B_0: Vacuum toroidal magnetic field on axis [T].
  """

  geometry_type: Annotated[Literal['chease'], TIME_INVARIANT] = 'chease'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  geometry_file: str = 'iterhybrid.mat2cols'
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> typing_extensions.Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
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
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    hires_factor: Sets up a higher resolution mesh with ``nrho_hires = nrho *
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
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
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
    logging.warning(
        '<B^2> and <1/B^2> not currently supported by FBT geometry;'
        ' approximating using analytical expressions for circular geometry.'
        ' This might cause inaccuracies in neoclassical transport.'
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
      self, calcphibdot: bool,
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
        geometries, calcphibdot=calcphibdot,
    )


class EQDSKConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the EQDSK geometry.

  Attributes:
    geometry_type: Always set to 'eqdsk'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    n_surfaces: Number of surfaces for which flux surface averages are
      calculated.
    last_surface_factor: Multiplication factor of the boundary poloidal flux,
      used for the contour defining geometry terms at the LCFS on the TORAX
      grid. Needed to avoid divergent integrations in diverted geometries.
    cocos: COCOS coordinate convention of the EQDSK file, specified as an
      integer in the range 1-8 or 11-18 inclusive.
  """

  geometry_type: Annotated[Literal['eqdsk'], TIME_INVARIANT] = 'eqdsk'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  geometry_file: str
  n_surfaces: pydantic.PositiveInt = 100
  last_surface_factor: torax_pydantic.OpenUnitInterval = 0.99
  cocos: geometry_loader.COCOSInt

  def build_geometry(self) -> standard_geometry.StandardGeometry:
    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_eqdsk,
            self.__dict__,
        )
    )


class IMASConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for the IMAS geometry.

  Currently written for COCOSv17 and DDv4. Note the IMASConfig is experimental
  for now and may undergo changes.

  Note the input here supports one and only one of the following:
  1. imas_filepath: Path to the IMAS netCDF file containing the equilibrium
      data. This is the only option that is covered by integration tests and can
      be used in the main TORAX run loop without additional dependencies.
  2. imas_uri: The IMAS uri containing the equilibrium data. Using this option
      requires access to imas-core, not part of the standard TORAX dependencies.
      This means this option is also not covered by integration tests.
  3. equilibrium_object: The equilibrium IDS containing the relevant data. This
      method does not currently support serialisation and is not compatible with
      the main TORAX run loop.

  Attributes:
    geometry_type: Always set to 'imas'.
    n_rho: Number of radial grid points.
    hires_factor: Only used when the initial condition ``psi`` is from plasma
      current. Sets up a higher resolution mesh with ``nrho_hires = nrho *
      hi_res_fac``, used for ``j`` to ``psi`` conversions.
    geometry_directory: Optionally overrides the default geometry directory.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the geometry file. If True, then the `psi`
      calculated from the geometry file is scaled to match the desired `I_p`.
    imas_filepath: Path to the IMAS netCDF file containing the equilibrium data.
      Only one of `imas_filepath`, `imas_uri`, or `equilibrium_object` can be
      set.
    imas_uri: The IMAS uri containing the equilibrium data. Using this option
      requires access to imas-core, not part of the standard TORAX dependencies.
      This means this option is also not covered by integration tests. Only one
      of imas_filepath, imas_uri or equilibrium_object can be set.
    equilibrium_object: The equilibrium IDS containing the relevant data. This
      method does not currently support serialisation and is not compatible with
      the with running TORAX using the provided APIs. To use this option you
      must implement a custom run loop. Only one of imas_filepath, imas_uri or
      equilibrium_object can be set.
    slice_time: Time of slice to load from IMAS IDS. If given, overrides
      slice_index.
    slice_index: Index of slice to load from IMAS IDS.
  """

  geometry_type: Annotated[Literal['imas'], TIME_INVARIANT] = 'imas'
  n_rho: Annotated[pydantic.PositiveInt, TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, TIME_INVARIANT] = None
  Ip_from_parameters: Annotated[bool, TIME_INVARIANT] = True
  imas_filepath: str | None = 'ITERhybrid_COCOS17_IDS_ddv4.nc'
  imas_uri: str | None = None
  equilibrium_object: ids_toplevel.IDSToplevel | None = None
  slice_index: pydantic.NonNegativeInt = 0
  slice_time: float | None = None

  @pydantic.model_validator(mode='after')
  def _validate_model(self) -> typing_extensions.Self:
    specified_inputs = [
        field
        for field in [
            self.equilibrium_object,
            self.imas_uri,
            self.imas_filepath,
        ]
        if field is not None
    ]
    if len(specified_inputs) != 1:
      raise ValueError(
          'IMAS geometry builder needs exactly one of `equilibrium_object`, '
          '`imas_uri` or `imas_filepath` to be a valid input. Inputs provided: '
          f'{specified_inputs}.'
      )
    return self

  def build_geometry(self) -> standard_geometry.StandardGeometry:

    return standard_geometry.build_standard_geometry(
        _apply_relevant_kwargs(
            standard_geometry.StandardGeometryIntermediates.from_IMAS,
            self.__dict__,
        )
    )


class GeometryConfig(torax_pydantic.BaseModelFrozen):
  """Pydantic model for a single geometry config."""

  config: (
      CircularConfig | CheaseConfig | FBTConfig | EQDSKConfig | IMASConfig
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
        assert isinstance(self.geometry_configs.config, FBTConfig)
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


def _apply_relevant_kwargs(f: Callable[..., T], kwargs: Mapping[str, Any]) -> T:
  """Apply only the kwargs actually used by the function."""
  relevant_kwargs = [i.name for i in inspect.signature(f).parameters.values()]
  kwargs = {k: kwargs[k] for k in relevant_kwargs}
  return f(**kwargs)
