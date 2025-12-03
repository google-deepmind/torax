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
"""Classes for loading and representing a FBT geometry."""
from collections.abc import Mapping
import enum
import logging
from typing import Annotated
from typing import Any
from typing import Literal, TypeAlias
import numpy as np
import pydantic
from torax._src import constants
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name
LY_OBJECT_TYPE: TypeAlias = (
    str | Mapping[str, torax_pydantic.NumpyArray | float]
)


@enum.unique
class DivertorDomain(enum.StrEnum):
  """Enum for selecting the divertor domain in the extended Lengyel model."""

  UPPER_NULL = 'upper_null'
  LOWER_NULL = 'lower_null'


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

  geometry_type: Annotated[Literal['fbt'], torax_pydantic.TIME_INVARIANT] = (
      'fbt'
  )
  n_rho: Annotated[pydantic.PositiveInt, torax_pydantic.TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, torax_pydantic.TIME_INVARIANT] = (
      None
  )
  Ip_from_parameters: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True
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
    intermediates = _from_fbt_single_slice(
        geometry_directory=self.geometry_directory,
        LY_object=self.LY_object,
        L_object=self.L_object,
        Ip_from_parameters=self.Ip_from_parameters,
        n_rho=self.n_rho,
        hires_factor=self.hires_factor,
    )

    return standard_geometry.build_standard_geometry(intermediates)

  # TODO(b/398191165): Remove this branch once the FBT bundle logic is
  # redesigned.
  def build_fbt_geometry_provider_from_bundle(
      self,
      calcphibdot: bool,
  ) -> geometry_provider.GeometryProvider:
    """Builds a `GeometryProvider` from the input config."""
    intermediates = _from_fbt_bundle(
        geometry_directory=self.geometry_directory,
        LY_bundle_object=self.LY_bundle_object,
        L_object=self.L_object,
        LY_to_torax_times=self.LY_to_torax_times,
        Ip_from_parameters=self.Ip_from_parameters,
        n_rho=self.n_rho,
        hires_factor=self.hires_factor,
    )
    geometries = {
        t: standard_geometry.build_standard_geometry(intermediates[t])
        for t in intermediates
    }
    return standard_geometry.StandardGeometryProvider.create_provider(
        geometries,
        calcphibdot=calcphibdot,
    )


def _from_fbt_single_slice(
    geometry_directory: str | None,
    LY_object: str | Mapping[str, np.ndarray],
    L_object: str | Mapping[str, np.ndarray],
    Ip_from_parameters: bool = True,
    n_rho: int = 25,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
) -> standard_geometry.StandardGeometryIntermediates:
  """Returns StandardGeometryIntermediates from a single slice FBT LY file.

  LY and L are FBT data files containing magnetic geometry information.
  The majority of the needed information is in the LY file. The L file
  is only needed to get the normalized poloidal flux coordinate, pQ.

  This method is for cases when the LY file on disk corresponds to a single
  time slice. Either a single time slice or sequence of time slices can be
  provided in the geometry config.

  Args:
    geometry_directory: Directory where to find the FBT file describing the
      magnetic geometry. If None, then it defaults to another dir. See
      `load_geo_data` implementation.
    LY_object: File name for LY data, or directly an LY single slice dict.
    L_object: File name for L data, or directly an L dict.
    Ip_from_parameters: If True, then Ip is taken from the config and the values
      in the Geometry are rescaled
    n_rho: Grid resolution used for all TORAX cell variables.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.

  Returns:
    A StandardGeometryIntermediates instance based on the input slice. This
    can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """
  if isinstance(LY_object, str):
    LY = geometry_loader.load_geo_data(
        geometry_directory, LY_object, geometry_loader.GeometrySource.FBT
    )
  elif isinstance(LY_object, Mapping):
    LY = LY_object
  else:
    raise ValueError('LY_object must be a string (file path) or a dictionary.')
  if isinstance(L_object, str):
    L = geometry_loader.load_geo_data(
        geometry_directory, L_object, geometry_loader.GeometrySource.FBT
    )
  elif isinstance(L_object, Mapping):
    L = L_object
  else:
    raise ValueError('L_object must be a string (file path) or a dictionary.')

  # Convert any scalar LY values to ndarrays such that validation method works
  for key in LY:
    if not isinstance(LY[key], np.ndarray):
      LY[key] = np.array(LY[key])

  # Raises a ValueError if the data is invalid.
  _validate_fbt_data(LY, L)
  return _from_fbt(
      LY, L, Ip_from_parameters, n_rho, hires_factor, divertor_domain
  )


def _from_fbt_bundle(
    geometry_directory: str | None,
    LY_bundle_object: str | Mapping[str, np.ndarray],
    L_object: str | Mapping[str, np.ndarray],
    LY_to_torax_times: np.ndarray | None,
    Ip_from_parameters: bool = True,
    n_rho: int = 25,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
) -> Mapping[float, standard_geometry.StandardGeometryIntermediates]:
  """Returns StandardGeometryIntermediates from a bundled FBT LY file.

  LY_bundle_object is an FBT data object containing a bundle of LY geometry
  slices at different times, packaged within a single object (as opposed to
  a sequence of standalone LY files). LY_to_torax_times is a 1D array of
  times, defining the times in the TORAX simulation corresponding to each
  slice in the LY bundle. All times in the LY bundle must be mapped to
  times in TORAX. The LY_bundle_object and L_object can either be file names
  for disk loading, or directly the data dicts.

  Args:
    geometry_directory: Directory where to find the FBT file describing the
      magnetic geometry. If None, then it defaults to another dir. See
      `load_geo_data` implementation.
    LY_bundle_object: Either file name for bundled LY data, e.g. as produced by
      liuqe meqlpack, or the data dict itself.
    L_object: Either file name for L data. Assumed to be the same L data for all
      LY slices in the bundle, or the data dict itself.
    LY_to_torax_times: User-provided times which map the times of the LY
      geometry slices to TORAX simulation times. A ValueError is raised if the
      number of array elements doesn't match the length of the LY_bundle array
      data. If None, then times are taken from the LY_bundle_object itself.
    Ip_from_parameters: If True, then Ip is taken from the config and the values
      in the Geometry are rescaled.
    n_rho: Grid resolution used for all TORAX cell variables.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.

  Returns:
    A mapping from user-provided (or inferred) times to
    StandardGeometryIntermediates instances based on the input slices. This
    can then be used to build a StandardGeometryProvider.
  """

  if isinstance(LY_bundle_object, str):
    LY_bundle = geometry_loader.load_geo_data(
        geometry_directory,
        LY_bundle_object,
        geometry_loader.GeometrySource.FBT,
    )
  elif isinstance(LY_bundle_object, Mapping):
    LY_bundle = LY_bundle_object
  else:
    raise ValueError(
        'LY_bundle_object must be a string (file path) or a dictionary.'
    )

  if isinstance(L_object, str):
    L = geometry_loader.load_geo_data(
        geometry_directory, L_object, geometry_loader.GeometrySource.FBT
    )
  elif isinstance(L_object, Mapping):
    L = L_object
  else:
    raise ValueError('L_object must be a string (file path) or a dictionary.')

  # Raises a ValueError if the data is invalid.
  _validate_fbt_data(LY_bundle, L)

  if LY_to_torax_times is None:
    LY_to_torax_times = LY_bundle['t']  # ndarray of times
  else:
    if len(LY_to_torax_times) != len(LY_bundle['t']):
      raise ValueError(f"""
          Length of LY_to_torax_times must match length of LY bundle data:
          len(LY_to_torax_times)={len(LY_to_torax_times)},
          len(LY_bundle['t'])={len(LY_bundle['t'])}
          """)

  intermediates = {}
  for idx, t in enumerate(LY_to_torax_times):
    data_slice = _get_LY_single_slice_from_bundle(LY_bundle, idx)
    intermediates[t] = _from_fbt(
        data_slice,
        L,
        Ip_from_parameters,
        n_rho,
        hires_factor,
        divertor_domain,
    )

  return intermediates


def _get_LY_single_slice_from_bundle(
    LY_bundle: Mapping[str, np.ndarray],
    idx: int,
) -> Mapping[str, np.ndarray]:
  """Returns a single LY slice from a bundled LY file, at index idx."""

  # The keys below are the required LY keys for the FBT geometry provider.
  required_keys = [
      'rBt',
      'aminor',
      'rgeom',
      'epsilon',
      'TQ',
      'FB',
      'FA',
      'Q0Q',
      'Q1Q',
      'Q2Q',
      'Q3Q',
      'Q4Q',
      'Q5Q',
      'ItQ',
      'deltau',
      'deltal',
      'kappa',
      'zA',
      'lX',
  ]

  # The keys below are the optional LY keys for the FBT geometry provider.
  # They are used to extract edge quantities for extended Lengyel.
  optional_keys = [
      'z_div',
      'Lpar_target',
      'Lpar_div',
      'alpha_target',
      'r_OMP',
      'r_target',
      'Bp_OMP',
  ]

  LY_single_slice_required = {
      key: LY_bundle[key][..., idx] for key in required_keys
  }
  LY_single_slice_optional = {
      key: LY_bundle[key][..., idx] for key in optional_keys if key in LY_bundle
  }
  LY_single_slice = LY_single_slice_required | LY_single_slice_optional

  # load FtPVQ if it exists, otherwise use FtPQ for toroidal flux.
  if 'FtPVQ' in LY_bundle:
    LY_single_slice['FtPVQ'] = LY_bundle['FtPVQ'][..., idx]
  else:
    # TODO(b/412965439) remove support for LY files that don't contain FtPVQ.
    logging.warning(
        'FtPVQ not found in LY bundle, using FtPQ instead. Please upgrade to'
        ' a newer version of MEQ as the source of the LY data. This will'
        ' throw an error in a future version.'
    )
    LY_single_slice['FtPVQ'] = LY_bundle['FtPQ'][..., idx]
  return LY_single_slice


def _from_fbt(
    LY: Mapping[str, np.ndarray | int],
    L: Mapping[str, np.ndarray],
    Ip_from_parameters: bool = True,
    n_rho: int = 25,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from a single FBT LY slice.

  Args:
    LY: A dictionary of relevant FBT LY geometry data.
    L: A dictionary of relevant FBT L geometry data.
    Ip_from_parameters: If True, then Ip is taken from the config and the values
      in the Geometry are rescaled.
    n_rho: Grid resolution used for all TORAX cell variables.
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations on initialization.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.

  Returns:
    A StandardGeometryIntermediates instance based on the input slice. This
    can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """
  # lX is a flag for diverted (1) or limited (0) geometry. Converted to
  # boolean when constructing the StandardGeometryIntermediates.
  if np.squeeze(LY['lX']) not in [0, 1]:
    raise ValueError(f"LY['lX'] must be 0 or 1, but got {LY['lX']}")
  # Convert to bool instead of dim 0 array of ints
  diverted = bool(LY['lX'] == 1)
  R_major = LY['rgeom'][-1]  # Major radius
  B_0 = LY['rBt'] / R_major  # Vacuum toroidal magnetic field on axis
  a_minor = LY['aminor'][-1]  # Minor radius
  # Toroidal flux including plasma contribution
  # load FtPVQ if it exists, otherwise use FtPQ for toroidal flux.
  if 'FtPVQ' in LY:
    Phi = LY['FtPVQ']
  else:
    # TODO(b/412965439)
    logging.warning(
        'FtPVQ not found in LY, using FtPQ instead. Please upgrade to'
        ' a newer version of MEQ as the source of the LY data. This will'
        ' throw an error in a future version.'
    )
    Phi = LY['FtPQ']

  rhon = np.sqrt(Phi / Phi[-1])  # Normalized toroidal flux coordinate
  psi = L['pQ'] ** 2 * (LY['FB'] - LY['FA']) + LY['FA']  # Poloidal flux
  # To avoid possible divisions by zero in diverted geometry. Value of what
  # replaces the zero does not matter, since it will be replaced by a spline
  # extrapolation in the post_init.
  LY_Q1Q = np.where(LY['Q1Q'] != 0, LY['Q1Q'], constants.CONSTANTS.eps)

  # TODO(b/426291465): Implement a more accurate calculation of <1/B^2>
  # (either here or upstream in MEQ)
  # Approximate with analytical expressions for circular geometry.
  flux_surf_avg_B2 = B_0**2 / np.sqrt(1.0 - LY['epsilon'] ** 2)
  flux_surf_avg_1_over_B2 = B_0**-2 * (1.0 + 1.5 * LY['epsilon'] ** 2)

  # Edge/Divertor geometry
  # These parameters are optional as older FBT files may not contain them.
  connection_length_target = None
  connection_length_divertor = None
  angle_of_incidence_target = None
  R_OMP = None
  R_target = None
  B_pol_OMP = None

  if 'z_div' in LY:
    # Ensure z_div is an array
    z_div = np.squeeze(LY['z_div'])

    # Find index corresponding to requested domain.
    if diverted:
      if divertor_domain == DivertorDomain.LOWER_NULL:
        idx_array = np.where(z_div < 0)[0]
      else:  # UPPER_NULL
        idx_array = np.where(z_div > 0)[0]
      if idx_array.size == 0:
        raise ValueError(
            f'{divertor_domain} not present in edge geometry data.'
        )
      # There should only be one entry per domain.
      idx = idx_array[0]
    else:
      # Limited geometry: minor difference between directions.
      idx = 0

    # Helper to safe get and index
    def _get_val(key):
      if key not in LY:
        return None
      val = np.squeeze(LY[key])
      if not val.shape:  # Scalar value
        return val
      else:
        return val[idx]

    # Lpar_target -> connection_length_target [m]
    connection_length_target = _get_val('Lpar_target')
    # Lpar_div -> connection_length_divertor [m]
    connection_length_divertor = _get_val('Lpar_div')
    # alpha_target [radians] -> angle_of_incidence_target [degrees]
    angle_of_incidence_target = np.rad2deg(_get_val('alpha_target'))
    # r_OMP -> R_OMP [m]
    R_OMP = _get_val('r_OMP')
    # r_target -> R_target [m]
    R_target = _get_val('r_target')
    # Bp_OMP -> B_pol_OMP [T]
    B_pol_OMP = _get_val('Bp_OMP')

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.FBT,
      Ip_from_parameters=Ip_from_parameters,
      R_major=R_major,
      a_minor=a_minor,
      B_0=B_0,
      psi=psi,
      Phi=Phi,
      Ip_profile=np.abs(LY['ItQ']),
      R_in=LY['rgeom'] - LY['aminor'],
      R_out=LY['rgeom'] + LY['aminor'],
      F=np.abs(LY['TQ']),
      int_dl_over_Bp=1 / LY_Q1Q,
      flux_surf_avg_1_over_R=LY['Q0Q'],
      flux_surf_avg_1_over_R2=LY['Q2Q'],
      flux_surf_avg_grad_psi2_over_R2=np.abs(LY['Q3Q']),
      flux_surf_avg_grad_psi=np.abs(LY['Q5Q']),
      flux_surf_avg_grad_psi2=np.abs(LY['Q4Q']),
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      delta_upper_face=LY['deltau'],
      delta_lower_face=LY['deltal'],
      elongation=LY['kappa'],
      vpr=4 * np.pi * Phi[-1] * rhon / (np.abs(LY['TQ']) * LY['Q2Q']),
      n_rho=n_rho,
      hires_factor=hires_factor,
      diverted=diverted,
      connection_length_target=connection_length_target,
      connection_length_divertor=connection_length_divertor,
      angle_of_incidence_target=angle_of_incidence_target,
      R_OMP=R_OMP,
      R_target=R_target,
      B_pol_OMP=B_pol_OMP,
      z_magnetic_axis=LY['zA'],
  )


def _validate_fbt_data(
    LY: Mapping[str, np.ndarray], L: Mapping[str, np.ndarray]
) -> None:
  """Validates the FBT data dictionaries.

  Works for both single slice and bundle LY data.

  Args:
    LY: A dictionary of FBT LY geometry data.
    L: A dictionary of FBT L geometry data.

  Raises a ValueError if the data is invalid.
  """

  # The checks for L['pQ'] and LY['t'] are done first since their existence
  # is needed for the shape checks.
  if 'pQ' not in L:
    raise ValueError("L data is missing the 'pQ' key.")
  if 't' not in LY:
    raise ValueError("LY data is missing the 't' key.")

  len_psinorm = len(L['pQ'])
  len_times = len(LY['t']) if LY['t'].shape else 1  # Handle scalar t
  time_only_shape = (len_times,) if len_times > 1 else ()
  psi_and_time_shape = (
      (len_psinorm, len_times) if len_times > 1 else (len_psinorm,)
  )

  required_LY_spec = {
      'rBt': time_only_shape,
      'aminor': psi_and_time_shape,
      'rgeom': psi_and_time_shape,
      'epsilon': psi_and_time_shape,
      'TQ': psi_and_time_shape,
      'FB': time_only_shape,
      'FA': time_only_shape,
      'Q0Q': psi_and_time_shape,
      'Q1Q': psi_and_time_shape,
      'Q2Q': psi_and_time_shape,
      'Q3Q': psi_and_time_shape,
      'Q4Q': psi_and_time_shape,
      'Q5Q': psi_and_time_shape,
      'ItQ': psi_and_time_shape,
      'deltau': psi_and_time_shape,
      'deltal': psi_and_time_shape,
      'kappa': psi_and_time_shape,
      'zA': time_only_shape,
      'lX': time_only_shape,
  }
  toroidal_flux_spec = {
      'FtPVQ': psi_and_time_shape,
      'FtPQ': psi_and_time_shape,
  }

  missing_LY_keys = required_LY_spec.keys() - LY.keys()
  if missing_LY_keys:
    raise ValueError(
        f'LY data is missing the following keys: {missing_LY_keys}'
    )
  missing_toroidal_flux = 'FtPVQ' not in LY.keys() and 'FtPQ' not in LY.keys()
  if missing_toroidal_flux:
    raise ValueError(
        'LY data is missing a toroidal flux-related key '
        'provide either FtPVQ or FtPQ.'
    )

  for key, shape in required_LY_spec.items():
    # all keys should be present to check them all
    if LY[key].shape != shape:
      raise ValueError(
          f"Incorrect shape for key '{key}' in LY data. "
          f'Expected {shape}:, got {LY[key].shape}.'
      )
  for key, shape in toroidal_flux_spec.items():
    # only check the shape if the key is present
    if key in LY and LY[key].shape != shape:
      raise ValueError(
          f"Incorrect shape for key '{key}' in LY data. "
          f'Expected {shape}:, got {LY[key].shape}.'
      )
