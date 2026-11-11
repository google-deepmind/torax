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
"""Functions for loading and representing an IMAS geometry."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from imas import ids_toplevel
import numpy as np
import pydantic
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.geometry import trapped_fraction
from torax._src.imas_tools.input import equilibrium as imas_geometry
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name,protected-access
class IMASSliceData(torax_pydantic.ConfigBase):
  """Equilibrium profiles and scalar quantities from an IMAS time slice.

  Attributes:
    psi: Poloidal flux profile [Wb].
    Phi: Toroidal flux profile [Wb].
    I_p_profile: Toroidal plasma current profile [A].
    R_in: Inboard major radius profile [m].
    R_out: Outboard major radius profile [m].
    F: Toroidal field flux function profile F = R * B_phi [m T].
    int_dl_over_Bp: Poloidal loop integral Int(R dl_p / |grad(psi)|).
    flux_surf_avg_1_over_R: Flux surface average <1/R> [m^-1].
    flux_surf_avg_1_over_R2: Flux surface average <1/R^2> [m^-2].
    flux_surf_avg_grad_psi: Flux surface average <|grad(psi)|> [Wb/m].
    flux_surf_avg_grad_psi2: Flux surface average <|grad(psi)|^2> [Wb^2/m^2].
    flux_surf_avg_grad_psi2_over_R2: Flux surface average <|grad(psi)|^2 / R^2>.
    flux_surf_avg_B2: Flux surface average <B^2> [T^2].
    flux_surf_avg_1_over_B2: Flux surface average <1/B^2> [T^-2].
    delta_upper_face: Upper triangularity profile.
    delta_lower_face: Lower triangularity profile.
    elongation: Elongation profile.
    vpr: Differential volume dV/drho_tor_norm [m^3].
    trapped_fraction: Effective trapped particle fraction profile.
    R_major: Major radius (geometric center of LCFS) [m].
    a_minor: Minor radius (half horizontal width of LCFS) [m].
    B_0: Vacuum toroidal magnetic field at `R_major` [T].
    z_magnetic_axis: Vertical coordinate of the magnetic axis [m].
    diverted: Whether the boundary configuration is diverted.
    r0_vacuum: Reference major radius of the input vacuum toroidal field [m].
  """

  psi: torax_pydantic.NumpyArray1D
  Phi: torax_pydantic.NumpyArray1D
  I_p_profile: torax_pydantic.NumpyArray1D
  R_in: torax_pydantic.NumpyArray1D
  R_out: torax_pydantic.NumpyArray1D
  F: torax_pydantic.NumpyArray1D
  int_dl_over_Bp: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_R: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_R2: torax_pydantic.NumpyArray1D
  flux_surf_avg_grad_psi: torax_pydantic.NumpyArray1D
  flux_surf_avg_grad_psi2: torax_pydantic.NumpyArray1D
  flux_surf_avg_grad_psi2_over_R2: torax_pydantic.NumpyArray1D
  flux_surf_avg_B2: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_B2: torax_pydantic.NumpyArray1D
  delta_upper_face: torax_pydantic.NumpyArray1D
  delta_lower_face: torax_pydantic.NumpyArray1D
  elongation: torax_pydantic.NumpyArray1D
  vpr: torax_pydantic.NumpyArray1D
  trapped_fraction: torax_pydantic.NumpyArray1D
  R_major: float
  a_minor: float
  B_0: float
  z_magnetic_axis: float | None = None
  diverted: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  r0_vacuum: float | None = None


def extract_slice_data(
    equilibrium: ids_toplevel.IDSToplevel,
    slice_index: int = 0,
    trapped_fraction_source: trapped_fraction.TrappedFractionSource = (
        trapped_fraction.TrappedFractionSource.SAUTER
    ),
) -> IMASSliceData:
  """Extracts an IMASSliceData container from an equilibrium time slice."""
  intermediates = imas_geometry._geometry_from_single_slice(
      equilibrium=equilibrium,
      face_centers=np.array([]),
      hires_factor=4,
      slice_index=slice_index,
      trapped_fraction_source=trapped_fraction_source,
  )
  cleaned = {}
  for k, v in intermediates.items():
    if k in IMASSliceData.model_fields:
      if isinstance(v, (int, float, bool, str)) or v is None:
        cleaned[k] = v
      else:
        cleaned[k] = np.asarray(v)
  return IMASSliceData.model_validate(cleaned)


class IMASConfig(base.GeometryConfigBase):
  """Loads an equilibrium Interface Data Structure (IDS) from IMAS.

  Currently written for COCOSv17 and DDv4. Note that `IMASConfig` is
  experimental and may undergo changes.

  Attributes:
    model_name: Always set to `'imas'`.
    imas_data: Single IMAS slice data or dictionary of time to slice data.

  Note:
    The constructor also accepts inputs normalized into imas_data:
      geometry_directory: Directory containing the IMAS file.
      imas_filepath: Path to the IMAS netCDF file containing equilibrium data.
      imas_uri: The IMAS uri containing the equilibrium data.
      equilibrium_object: The equilibrium IDS containing the relevant data.
      slice_time: Time slice in seconds to extract from the IMAS equilibrium.
      explicit_convert: Whether to explicitly convert the data.
  """

  @property
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction.TrappedFractionSource]:
    return frozenset({
        trapped_fraction.TrappedFractionSource.SAUTER,
        trapped_fraction.TrappedFractionSource.FILE,
        trapped_fraction.TrappedFractionSource.EXACT,
    })

  model_name: Annotated[
      Literal['imas'],
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = 'imas'
  imas_data: IMASSliceData | dict[torax_pydantic.Second, IMASSliceData]

  @pydantic.model_validator(mode='before')
  @classmethod
  def _before_validator(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    data_dict = dict(data)
    if 'imas_data' in data_dict:
      data_dict.pop('imas_filepath', None)
      data_dict.pop('imas_uri', None)
      data_dict.pop('equilibrium_object', None)
      data_dict.pop('geometry_directory', None)
      data_dict.pop('slice_time', None)
      data_dict.pop('slice_index', None)
      data_dict.pop('explicit_convert', None)
      return data_dict

    specified_inputs = [
        k
        for k in ['equilibrium_object', 'imas_uri', 'imas_filepath']
        if data_dict.get(k) is not None
    ]
    if len(specified_inputs) > 1:
      raise ValueError(
          'IMAS geometry builder needs exactly one of `equilibrium_object`, '
          '`imas_uri` or `imas_filepath` to be a valid input. Inputs provided: '
          f'{[data_dict[k] for k in specified_inputs]}.'
      )
    if not specified_inputs:
      raise ValueError(
          'IMAS geometry builder needs exactly one of `equilibrium_object`, '
          '`imas_uri` or `imas_filepath` to be a valid input.'
      )

    equilibrium_object = data_dict.pop('equilibrium_object', None)
    imas_uri = data_dict.pop('imas_uri', None)
    imas_filepath = data_dict.pop('imas_filepath', None)
    geometry_directory = data_dict.pop('geometry_directory', None)
    slice_time = data_dict.pop('slice_time', None)
    slice_index = data_dict.pop('slice_index', None)
    explicit_convert = data_dict.pop('explicit_convert', True)

    trapped_fraction_source = data_dict.get(
        'trapped_fraction_source',
        trapped_fraction.TrappedFractionSource.SAUTER,
    )
    if isinstance(trapped_fraction_source, str):
      trapped_fraction_source = trapped_fraction.TrappedFractionSource(
          trapped_fraction_source
      )

    equilibrium = imas_geometry._load_equilibrium(
        geometry_directory=geometry_directory,
        equilibrium_object=equilibrium_object,
        imas_uri=imas_uri,
        imas_filepath=imas_filepath,
        explicit_convert=explicit_convert,
    )

    if not equilibrium.time_slice or len(equilibrium.time_slice) == 0:
      data_dict['imas_data'] = {}
      return data_dict

    if slice_time is not None:
      times = np.asarray(equilibrium.time)
      idx = int(np.argmin(np.abs(times - slice_time)))
      data_dict['imas_data'] = extract_slice_data(
          equilibrium, idx, trapped_fraction_source
      )
    elif slice_index is not None:
      data_dict['imas_data'] = extract_slice_data(
          equilibrium, slice_index, trapped_fraction_source
      )
    elif len(equilibrium.time_slice) == 1:
      data_dict['imas_data'] = extract_slice_data(
          equilibrium, 0, trapped_fraction_source
      )
    else:
      times = np.asarray(equilibrium.time)
      data_dict['imas_data'] = {
          float(times[i]): extract_slice_data(
              equilibrium, i, trapped_fraction_source
          )
          for i in range(len(equilibrium.time_slice))
      }

    return data_dict

  def _build_slice_geometry(
      self, slice_data: IMASSliceData
  ) -> standard_geometry.StandardGeometry:
    intermediates = standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.IMAS,
        R_major=np.array(slice_data.R_major),
        a_minor=np.array(slice_data.a_minor),
        B_0=np.array(slice_data.B_0),
        psi=slice_data.psi,
        I_p_profile=slice_data.I_p_profile,
        Phi=slice_data.Phi,
        R_in=slice_data.R_in,
        R_out=slice_data.R_out,
        F=slice_data.F,
        int_dl_over_Bp=slice_data.int_dl_over_Bp,
        flux_surf_avg_1_over_R=slice_data.flux_surf_avg_1_over_R,
        flux_surf_avg_1_over_R2=slice_data.flux_surf_avg_1_over_R2,
        flux_surf_avg_grad_psi=slice_data.flux_surf_avg_grad_psi,
        flux_surf_avg_grad_psi2=slice_data.flux_surf_avg_grad_psi2,
        flux_surf_avg_grad_psi2_over_R2=(
            slice_data.flux_surf_avg_grad_psi2_over_R2
        ),
        flux_surf_avg_B2=slice_data.flux_surf_avg_B2,
        flux_surf_avg_1_over_B2=slice_data.flux_surf_avg_1_over_B2,
        trapped_fraction=slice_data.trapped_fraction,
        delta_upper_face=slice_data.delta_upper_face,
        delta_lower_face=slice_data.delta_lower_face,
        elongation=slice_data.elongation,
        vpr=slice_data.vpr,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
        z_magnetic_axis=(
            np.array(slice_data.z_magnetic_axis)
            if slice_data.z_magnetic_axis is not None
            else None
        ),
        diverted=slice_data.diverted,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
        r0_vacuum=(
            np.array(slice_data.r0_vacuum)
            if slice_data.r0_vacuum is not None
            else None
        ),
    )
    return standard_geometry.build_standard_geometry(intermediates)

  def _build_geometry(
      self,
  ) -> (
      standard_geometry.StandardGeometry
      | Mapping[float, standard_geometry.StandardGeometry]
  ):
    if not self.imas_data:
      raise ValueError('Equilibrium IDS contains no time slices.')
    if isinstance(self.imas_data, IMASSliceData):
      return self._build_slice_geometry(self.imas_data)
    return {
        float(t): self._build_slice_geometry(data)
        for t, data in self.imas_data.items()
    }
