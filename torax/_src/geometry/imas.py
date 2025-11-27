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
from typing import Annotated, Literal

from imas import ids_toplevel
import pydantic
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.imas_tools.input import equilibrium as imas_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions


# pylint: disable=invalid-name
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
    slice_index: Index of slice to load from IMAS IDS.
  """

  geometry_type: Annotated[Literal['imas'], torax_pydantic.TIME_INVARIANT] = (
      'imas'
  )
  n_rho: Annotated[pydantic.PositiveInt, torax_pydantic.TIME_INVARIANT] = 25
  hires_factor: pydantic.PositiveInt = 4
  geometry_directory: Annotated[str | None, torax_pydantic.TIME_INVARIANT] = (
      None
  )
  Ip_from_parameters: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True
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
    inputs = imas_geometry.geometry_from_IMAS(
        geometry_directory=self.geometry_directory,
        equilibrium_object=self.equilibrium_object,
        imas_uri=self.imas_uri,
        imas_filepath=self.imas_filepath,
        Ip_from_parameters=self.Ip_from_parameters,
        n_rho=self.n_rho,
        hires_factor=self.hires_factor,
        slice_time=self.slice_time,
        slice_index=self.slice_index,
    )
    intermediates = standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.IMAS, **inputs
    )

    return standard_geometry.build_standard_geometry(intermediates)
