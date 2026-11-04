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

from typing import Annotated, Literal, Self

from imas import ids_toplevel
import pydantic
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_provider
from torax._src.geometry import standard_geometry
from torax._src.geometry import trapped_fraction
from torax._src.imas_tools.input import equilibrium as imas_geometry
from torax._src.torax_pydantic import torax_pydantic


# pylint: disable=invalid-name
class IMASConfig(base.GeometryConfigBase):
  """Loads an equilibrium Interface Data Structure (IDS) from IMAS.

  Currently written for COCOSv17 and DDv4. Note that `IMASConfig` is
  experimental and may undergo changes.

  Attributes:
    geometry_type: Always set to `'imas'`.
    geometry_directory: Optionally overrides the default geometry directory
      (`torax/data/third_party/geo`). Should be set to an absolute path.
    imas_filepath: Path to the IMAS netCDF file containing the equilibrium IDS
      data. Only one of `imas_filepath`, `imas_uri`, or `equilibrium_object` can
      be set.
    imas_uri: IMAS URI containing the equilibrium IDS data. Using this option
      requires access to `imas-core` (not part of standard TORAX dependencies).
      Only one of `imas_filepath`, `imas_uri`, or `equilibrium_object` can be
      set.
    equilibrium_object: In-memory `imas.ids_toplevel.IDSToplevel` equilibrium
      IDS object inserted directly. Does not currently support serialization and
      requires a custom run loop. Only one of `imas_filepath`, `imas_uri`, or
      `equilibrium_object` can be set.
    slice_time: Time [s] of a single slice to load from the IMAS IDS. If `None`
      (default), all time slices in the IDS are loaded to build a time-dependent
      `StandardGeometryProvider`. If a float is provided, only the single time
      slice closest to the given time is loaded.
    explicit_convert: Whether to explicitly convert the IDS to the current Data
      Dictionary (DD) version before loading. Explicit conversion is required
      when converting between major DD versions. See the |imas_multi_dd_docs|
      for details.
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

  geometry_type: Annotated[Literal['imas'], torax_pydantic.TIME_INVARIANT] = (
      'imas'
  )
  geometry_directory: Annotated[str | None, torax_pydantic.TIME_INVARIANT] = (
      None
  )
  imas_filepath: str | None = None
  imas_uri: str | None = None
  equilibrium_object: ids_toplevel.IDSToplevel | None = None
  slice_time: float | None = None
  explicit_convert: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True

  @pydantic.model_validator(mode='after')
  def _validate_model(self) -> Self:
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
    """Builds a StandardGeometry from a single IDS time slice.

    Loads a single time slice from the equilibrium IDS and returns a
    `StandardGeometry`. If `slice_time` is set, the slice closest in time is
    selected; otherwise, the first time slice (index 0) is loaded.

    Returns:
      A StandardGeometry built from the selected time slice.
    """
    inputs = imas_geometry.geometry_from_single_IMAS_slice(
        geometry_directory=self.geometry_directory,
        equilibrium_object=self.equilibrium_object,
        imas_uri=self.imas_uri,
        imas_filepath=self.imas_filepath,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
        explicit_convert=self.explicit_convert,
        slice_time=self.slice_time,
        trapped_fraction_source=self.trapped_fraction_source,
    )
    intermediates = standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.IMAS, **inputs
    )
    return standard_geometry.build_standard_geometry(intermediates)

  def build_geometry_provider(
      self,
      calc_Phi_b_dot: bool,
  ) -> geometry_provider.GeometryProvider:
    """Builds a time-dependent GeometryProvider from all IDS time slices.

    Loads all time slices from the equilibrium IDS and returns a
    `StandardGeometryProvider` that interpolates geometry across them.

    Args:
      calc_Phi_b_dot: Whether to calculate Phi_b_dot.

    Returns:
      A StandardGeometryProvider interpolating across all time slices.
    """
    all_inputs = imas_geometry.geometry_from_IMAS(
        geometry_directory=self.geometry_directory,
        equilibrium_object=self.equilibrium_object,
        imas_uri=self.imas_uri,
        imas_filepath=self.imas_filepath,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
        explicit_convert=self.explicit_convert,
        trapped_fraction_source=self.trapped_fraction_source,
    )
    geometries = {}
    for t, inputs in all_inputs.items():
      intermediates = standard_geometry.StandardGeometryIntermediates(
          geometry_type=geometry.GeometryType.IMAS, **inputs
      )
      geometries[t] = standard_geometry.build_standard_geometry(intermediates)
    return standard_geometry.StandardGeometryProvider.create_provider(
        geometries,
        calc_Phi_b_dot=calc_Phi_b_dot,
    )
