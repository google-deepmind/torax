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
"""Classes for loading and representing a MEQ geometry."""

from collections.abc import Mapping
import enum
import logging
import pathlib
from typing import Annotated, Any, Literal, Self, TypeAlias

import jax
import numpy as np
import pydantic
from torax._src import constants
from torax._src import math_utils
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
from torax._src.geometry import trapped_fraction as trapped_fraction_lib
from torax._src.torax_pydantic import torax_pydantic

# pylint: disable=invalid-name
LY_OBJECT_TYPE: TypeAlias = (
    str | Mapping[str, torax_pydantic.NumpyArray | float]
)

# Edge model geometry keys expected in LY data.
_PROFILE_KEYS = (
    'TQ',
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
    'aminor',
    'rgeom',
    'epsilon',
    'FtPVQ',
)
_SCALAR_TIME_KEYS = ('rBt', 'FB', 'FA', 'zA', 'lX', 't')
_EDGE_MODEL_OMP_KEYS = frozenset({'r_OMP', 'Bp_OMP'})
_EDGE_MODEL_TARGET_KEYS = frozenset({
    'Lpar_target',
    'Lpar_div',
    'alpha_target',
    'r_target',
})
_ALL_EDGE_KEYS = _EDGE_MODEL_OMP_KEYS | _EDGE_MODEL_TARGET_KEYS
_REQUIRED_KEYS = (
    tuple(k for k in _PROFILE_KEYS if k != 'FtPVQ') + _SCALAR_TIME_KEYS
)


def _is_bundle_data(data: Mapping[str, Any]) -> bool:
  """Returns whether the mapping represents a time bundle of slices."""
  if 't' in data:
    t = data['t']
    if isinstance(t, (int, float, np.floating, np.integer)):
      return False
    if isinstance(t, np.ndarray):
      return t.size > 1
    if isinstance(t, (list, tuple)):
      if len(t) == 2 and isinstance(t[0], str) and isinstance(t[1], list):
        return len(t[1]) > 1
      return len(t) > 1
  if 'TQ' in data:
    tq = data['TQ']
    if isinstance(tq, np.ndarray):
      return tq.ndim == 2
    if isinstance(tq, (list, tuple)):
      if len(tq) == 2 and isinstance(tq[0], str) and isinstance(tq[1], list):
        return bool(tq[1] and isinstance(tq[1][0], (list, tuple)))
      return bool(tq and isinstance(tq[0], (list, tuple)))
  return False


def _clean_meq_ly_dict(
    data: Any, *, target_cls: type[Any], is_bundle: bool
) -> Any:
  """Cleans and validates an input mapping for MEQLYSlice or MEQLYBundle."""
  if not isinstance(data, Mapping):
    if 'L' not in data:
      raise ValueError("Either 'L' or 'L_object' must be provided.")
    if 'LY' not in data:
      raise ValueError(
          "One of 'LY', 'LY_object', or 'LY_bundle_object' must be provided."
      )
    return data
  cleaned = dict(data)
  for k in ('__header__', '__version__', '__globals__', 'shot'):
    cleaned.pop(k, None)

  # FtPVQ / FtPQ fallback
  if 'FtPVQ' not in cleaned and 'FtPQ' in cleaned:
    logging.warning(
        'FtPVQ not found in LY data, using FtPQ instead. Please upgrade to'
        ' a newer version of MEQ as the source of the LY data. This will'
        ' throw an error in a future version.'
    )
    cleaned['FtPVQ'] = cleaned.pop('FtPQ')
  elif 'FtPVQ' not in cleaned and 'FtPQ' not in cleaned:
    raise ValueError(
        'LY data is missing a toroidal flux-related key '
        'provide either FtPVQ or FtPQ.'
    )

  for key in _REQUIRED_KEYS:
    if key not in cleaned:
      raise ValueError(f"LY data is missing the '{key}' key.")

  present_optional = _ALL_EDGE_KEYS.intersection(cleaned.keys())
  if present_optional and len(present_optional) != len(_ALL_EDGE_KEYS):
    logging.warning(
        'Some but not all optional edge model keys are present in LY data. '
        'This may lead to incomplete edge model initialization. '
        'Missing data: %s',
        _ALL_EDGE_KEYS - present_optional,
    )
    for k in present_optional:
      cleaned.pop(k, None)

  if is_bundle:
    for key in _SCALAR_TIME_KEYS:
      if isinstance(cleaned[key], np.ndarray):
        cleaned[key] = np.atleast_1d(cleaned[key].squeeze())
    for key in ('r_OMP', 'Bp_OMP'):
      if key in cleaned and isinstance(cleaned[key], np.ndarray):
        cleaned[key] = np.atleast_1d(cleaned[key].squeeze())
  else:
    for key in _PROFILE_KEYS:
      if isinstance(cleaned[key], np.ndarray):
        cleaned[key] = np.atleast_1d(cleaned[key].squeeze())
    for key in ('rBt', 'FB', 'FA', 'zA', 't'):
      val = cleaned[key]
      if isinstance(val, (int, float, np.number)):
        cleaned[key] = float(val)
      elif isinstance(val, np.ndarray):
        if val.size != 1:
          raise ValueError(
              f"Incorrect shape for key '{key}' in LY data. Expected scalar,"
              f' got {val.shape}.'
          )
        cleaned[key] = float(val.squeeze())
    if 'lX' in cleaned:
      val = cleaned['lX']
      if isinstance(val, (int, np.integer)):
        cleaned['lX'] = int(val)
      elif isinstance(val, np.ndarray):
        if val.size != 1:
          raise ValueError(
              "Incorrect shape for key 'lX' in LY data. Expected scalar,"
              f' got {val.shape}.'
          )
        cleaned['lX'] = int(val.squeeze())
    for key in ('r_OMP', 'Bp_OMP'):
      if key in cleaned and cleaned[key] is not None:
        val = cleaned[key]
        if isinstance(val, (int, float, np.number)):
          cleaned[key] = float(val)
        elif isinstance(val, np.ndarray):
          if val.size != 1:
            raise ValueError(
                f"Incorrect shape for key '{key}' in LY data. Expected scalar,"
                f' got {val.shape}.'
            )
          cleaned[key] = float(val.squeeze())
    for key in ('Lpar_target', 'Lpar_div', 'alpha_target', 'r_target'):
      if key in cleaned and cleaned[key] is not None:
        val = cleaned[key]
        if isinstance(val, np.ndarray):
          val = val.squeeze()
          if val.ndim == 0:
            cleaned[key] = float(val)
          elif val.ndim == 1:
            cleaned[key] = val
          else:
            raise ValueError(
                f"Incorrect shape for key '{key}' in LY data. Expected scalar"
                f' or 1D array, got {val.shape}.'
            )
  known_fields = set(target_cls.model_fields.keys()).union({'FtPVQ'})
  return {k: v for k, v in cleaned.items() if k in known_fields}


@enum.unique
class DivertorDomain(enum.StrEnum):
  """Enum for selecting the divertor domain in the extended Lengyel model."""

  UPPER_NULL = 'upper_null'
  LOWER_NULL = 'lower_null'


class MEQLData(torax_pydantic.ConfigBase):
  """Normalized poloidal flux coordinate data from MEQ L file."""

  pQ: torax_pydantic.NumpyArray1D

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_meq_l_data(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    if 'pQ' not in data:
      raise ValueError("L data is missing the 'pQ' key.")
    if isinstance(data['pQ'], np.ndarray):
      pQ = np.atleast_1d(data['pQ'].squeeze())
      if pQ.ndim != 1:
        raise ValueError(
            "Incorrect shape for key 'pQ' in L data. Expected 1D array, got"
            f' {pQ.shape}.'
        )
      return {'pQ': pQ}
    return {'pQ': data['pQ']}

  @pydantic.model_validator(mode='after')
  def _validate_shape(self) -> Self:
    if self.pQ.ndim != 1:
      raise ValueError(
          "Incorrect shape for key 'pQ' in L data. Expected 1D array, got"
          f' {self.pQ.shape}.'
      )
    return self

  def to_l_dict(self) -> dict[str, Any]:
    return {'pQ': self.pQ}


class MEQLYSlice(torax_pydantic.ConfigBase):
  """Single time slice of MEQ LY geometry data."""

  TQ: torax_pydantic.NumpyArray1D
  Q0Q: torax_pydantic.NumpyArray1D
  Q1Q: torax_pydantic.NumpyArray1D
  Q2Q: torax_pydantic.NumpyArray1D
  Q3Q: torax_pydantic.NumpyArray1D
  Q4Q: torax_pydantic.NumpyArray1D
  Q5Q: torax_pydantic.NumpyArray1D
  ItQ: torax_pydantic.NumpyArray1D
  deltau: torax_pydantic.NumpyArray1D
  deltal: torax_pydantic.NumpyArray1D
  kappa: torax_pydantic.NumpyArray1D
  FtPVQ: torax_pydantic.NumpyArray1D
  aminor: torax_pydantic.NumpyArray1D
  rgeom: torax_pydantic.NumpyArray1D
  epsilon: torax_pydantic.NumpyArray1D

  rBt: float
  FB: float
  FA: float
  zA: float
  lX: Annotated[int, torax_pydantic.JAX_STATIC]
  t: float

  r_OMP: float | None = None
  Bp_OMP: float | None = None
  Lpar_target: float | torax_pydantic.NumpyArray1D | None = None
  Lpar_div: float | torax_pydantic.NumpyArray1D | None = None
  alpha_target: float | torax_pydantic.NumpyArray1D | None = None
  r_target: float | torax_pydantic.NumpyArray1D | None = None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_meq_ly_slice(cls, data: Any) -> Any:
    return _clean_meq_ly_dict(data, target_cls=cls, is_bundle=False)

  @pydantic.model_validator(mode='after')
  def _validate_arrays(self) -> Self:
    n_psi = len(self.TQ)
    for key in _PROFILE_KEYS:
      val = getattr(self, key)
      if val.ndim != 1 or len(val) != n_psi:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. Expected ({n_psi},),"
            f' got {val.shape}.'
        )
    return self

  def to_slice_dict(self) -> dict[str, Any]:
    return {
        k: getattr(self, k)
        for k in self.__class__.model_fields
        if getattr(self, k) is not None
    }


class MEQLYBundle(torax_pydantic.ConfigBase):
  """Multi-slice bundled MEQ LY geometry data."""

  TQ: torax_pydantic.NumpyArray
  Q0Q: torax_pydantic.NumpyArray
  Q1Q: torax_pydantic.NumpyArray
  Q2Q: torax_pydantic.NumpyArray
  Q3Q: torax_pydantic.NumpyArray
  Q4Q: torax_pydantic.NumpyArray
  Q5Q: torax_pydantic.NumpyArray
  ItQ: torax_pydantic.NumpyArray
  deltau: torax_pydantic.NumpyArray
  deltal: torax_pydantic.NumpyArray
  kappa: torax_pydantic.NumpyArray
  FtPVQ: torax_pydantic.NumpyArray
  aminor: torax_pydantic.NumpyArray
  rgeom: torax_pydantic.NumpyArray
  epsilon: torax_pydantic.NumpyArray

  rBt: torax_pydantic.NumpyArray1D
  FB: torax_pydantic.NumpyArray1D
  FA: torax_pydantic.NumpyArray1D
  zA: torax_pydantic.NumpyArray1D
  lX: torax_pydantic.NumpyArray1D
  t: torax_pydantic.NumpyArray1D

  r_OMP: torax_pydantic.NumpyArray1D | None = None
  Bp_OMP: torax_pydantic.NumpyArray1D | None = None
  Lpar_target: torax_pydantic.NumpyArray | None = None
  Lpar_div: torax_pydantic.NumpyArray | None = None
  alpha_target: torax_pydantic.NumpyArray | None = None
  r_target: torax_pydantic.NumpyArray | None = None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_meq_ly_bundle(cls, data: Any) -> Any:
    return _clean_meq_ly_dict(data, target_cls=cls, is_bundle=True)

  @pydantic.model_validator(mode='after')
  def _validate_bundle_shapes(self) -> Self:
    n_times = len(self.t)
    if self.TQ.ndim != 2:
      raise ValueError(
          "Incorrect shape for key 'TQ' in LY data. Expected 2D array,"
          f' got {self.TQ.shape}.'
      )
    n_psi = self.TQ.shape[0]
    expected_2d = (n_psi, n_times)
    expected_1d = (n_times,)

    for key in _PROFILE_KEYS:
      arr = getattr(self, key)
      if arr.shape != expected_2d:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. Expected"
            f' {expected_2d}:, got {arr.shape}.'
        )

    for key in _SCALAR_TIME_KEYS:
      arr = getattr(self, key)
      if arr.shape != expected_1d:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. Expected"
            f' {expected_1d}:, got {arr.shape}.'
        )

    for key in ('r_OMP', 'Bp_OMP'):
      arr = getattr(self, key)
      if arr is not None and arr.shape != expected_1d:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. Expected"
            f' {expected_1d}:, got {arr.shape}.'
        )
    return self

  def get_slice_dict(self, idx: int) -> dict[str, Any]:
    """Returns a single slice dictionary from the bundle at time index idx."""
    slice_dict = {key: getattr(self, key)[..., idx] for key in _PROFILE_KEYS}
    for key in _SCALAR_TIME_KEYS:
      val = getattr(self, key)[idx]
      slice_dict[key] = int(val) if key == 'lX' else float(val)
    for key in _ALL_EDGE_KEYS:
      val = getattr(self, key)
      if val is not None:
        slice_dict[key] = val[..., idx]
    return slice_dict


class MEQConfig(base.GeometryConfigBase):
  """Loads MEQ (e.g., FBT, FGE, or LIUQE) geometry files or in-memory dicts.

  Note that `<B^2>` and `<1/B^2>` are not currently supported by MEQ geometry
  and are approximated using analytical circular-geometry expressions, which may
  introduce inaccuracies in neoclassical transport.

  Attributes:
    model_name: Always set to `'meq'`.
    L: MEQ L data object, or mapping of timestamp to MEQ L data object.
    LY: MEQ LY slice, bundle, or mapping of timestamp to MEQ LY slice.
    LY_to_torax_times: TORAX simulation times [s] corresponding to the
      individual slices in `LY_bundle_object`. If not provided, the times are
      taken from the bundle itself. Its length must match the number of slices
      in the bundle.
    divertor_domain: Selects the divertor domain used to extract edge geometry
      parameters for the :ref:`extended_lengyel` edge model when diverted.
      Either `'lower_null'` or `'upper_null'`.

  Note:
    The constructor also accepts inputs normalized into L and LY:
      geometry_directory: Directory containing geometry files.
      L_object: Single MEQ L geometry file path or dict.
      LY_object: Single-slice MEQ LY geometry file path or dict.
      LY_bundle_object: Multi-slice MEQ LY bundle file path or dict.
  """

  @property
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction_lib.TrappedFractionSource]:
    return frozenset({trapped_fraction_lib.TrappedFractionSource.SAUTER})

  model_name: Annotated[
      Literal['meq'],
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = 'meq'
  L: MEQLData | dict[torax_pydantic.Second, MEQLData]
  LY: MEQLYSlice | MEQLYBundle | dict[torax_pydantic.Second, MEQLYSlice]
  LY_to_torax_times: torax_pydantic.NumpyArray1D | None = None
  divertor_domain: Annotated[DivertorDomain, torax_pydantic.JAX_STATIC] = (
      DivertorDomain.LOWER_NULL
  )

  def __init__(
      self,
      L: MEQLData | dict[torax_pydantic.Second, MEQLData] | None = None,
      LY: (
          MEQLYSlice
          | MEQLYBundle
          | dict[torax_pydantic.Second, MEQLYSlice]
          | None
      ) = None,
      L_object: Any = None,
      LY_object: Any = None,
      LY_bundle_object: Any = None,
      geometry_directory: str | None = None,
      **kwargs: Any,
  ):
    data: dict[str, Any] = {}
    if L is not None:
      data['L'] = L
    if LY is not None:
      data['LY'] = LY
    if L_object is not None:
      data['L_object'] = L_object
    if LY_object is not None:
      data['LY_object'] = LY_object
    if LY_bundle_object is not None:
      data['LY_bundle_object'] = LY_bundle_object
    if geometry_directory is not None:
      data['geometry_directory'] = geometry_directory
    data.update(kwargs)
    super().__init__(**data)

  @property
  def LY_bundle_object(self) -> MEQLYBundle | None:
    return self.LY if isinstance(self.LY, MEQLYBundle) else None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_data(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    data = dict(data)

    # 2. Check conflicting arguments
    if 'LY_bundle_object' in data and 'LY_object' in data:
      raise ValueError(
          "Cannot use 'LY_object' together with a bundled MEQ file"
      )
    if 'LY' in data and ('LY_object' in data or 'LY_bundle_object' in data):
      raise ValueError(
          "Cannot specify both 'LY' and 'LY_object'/'LY_bundle_object'"
      )

    geo_dir = data.pop('geometry_directory', None)

    # 3. Resolve L_object
    if 'L_object' in data:
      l_val = data.pop('L_object')
      if isinstance(l_val, (str, pathlib.Path)):
        data['L'] = geometry_loader.load_geo_data(
            geo_dir, str(l_val), geometry_loader.GeometrySource.MEQ
        )
      elif isinstance(l_val, Mapping):
        if 'pQ' in l_val:
          data['L'] = dict(l_val)
        else:
          loaded_l = {}
          for t, val in l_val.items():
            if isinstance(val, (str, pathlib.Path)):
              val = geometry_loader.load_geo_data(
                  geo_dir, str(val), geometry_loader.GeometrySource.MEQ
              )
            loaded_l[float(t)] = val
          data['L'] = loaded_l
      else:
        raise ValueError(
            'L_object must be a string (file path) or a dictionary.'
        )

    # 4. Resolve LY_object
    if 'LY_object' in data:
      ly_val = data.pop('LY_object')
      if isinstance(ly_val, (str, pathlib.Path)):
        data['LY'] = geometry_loader.load_geo_data(
            geo_dir, str(ly_val), geometry_loader.GeometrySource.MEQ
        )
      elif isinstance(ly_val, Mapping):
        if 'TQ' in ly_val:
          data['LY'] = dict(ly_val)
        else:
          loaded_ly = {}
          for t, val in ly_val.items():
            if isinstance(val, (str, pathlib.Path)):
              val = geometry_loader.load_geo_data(
                  geo_dir, str(val), geometry_loader.GeometrySource.MEQ
              )
            loaded_ly[float(t)] = val
          data['LY'] = loaded_ly
      else:
        raise ValueError(
            'LY_object must be a string (file path) or a dictionary.'
        )

    # 5. Resolve LY_bundle_object
    if 'LY_bundle_object' in data:
      ly_b_val = data.pop('LY_bundle_object')
      if isinstance(ly_b_val, (str, pathlib.Path)):
        data['LY'] = geometry_loader.load_geo_data(
            geo_dir, str(ly_b_val), geometry_loader.GeometrySource.MEQ
        )
      elif isinstance(ly_b_val, Mapping):
        data['LY'] = dict(ly_b_val)
      else:
        raise ValueError(
            'LY_bundle_object must be a string (file path) or a dictionary.'
        )
      if isinstance(data['LY'], Mapping) and not isinstance(
          data['LY'], (MEQLYSlice, MEQLYBundle)
      ):
        data['LY'] = MEQLYBundle.model_validate(data['LY'])

    # Convert L to MEQLData if Mapping
    if (
        'L' in data
        and isinstance(data['L'], Mapping)
        and not isinstance(data['L'], MEQLData)
    ):
      if 'pQ' in data['L']:
        data['L'] = MEQLData.model_validate(data['L'])
      else:
        data['L'] = {
            float(t): MEQLData.model_validate(v) for t, v in data['L'].items()
        }

    # Convert LY if Mapping
    if (
        'LY' in data
        and isinstance(data['LY'], Mapping)
        and not isinstance(data['LY'], (MEQLYSlice, MEQLYBundle))
    ):
      if 'TQ' in data['LY']:
        if _is_bundle_data(data['LY']):
          data['LY'] = MEQLYBundle.model_validate(data['LY'])
        else:
          data['LY'] = MEQLYSlice.model_validate(data['LY'])
      else:
        data['LY'] = {
            float(t): MEQLYSlice.model_validate(v)
            for t, v in data['LY'].items()
        }

    return data

  def _resolve_l_for_slice(self, t: float | None = None) -> Mapping[str, Any]:
    if isinstance(self.L, MEQLData):
      return self.L.to_l_dict()
    if isinstance(self.L, Mapping):
      if not self.L:
        return {}
      if t is not None and t in self.L:
        val = self.L[t]
      else:
        val = next(iter(self.L.values()))
      return val.to_l_dict() if isinstance(val, MEQLData) else val
    return self.L

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    if self.LY_to_torax_times is not None:
      if not isinstance(self.LY, MEQLYBundle):
        raise ValueError(
            'LY_bundle_object must be set when using LY_to_torax_times.'
        )
      if len(self.LY_to_torax_times) != len(self.LY.t):
        raise ValueError(
            'Length of LY_to_torax_times must match length of LY bundle data:'
            f' len(LY_to_torax_times)={len(self.LY_to_torax_times)},'
            f" len(LY_bundle['t'])={len(self.LY.t)}"
        )

    # Check pQ length matching
    if isinstance(self.L, MEQLData):
      n_psi = len(self.L.pQ)
      if isinstance(self.LY, MEQLYSlice):
        if len(self.LY.TQ) != n_psi:
          raise ValueError(
              f"Incorrect shape for key 'TQ' in LY data. Expected ({n_psi},),"
              f' got {self.LY.TQ.shape}.'
          )
      elif isinstance(self.LY, MEQLYBundle):
        if self.LY.TQ.shape[0] != n_psi:
          raise ValueError(
              f"Incorrect shape for key 'TQ' in LY data. Expected ({n_psi},"
              f' {len(self.LY.t)}), got {self.LY.TQ.shape}.'
          )
      elif isinstance(self.LY, dict):
        for t, slice_data in self.LY.items():
          if len(slice_data.TQ) != n_psi:
            raise ValueError(
                f"Incorrect shape for key 'TQ' in LY data at time {t}. Expected"
                f' ({n_psi},), got {slice_data.TQ.shape}.'
            )
    return self

  def _build_geometry(
      self,
  ) -> (
      standard_geometry.StandardGeometry
      | Mapping[float, standard_geometry.StandardGeometry]
  ):
    if isinstance(self.LY, MEQLYSlice):
      if self.LY_to_torax_times is not None:
        raise ValueError('LY_to_torax_times can only be used with MEQLYBundle.')
      intermediates = _from_meq(
          LY=self.LY.to_slice_dict(),
          L=self._resolve_l_for_slice(),
          face_centers=self.get_face_centers(),
          hires_factor=self.hires_factor,
          divertor_domain=self.divertor_domain,
          trapped_fraction_source=self.trapped_fraction_source,
      )
      return standard_geometry.build_standard_geometry(intermediates)
    elif isinstance(self.LY, MEQLYBundle):
      times = (
          self.LY_to_torax_times
          if self.LY_to_torax_times is not None
          else self.LY.t
      )
      if len(times) != len(self.LY.t):
        raise ValueError(
            'Length of LY_to_torax_times must match length of LY bundle data:'
            f' len(LY_to_torax_times)={len(times)},'
            f" len(LY_bundle['t'])={len(self.LY.t)}"
        )
      l_dict = self._resolve_l_for_slice()
      geos = {}
      for idx in range(len(times)):
        slice_dict = self.LY.get_slice_dict(idx)
        intermediates = _from_meq(
            LY=slice_dict,
            L=l_dict,
            face_centers=self.get_face_centers(),
            hires_factor=self.hires_factor,
            divertor_domain=self.divertor_domain,
            trapped_fraction_source=self.trapped_fraction_source,
        )
        geos[float(times[idx])] = standard_geometry.build_standard_geometry(
            intermediates
        )
      if len(geos) == 1:
        return next(iter(geos.values()))
      return geos
    elif isinstance(self.LY, dict):
      geos = {}
      for t in sorted(self.LY.keys()):
        ly_slice = self.LY[t]
        ly_dict = (
            ly_slice.to_slice_dict()
            if isinstance(ly_slice, MEQLYSlice)
            else ly_slice
        )
        intermediates = _from_meq(
            LY=ly_dict,
            L=self._resolve_l_for_slice(t),
            face_centers=self.get_face_centers(),
            hires_factor=self.hires_factor,
            divertor_domain=self.divertor_domain,
            trapped_fraction_source=self.trapped_fraction_source,
        )
        geos[float(t)] = standard_geometry.build_standard_geometry(
            intermediates
        )
      if len(geos) == 1:
        return next(iter(geos.values()))
      return geos
    else:
      raise TypeError(f'Unsupported LY type: {type(self.LY)}')


def _from_meq_single_slice(
    geometry_directory: str | None,
    LY_object: str | Mapping[str, np.ndarray],
    L_object: str | Mapping[str, np.ndarray],
    face_centers: np.ndarray,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
    trapped_fraction_source: (
        trapped_fraction_lib.TrappedFractionSource
    ) = trapped_fraction_lib.TrappedFractionSource.SAUTER,
) -> standard_geometry.StandardGeometryIntermediates:
  """Returns StandardGeometryIntermediates from a single slice MEQ LY file.

  LY and L are MEQ data files containing magnetic geometry information.
  The majority of the needed information is in the LY file. The L file
  is only needed to get the normalized poloidal flux coordinate, pQ.

  This method is for cases when the LY file on disk corresponds to a single
  time slice. Either a single time slice or sequence of time slices can be
  provided in the geometry config.

  Args:
    geometry_directory: Directory where to find the MEQ file describing the
      magnetic geometry. If None, then it defaults to another dir. See
      `load_geo_data` implementation.
    LY_object: File name for LY data, or directly an LY single slice dict.
    L_object: File name for L data, or directly an L dict.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.
    trapped_fraction_source: Selects how the effective trapped particle fraction
      is computed; see `trapped_fraction.TrappedFractionSource`.

  Returns:
    A StandardGeometryIntermediates instance based on the input slice. This
    can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """
  if isinstance(LY_object, str):
    LY = geometry_loader.load_geo_data(
        geometry_directory, LY_object, geometry_loader.GeometrySource.MEQ
    )
  elif isinstance(LY_object, Mapping):
    LY = LY_object
  else:
    raise ValueError('LY_object must be a string (file path) or a dictionary.')
  if isinstance(L_object, str):
    L = geometry_loader.load_geo_data(
        geometry_directory, L_object, geometry_loader.GeometrySource.MEQ
    )
  elif isinstance(L_object, Mapping):
    L = L_object
  else:
    raise ValueError('L_object must be a string (file path) or a dictionary.')

  # Convert any scalar LY values to ndarrays such that validation method works
  for key in LY:
    if not isinstance(LY[key], np.ndarray):
      LY[key] = np.array(LY[key])  # pyrefly: ignore[unsupported-operation]

  LY, L = jax.tree_util.tree_map(np.squeeze, (LY, L))

  # Raises a ValueError if the data is invalid.
  _validate_meq_data(LY, L)
  return _from_meq(
      LY,
      L,
      face_centers,
      hires_factor,
      divertor_domain,
      trapped_fraction_source=trapped_fraction_source,
  )


def _from_meq_bundle(
    geometry_directory: str | None,
    LY_bundle_object: str | Mapping[str, np.ndarray],
    L_object: str | Mapping[str, np.ndarray],
    LY_to_torax_times: np.ndarray | None,
    face_centers: np.ndarray,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
    trapped_fraction_source: (
        trapped_fraction_lib.TrappedFractionSource
    ) = trapped_fraction_lib.TrappedFractionSource.SAUTER,
) -> Mapping[float, standard_geometry.StandardGeometryIntermediates]:
  """Returns StandardGeometryIntermediates from a bundled MEQ LY file.

  LY_bundle_object is an MEQ data object containing a bundle of LY geometry
  slices at different times, packaged within a single object (as opposed to
  a sequence of standalone LY files). LY_to_torax_times is a 1D array of
  times, defining the times in the TORAX simulation corresponding to each
  slice in the LY bundle. All times in the LY bundle must be mapped to
  times in TORAX. The LY_bundle_object and L_object can either be file names
  for disk loading, or directly the data dicts.

  Args:
    geometry_directory: Directory where to find the MEQ file describing the
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
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.
    trapped_fraction_source: Selects how the effective trapped particle fraction
      is computed; see `trapped_fraction.TrappedFractionSource`.

  Returns:
    A mapping from user-provided (or inferred) times to
    StandardGeometryIntermediates instances based on the input slices. This
    can then be used to build StandardGeometry instances.
  """

  if isinstance(LY_bundle_object, str):
    LY_bundle = geometry_loader.load_geo_data(
        geometry_directory,
        LY_bundle_object,
        geometry_loader.GeometrySource.MEQ,
    )
  elif isinstance(LY_bundle_object, Mapping):
    LY_bundle = LY_bundle_object
  else:
    raise ValueError(
        'LY_bundle_object must be a string (file path) or a dictionary.'
    )

  if isinstance(L_object, str):
    L = geometry_loader.load_geo_data(
        geometry_directory, L_object, geometry_loader.GeometrySource.MEQ
    )
  elif isinstance(L_object, Mapping):
    L = L_object
  else:
    raise ValueError('L_object must be a string (file path) or a dictionary.')

  LY_bundle, L = jax.tree_util.tree_map(np.squeeze, (LY_bundle, L))

  # Raises a ValueError if the data is invalid.
  _validate_meq_data(LY_bundle, L)

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
    intermediates[t] = _from_meq(
        data_slice,
        L,
        face_centers,
        hires_factor,
        divertor_domain,
        trapped_fraction_source=trapped_fraction_source,
    )

  return intermediates


def _get_LY_single_slice_from_bundle(
    LY_bundle: Mapping[str, np.ndarray],
    idx: int,
) -> Mapping[str, np.ndarray]:
  """Returns a single LY slice from a bundled LY file, at index idx."""

  # The keys below are the required LY keys for the MEQ geometry provider.
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

  # The keys below are the optional LY keys for the MEQ geometry provider.
  # They are used to extract edge quantities for extended Lengyel.
  optional_keys = list(_ALL_EDGE_KEYS)

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


def _from_meq(
    LY: Mapping[str, np.ndarray | int],
    L: Mapping[str, np.ndarray],
    face_centers: np.ndarray,
    hires_factor: int = 4,
    divertor_domain: DivertorDomain = DivertorDomain.LOWER_NULL,
    trapped_fraction_source: (
        trapped_fraction_lib.TrappedFractionSource
    ) = trapped_fraction_lib.TrappedFractionSource.SAUTER,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from a single MEQ LY slice.

  Args:
    LY: A dictionary of relevant MEQ LY geometry data.
    L: A dictionary of relevant MEQ L geometry data.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations on initialization.
    divertor_domain: The divertor domain (upper or lower null) for extracting
      edge quantities when diverted.
    trapped_fraction_source: Selects how the effective trapped particle fraction
      is computed; see `trapped_fraction.TrappedFractionSource`.

  Returns:
    A StandardGeometryIntermediates instance based on the input slice. This
    can then be used to build a StandardGeometry by passing to
    `build_standard_geometry`.
  """
  # This is kept as an input argument while we fully investigate the best way
  # to chose the edge quantities to use.
  del divertor_domain
  # lX is a flag for diverted (1) or limited (0) geometry. Converted to
  # boolean when constructing the StandardGeometryIntermediates.
  if LY['lX'] not in [0, 1]:
    raise ValueError(f"LY['lX'] must be 0 or 1, but got {LY['lX']}")
  # Convert to bool instead of dim 0 array of ints
  diverted = bool(LY['lX'] == 1)
  R_major = LY['rgeom'][-1]  # Major radius  # pyrefly: ignore[bad-index]
  B_0 = LY['rBt'] / R_major  # Vacuum toroidal magnetic field at R_major
  a_minor = LY['aminor'][-1]  # Minor radius  # pyrefly: ignore[bad-index]
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

  rhon = np.sqrt(Phi / Phi[-1])  # Normalized toroidal flux coordinate  # pyrefly: ignore[bad-index]
  psi = L['pQ'] ** 2 * (LY['FB'] - LY['FA']) + LY['FA']  # Poloidal flux
  # To avoid possible divisions by zero in diverted geometry. Value of what
  # replaces the zero does not matter, since it will be replaced by a spline
  # extrapolation in the post_init.
  LY_Q1Q = np.where(LY['Q1Q'] != 0, LY['Q1Q'], constants.CONSTANTS.eps)

  # TODO(b/426291465): Implement a more accurate calculation of <1/B^2>
  # (either here or upstream in MEQ)
  # Approximate with analytical expressions for circular geometry.
  flux_surf_avg_B2 = math_utils.safe_divide(
      num=B_0**2, denom=np.sqrt(1.0 - LY['epsilon'] ** 2), eps=1e-7
  )
  flux_surf_avg_1_over_B2 = B_0**-2 * (1.0 + 1.5 * LY['epsilon'] ** 2)
  match trapped_fraction_source:
    case trapped_fraction_lib.TrappedFractionSource.SAUTER:
      delta = (LY['deltau'] + LY['deltal']) / 2.0
      trapped_fraction = trapped_fraction_lib.calculate_sauter_trapped_fraction(
          epsilon=np.asarray(LY['epsilon']), delta=np.asarray(delta)
      )
    case _:
      raise ValueError(
          f'Unsupported trapped_fraction_source: {trapped_fraction_source}.'
          ' Supported options: '
          f'{trapped_fraction_lib.TrappedFractionSource.SAUTER.value}.'
      )

  # Edge/Divertor geometry
  # These parameters are optional as older MEQ files may not contain them.
  connection_length_target = None
  connection_length_divertor = None
  angle_of_incidence_target = None
  R_OMP = None
  R_target = None
  B_pol_OMP = None

  # All edge model keys must be present to extract edge quantities.
  all_edge_keys = _ALL_EDGE_KEYS

  if all_edge_keys.issubset(LY.keys()):
    Lpar_div = np.asarray(LY['Lpar_div'])
    # OMP quantities are scalars so no indexing is needed.
    R_OMP = LY['r_OMP']
    B_pol_OMP = LY['Bp_OMP']

    if Lpar_div.ndim == 0:
      # Scalar values: all edge quantities are scalars, no indexing needed.
      connection_length_target = LY['Lpar_target']
      connection_length_divertor = Lpar_div
      angle_of_incidence_target = np.rad2deg(LY['alpha_target'])
      R_target = LY['r_target']
    else:
      # Multiple directions traced. Select the index with the shortest
      # connection length to the divertor (Lpar_div).
      idx = int(np.argmin(Lpar_div))
      connection_length_target = LY['Lpar_target'][idx]  # pyrefly: ignore[bad-index]
      connection_length_divertor = Lpar_div[idx]
      angle_of_incidence_target = np.rad2deg(LY['alpha_target'][idx])  # pyrefly: ignore[bad-index]
      R_target = LY['r_target'][idx]  # pyrefly: ignore[bad-index]

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.MEQ,
      R_major=R_major,
      a_minor=a_minor,
      B_0=B_0,
      psi=psi,
      Phi=Phi,  # pyrefly: ignore[bad-argument-type]
      I_p_profile=LY['ItQ'],  # pyrefly: ignore[bad-argument-type]
      R_in=LY['rgeom'] - LY['aminor'],  # pyrefly: ignore[bad-argument-type]
      R_out=LY['rgeom'] + LY['aminor'],  # pyrefly: ignore[bad-argument-type]
      F=LY['TQ'],  # pyrefly: ignore[bad-argument-type]
      int_dl_over_Bp=1 / LY_Q1Q,
      flux_surf_avg_1_over_R=LY['Q0Q'],  # pyrefly: ignore[bad-argument-type]
      flux_surf_avg_1_over_R2=LY['Q2Q'],  # pyrefly: ignore[bad-argument-type]
      flux_surf_avg_grad_psi2_over_R2=LY['Q3Q'],  # pyrefly: ignore[bad-argument-type]
      flux_surf_avg_grad_psi=2 * np.pi * LY['Q5Q'],  # pyrefly: ignore[bad-argument-type]
      flux_surf_avg_grad_psi2=LY['Q4Q'],  # pyrefly: ignore[bad-argument-type]
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      trapped_fraction=trapped_fraction,
      delta_upper_face=LY['deltau'],  # pyrefly: ignore[bad-argument-type]
      delta_lower_face=LY['deltal'],  # pyrefly: ignore[bad-argument-type]
      elongation=LY['kappa'],  # pyrefly: ignore[bad-argument-type]
      vpr=4 * np.pi * Phi[-1] * rhon / (LY['TQ'] * LY['Q2Q']),  # pyrefly: ignore[bad-index]
      face_centers=face_centers,
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


def _validate_meq_data(
    LY: Mapping[str, np.ndarray], L: Mapping[str, np.ndarray]
) -> None:
  """Validates the MEQ data dictionaries.

  Works for both single slice and bundle LY data.

  Args:
    LY: A dictionary of MEQ LY geometry data.
    L: A dictionary of MEQ L geometry data.

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
  # Optional edge model inputs
  edge_model_omp_keys = _EDGE_MODEL_OMP_KEYS
  edge_model_target_keys = _EDGE_MODEL_TARGET_KEYS

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

  # Validate presence of all optional edge model inputs
  all_optional_keys = edge_model_omp_keys | edge_model_target_keys
  present_optional_keys = all_optional_keys.intersection(LY.keys())
  if present_optional_keys and len(present_optional_keys) != len(
      all_optional_keys
  ):
    missing_optional_keys = all_optional_keys - present_optional_keys
    logging.warning(
        'Some but not all optional edge model keys are present in LY data. '
        'This may lead to incomplete edge model initialization. '
        'Missing data: %s',
        missing_optional_keys,
    )

  # 1. OMP keys must match time_only_shape
  for key in edge_model_omp_keys:
    if key in LY:
      if LY[key].shape != time_only_shape:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. "
            f'Expected {time_only_shape}, got {LY[key].shape}.'
        )

  # 2. Target keys can be time_only_shape (n_directions=1)
  # or (2,) + time_only_shape (n_directions=2)
  shape_n_directions2 = (2,) + time_only_shape
  for key in edge_model_target_keys:
    if key in LY:
      shape = LY[key].shape
      if shape != time_only_shape and shape != shape_n_directions2:
        raise ValueError(
            f"Incorrect shape for key '{key}' in LY data. "
            f'Expected {time_only_shape} or {shape_n_directions2}, got {shape}.'
        )
