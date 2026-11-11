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
"""Base class for geometry configuration."""

import abc
from collections.abc import Mapping
import dataclasses
import functools
from typing import Annotated, Any, Self

import jax
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import jax_utils
from torax._src.geometry import geometry
from torax._src.geometry import geometry_model
from torax._src.geometry import runtime_params as geometry_runtime_params
from torax._src.geometry import trapped_fraction
from torax._src.torax_pydantic import torax_pydantic

TrappedFractionSource = trapped_fraction.TrappedFractionSource


# pylint: disable=invalid-name


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class InterpolatedGeometry:
  """Time-interpolated `Geometry` over a sorted 1D time grid.

  Stacks `Geometry` PyTrees across time and interpolates leaves via
  `jax.tree.map`. A dedicated container is used rather than `TimeVaryingArray`
  because `Geometry` is a heterogeneous PyTree (~50 fields) containing
  non-floating leaves (e.g. `diverted: bool`) and static `geometry_type`
  metadata that cannot be represented as a single numeric `TimeVaryingArray`.

  `InterpolatedGeometry` is registered as a JAX dataclass PyTree rather than a
  Pydantic `ConfigBase` model because it holds precomputed stacked JAX array
  leaves passed dynamically through `jax.jit` rather than user-facing
  serialized configuration fields.
  """

  times: array_typing.FloatVector
  stacked_geometry: geometry.Geometry

  def __post_init__(self):
    def _to_jnp(leaf: Any) -> Any:
      if isinstance(leaf, np.ndarray | jax.Array):
        if np.issubdtype(leaf.dtype, np.bool_):
          return jnp.asarray(leaf, dtype=jnp.bool_)
        return jnp.asarray(leaf, dtype=jax_utils.get_dtype())
      return leaf

    if isinstance(self.stacked_geometry, geometry.Geometry):
      object.__setattr__(
          self,
          'stacked_geometry',
          jax.tree.map(_to_jnp, self.stacked_geometry),
      )
    if isinstance(self.times, np.ndarray | jax.Array | list | tuple):
      object.__setattr__(
          self,
          'times',
          jnp.asarray(self.times, dtype=jax_utils.get_dtype()),
      )

  def get_value(self, t: array_typing.FloatScalar) -> geometry.Geometry:
    """Linearly interpolates `stacked_geometry` at time `t` (clamped at bounds)."""
    if self.times.shape[0] == 1:
      return jax.tree.map(lambda arr: arr[0], self.stacked_geometry)
    t_clamped = jnp.clip(t, self.times[0], self.times[-1])
    idx = jnp.searchsorted(self.times, t_clamped, side='right') - 1
    i0 = jnp.clip(idx, 0, self.times.shape[0] - 2)
    i1 = i0 + 1
    t0 = self.times[i0]
    t1 = self.times[i1]
    denom = jnp.where(t1 > t0, t1 - t0, 1.0)
    weight = jnp.where(t1 > t0, (t_clamped - t0) / denom, 0.0)

    def _interp_leaf(arr: Any) -> Any:
      if not isinstance(arr, jax.Array | np.ndarray):
        return arr
      if jnp.issubdtype(arr.dtype, jnp.bool_):
        val = (1.0 - weight) * arr[i0].astype(weight.dtype) + weight * arr[
            i1
        ].astype(weight.dtype)
        return jnp.bool_(val > 0.5)
      return ((1.0 - weight) * arr[i0] + weight * arr[i1]).astype(
          jax_utils.get_dtype()
      )

    return jax.tree.map(_interp_leaf, self.stacked_geometry)

  def get_Phi_b_dot(
      self, t: array_typing.FloatScalar
  ) -> array_typing.FloatScalar:
    """Returns the piecewise-linear time derivative of `Phi_b` at time `t`."""
    if self.times.shape[0] <= 1:
      return jnp.zeros((), dtype=jax_utils.get_dtype())
    in_bounds = (t >= self.times[0]) & (t <= self.times[-1])
    t_clamped = jnp.clip(t, self.times[0], self.times[-1])
    idx = jnp.searchsorted(self.times, t_clamped, side='right') - 1
    i0 = jnp.clip(idx, 0, self.times.shape[0] - 2)
    i1 = i0 + 1
    dt_seg = self.times[i1] - self.times[i0]
    phi_b_arr = jnp.asarray(self.stacked_geometry.Phi_b)
    dphi = phi_b_arr[i1] - phi_b_arr[i0]
    slope = jnp.where(dt_seg > 0.0, dphi / dt_seg, 0.0)
    return jnp.where(in_bounds, slope, 0.0).astype(jax_utils.get_dtype())


def _build_interpolated_geometry(
    geometries: Mapping[float, geometry.Geometry],
) -> InterpolatedGeometry:
  """Builds an `InterpolatedGeometry` from a time mapping of `Geometry` slices."""
  sorted_times = sorted(geometries.keys())
  geos = [geometries[t] for t in sorted_times]
  return InterpolatedGeometry(
      times=jnp.asarray(sorted_times, dtype=jax_utils.get_dtype()),
      stacked_geometry=geometry.stack_geometries(geos),
  )


class GeometryConfigBase(torax_pydantic.ConfigBase):
  """Base class for all geometry configuration classes.

  Attributes:
    model_name: Geometry model identifier.
    n_rho: Number of radial grid cells. Creates a uniform grid with cell width
      $drho_norm = 1 / n_rho$. Must be at least 4. Defaults to `25` when neither
      `n_rho` nor `face_centers` is specified. Either `n_rho` or `face_centers`
      must be specified.
    face_centers: Explicit array of face center coordinates [$rho_norm$]
      (ranging from 0 to 1) for defining non-uniform radial grids. For a grid
      with $N$ cells, there must be $N + 1$ face centers. The array must start
      at `0.0`, end at `1.0`, be strictly increasing, and have at least 5
      elements (4 cells). This enables finer resolution in regions of interest,
      such as near the plasma edge::  import numpy as np  # Coarse in core (8
      cells in [0, 0.8]), fine near edge (8 cells in [0.8, 1]). core_faces =
      np.linspace(0, 0.8, 9) edge_faces = np.linspace(0.8, 1.0, 9)[1:]
      face_centers = np.concatenate([core_faces, edge_faces]) geometry_config =
      { 'model_name': 'chease', 'geometry_file': 'iterhybrid.mat2cols',
      'face_centers': face_centers, }
    hires_factor: Only used when the initial condition `psi` is from plasma
      current. Sets up a higher resolution mesh with `nrho_hires = n_rho *
      hires_factor`, used for `j` to `psi` conversions.
    trapped_fraction_source: Selects how the effective trapped particle
      fraction, used by the neoclassical `bootstrap_current`, `conductivity`,
      and `transport` models, is computed. Computed once at geometry
      construction time. Options are:  - `'sauter'` (default): Uses the
      analytical approximation from |sauter16|. Supported by every geometry
      source. - `'file'`: Reads the value precomputed by the input equilibrium
      code directly from the geometry file. Only supported for CHEASE and IMAS
      geometries. - `'exact'`: Computes the exact bounce-averaged integral
      directly from the traced 2D equilibrium, using the formulation of
      |lin-liu-miller1995|. Only supported for EQDSK and IMAS geometries.
    calc_Phi_b_dot: Toggles whether to calculate time-dependent toroidal flux
      derivatives ($Phi_b_dot$) in `GeometryState` (across steps via interval
      secant and at `t_initial` from time-dependent prescribed geometry).

  Subclasses must override:
    - `_supported_trapped_fraction_sources`: Property returning the `frozenset`
      of supported `trapped_fraction.TrappedFractionSource` options.
    - `_build_geometry`: Method returning either a single `geometry.Geometry`
      (for time-invariant configs) or a `{time: geometry.Geometry}` mapping (for
      multi-slice time-dependent configs) constructed on the host.

  Future self-consistent (state-dependent) equilibrium configs may also override
  `build_runtime_params` and `build_geometry_model` (which default to
  `PrescribedGeometryRuntimeParams` and `PrescribedGeometryModel`).

  Equilibrium construction (file I/O, contour tracing, spline fitting, and
  flux-surface integration) runs on the host and is cached in
  `_precomputed_geometry`. This cached property is excluded from Pydantic
  `model_dump` serialization, passed as a dynamic child PyTree in
  `tree_flatten` / `tree_unflatten` across JAX JIT boundaries, and automatically
  invalidated via `ConfigBase.clear_cached_properties()` whenever fields are
  updated via `update_fields()`.
  """

  model_name: Annotated[
      str,
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = ''
  n_rho: Annotated[
      int | None,
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = None
  face_centers: Annotated[
      torax_pydantic.NumpyArray1DSorted | None, torax_pydantic.TIME_INVARIANT
  ] = None
  hires_factor: Annotated[
      pydantic.PositiveInt,
      torax_pydantic.JAX_STATIC,
  ] = 4
  trapped_fraction_source: Annotated[
      trapped_fraction.TrappedFractionSource,
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = trapped_fraction.TrappedFractionSource.SAUTER
  calc_Phi_b_dot: Annotated[bool, torax_pydantic.JAX_STATIC] = True

  @property
  @abc.abstractmethod
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction.TrappedFractionSource]:
    """Supported trapped fraction sources for this geometry."""

  @abc.abstractmethod
  def _build_geometry(
      self,
  ) -> geometry.Geometry | Mapping[float, geometry.Geometry]:
    """Builds a single `Geometry` or a `{time: Geometry}` mapping on the host."""

  @pydantic.model_validator(mode='before')
  @classmethod
  def _validate_inputs(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data
    # Set default n_rho if not provided.
    if 'n_rho' not in data and 'face_centers' not in data:
      data['n_rho'] = 25
    return data

  @pydantic.model_validator(mode='after')
  def _validate_n_rho_or_face_centers(self) -> Self:
    """Validates that there are at least 4 cells."""
    if self.n_rho is None and self.face_centers is None:
      raise ValueError('Either n_rho or face_centers must be set.')

    if self.face_centers is not None:
      if len(self.face_centers) < 5:
        raise ValueError(
            'face_centers must have at least 5 elements (4 cells). Got'
            f' {len(self.face_centers)}'
        )
      if not np.isclose(self.face_centers[0], 0.0) or not np.isclose(
          self.face_centers[-1], 1.0
      ):
        raise ValueError(
            'face_centers must start at 0.0 and end at 1.0. Got'
            f' {self.face_centers}'
        )

    if self.n_rho is not None and self.n_rho < 4:
      raise ValueError('n_rho must be at least 4')

    return self

  @pydantic.model_validator(mode='after')
  def _validate_trapped_fraction_source(self) -> Self:
    """Validates that trapped_fraction_source is supported by this geometry."""
    if (
        self.trapped_fraction_source
        not in self._supported_trapped_fraction_sources
    ):
      allowed = ', '.join(
          sorted(s.value for s in self._supported_trapped_fraction_sources)
      )
      raise ValueError(
          f'trapped_fraction_source={self.trapped_fraction_source.value} is'
          f' not supported for {type(self).__name__}. Supported options:'
          f' {allowed}.'
      )
    return self

  @functools.cached_property
  def _precomputed_geometry(self) -> geometry.Geometry | InterpolatedGeometry:
    """Precomputes the single-slice or time-interpolated `Geometry` on the host."""
    geometries = self._build_geometry()
    if not isinstance(geometries, Mapping):
      return geometries
    if not geometries:
      raise ValueError('No geometries provided.')
    if len(geometries) == 1:
      return next(iter(geometries.values()))
    return _build_interpolated_geometry(geometries)

  def precompute(self) -> None:
    """Precomputes host-side equilibrium geometry outside JAX tracing."""
    _ = self._precomputed_geometry

  def tree_flatten(self):
    dynamic_children, static_children = super().tree_flatten()
    return (*dynamic_children, self._precomputed_geometry), static_children

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    *dynamic_children, precomputed_geometry = children
    obj = super().tree_unflatten(aux_data, dynamic_children)
    obj.__dict__['_precomputed_geometry'] = precomputed_geometry
    return obj

  def get_face_centers(self) -> np.ndarray:
    """Returns face_centers, computing from n_rho if needed."""
    if self.face_centers is not None:
      return self.face_centers
    assert self.n_rho is not None
    return torax_pydantic.Grid1D.uniform_from_n_rho(self.n_rho).face_centers

  def get_grid(self) -> torax_pydantic.Grid1D:
    """Returns the radial `Grid1D` for this geometry configuration."""
    return torax_pydantic.Grid1D(face_centers=self.get_face_centers())

  def build_runtime_params(
      self, t: array_typing.FloatScalar
  ) -> geometry_runtime_params.PrescribedGeometryRuntimeParams:
    """Builds prescribed geometry runtime parameters evaluated at time `t`."""
    precomputed = self._precomputed_geometry
    if isinstance(precomputed, InterpolatedGeometry):
      geo_t = precomputed.get_value(t)
      initial_phi_b_dot = (
          precomputed.get_Phi_b_dot(t)
          if self.calc_Phi_b_dot
          else jnp.zeros((), dtype=jax_utils.get_dtype())
      )
    else:
      geo_t = precomputed
      initial_phi_b_dot = jnp.zeros((), dtype=jax_utils.get_dtype())
    return geometry_runtime_params.PrescribedGeometryRuntimeParams(
        geometry=geo_t,
        calc_Phi_b_dot=self.calc_Phi_b_dot,
        initial_Phi_b_dot=initial_phi_b_dot,
    )

  def build_geometry_model(self) -> geometry_model.PrescribedGeometryModel:
    """Builds the static `GeometryModel` for this configuration."""
    return geometry_model.PrescribedGeometryModel()

  @property
  def geometry_type(self) -> geometry.GeometryType:
    """Returns the GeometryType enum corresponding to this configuration."""
    return geometry.GeometryType[self.model_name.upper()]

  def __eq__(self, other: object) -> bool:
    """Equality operator for GeometryConfigBase."""
    if not isinstance(other, GeometryConfigBase) or type(other) is not type(
        self
    ):
      return False
    return (
        np.array_equal(self.get_face_centers(), other.get_face_centers())
        and self.hires_factor == other.hires_factor
    )
