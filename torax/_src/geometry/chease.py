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
"""Functions for loading and representing a CHEASE geometry."""

from collections.abc import Mapping
from typing import Annotated, Any, Final, Literal, Self
import numpy as np
import pydantic
from torax._src import constants
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry
from torax._src.geometry import trapped_fraction as trapped_fraction_lib
from torax._src.torax_pydantic import torax_pydantic

_MAT2COLS_KEY_MAP: Final[dict[str, str]] = {
    'PSIchease=psi/2pi': 'psi_chease',
    'Ipprofile': 'ip_profile',
    'RHO_TOR=sqrt(Phi/pi/B0)': 'rho_tor',
    'R_INBOARD': 'r_inboard',
    'R_OUTBOARD': 'r_outboard',
    'T=RBphi': 't_rbphi',
    'Int(Rdlp/|grad(psi)|)=Int(Jdchi)': 'int_dl_over_bp',
    '<1/R>profile': 'inv_r',
    '<1/R**2>': 'inv_r2',
    '<Bp**2>': 'bp2',
    '<|grad(psi)|>': 'grad_psi',
    '<|grad(psi)|**2>': 'grad_psi2',
    '<B**2>': 'b2',
    '<1/B**2>': 'inv_b2',
    'FTRAP': 'ftrap',
    'delta_upper': 'delta_upper',
    'delta_bottom': 'delta_bottom',
    'elongation': 'elongation',
}

_CHEASE_ZERO_ON_AXIS = frozenset({
    'Ipprofile',
    'RHO_TOR=sqrt(Phi/pi/B0)',
    '<|grad(psi)|>',
    '<|grad(psi)|**2>',
    '<Bp**2>',
    'delta_upper',
    'delta_bottom',
    'FTRAP',
})


def _maybe_prepend_magnetic_axis(
    chease_data: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
  """Prepends the magnetic axis (rho_norm=0) to off-axis CHEASE profiles."""
  out = dict(chease_data)
  if abs(float(np.asarray(out['RHO_TOR=sqrt(Phi/pi/B0)'])[0])) == 0.0:
    return out
  r_axis = 0.5 * (
      np.asarray(out['R_INBOARD'])[0] + np.asarray(out['R_OUTBOARD'])[0]
  )
  axis_values = dict.fromkeys(_CHEASE_ZERO_ON_AXIS, 0.0)
  axis_values['R_INBOARD'] = r_axis
  axis_values['R_OUTBOARD'] = r_axis
  for name, profile in out.items():
    if np.ndim(profile) != 1:
      continue
    axis_value = axis_values.get(name, np.asarray(profile)[0])
    out[name] = np.insert(profile, 0, axis_value)
  return out


# pylint: disable=invalid-name
class CheaseData(torax_pydantic.ConfigBase):
  """Pydantic model containing CHEASE 1D profile columns and scalar overrides.

  Attributes:
    psi_chease: Normalized poloidal flux psi / 2pi.
    ip_profile: Toroidal plasma current profile.
    rho_tor: Toroidal flux coordinate sqrt(Phi / pi / B0).
    r_inboard: Inboard major radius profile.
    r_outboard: Outboard major radius profile.
    t_rbphi: Toroidal field flux function T = R * B_phi.
    int_dl_over_bp: Poloidal loop integral Int(R dl_p / |grad(psi)|).
    inv_r: Flux-surface average <1/R>.
    inv_r2: Flux-surface average <1/R^2>.
    bp2: Flux-surface average <Bp^2>.
    grad_psi: Flux-surface average <|grad(psi)|>.
    grad_psi2: Flux-surface average <|grad(psi)|^2>.
    b2: Flux-surface average <B^2>.
    inv_b2: Flux-surface average <1/B^2>.
    delta_upper: Upper triangularity profile.
    delta_bottom: Lower triangularity profile.
    elongation: Elongation profile.
    ftrap: Effective trapped particle fraction from file (optional).
  """

  psi_chease: torax_pydantic.NumpyArray1D
  ip_profile: torax_pydantic.NumpyArray1D
  rho_tor: torax_pydantic.NumpyArray1D
  r_inboard: torax_pydantic.NumpyArray1D
  r_outboard: torax_pydantic.NumpyArray1D
  t_rbphi: torax_pydantic.NumpyArray1D
  int_dl_over_bp: torax_pydantic.NumpyArray1D
  inv_r: torax_pydantic.NumpyArray1D
  inv_r2: torax_pydantic.NumpyArray1D
  bp2: torax_pydantic.NumpyArray1D
  grad_psi: torax_pydantic.NumpyArray1D
  grad_psi2: torax_pydantic.NumpyArray1D
  b2: torax_pydantic.NumpyArray1D
  inv_b2: torax_pydantic.NumpyArray1D
  delta_upper: torax_pydantic.NumpyArray1D
  delta_bottom: torax_pydantic.NumpyArray1D
  elongation: torax_pydantic.NumpyArray1D
  ftrap: torax_pydantic.NumpyArray1D | None = None

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_chease_data(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    data_dict = dict(data)
    if 'RHO_TOR=sqrt(Phi/pi/B0)' in data_dict:
      data_dict = _maybe_prepend_magnetic_axis(data_dict)
    renamed: dict[str, Any] = {}
    for k, v in data_dict.items():
      target_key = _MAT2COLS_KEY_MAP.get(k, k)
      if target_key in cls.model_fields:
        renamed[target_key] = v
      elif k in cls.model_fields:
        renamed[k] = v
    return renamed

  @pydantic.model_validator(mode='after')
  def _validate_arrays(self) -> Self:
    n_pts = len(self.psi_chease)
    for field_name in (
        'ip_profile',
        'rho_tor',
        'r_inboard',
        'r_outboard',
        't_rbphi',
        'int_dl_over_bp',
        'inv_r',
        'inv_r2',
        'bp2',
        'grad_psi',
        'grad_psi2',
        'b2',
        'inv_b2',
        'delta_upper',
        'delta_bottom',
        'elongation',
    ):
      arr = getattr(self, field_name)
      if len(arr) != n_pts:
        raise ValueError(
            f"Array '{field_name}' length ({len(arr)}) does not match"
            f" 'psi_chease' length ({n_pts})."
        )
    if self.ftrap is not None and len(self.ftrap) != n_pts:
      raise ValueError(
          f"Array 'ftrap' length ({len(self.ftrap)}) does not match"
          f" 'psi_chease' length ({n_pts})."
      )
    return self


class CheaseConfig(base.GeometryConfigBase):
  """Loads a normalized magnetic equilibrium from a CHEASE geometry file.

  Attributes:
    model_name: Always set to `'chease'`.
    chease_data: CHEASE equilibrium dataset or dictionary of time to dataset.
    R_major: Major radius used for denormalization, defined as the midplane
      average of the LCFS radius, $R_major = (R_max + R_min) / 2$ [m].
    a_minor: Minor radius used for denormalization, defined as half the
      horizontal width of the LCFS, $a_minor = (R_max - R_min) / 2$ [m]. Must
      satisfy `a_minor <= R_major`.
    B_0: Vacuum toroidal magnetic field at `R_major` used for denormalization
      [T].

  Note:
    The constructor also accepts inputs normalized into chease_data:
      geometry_file: File name of the CHEASE geometry file.
      geometry_directory: Directory containing the geometry file.
  """

  @property
  def _supported_trapped_fraction_sources(
      self,
  ) -> frozenset[trapped_fraction_lib.TrappedFractionSource]:
    return frozenset({
        trapped_fraction_lib.TrappedFractionSource.SAUTER,
        trapped_fraction_lib.TrappedFractionSource.FILE,
    })

  model_name: Annotated[
      Literal['chease'],
      torax_pydantic.TIME_INVARIANT,
      torax_pydantic.JAX_STATIC,
  ] = 'chease'
  chease_data: CheaseData | dict[torax_pydantic.Second, CheaseData]
  R_major: torax_pydantic.Meter = 6.2
  a_minor: torax_pydantic.Meter = 2.0
  B_0: torax_pydantic.Tesla = 5.3

  def __init__(
      self,
      chease_data: (
          CheaseData | dict[torax_pydantic.Second, CheaseData] | None
      ) = None,
      geometry_file: str | Mapping[float, str] | None = None,
      geometry_directory: str | None = None,
      **kwargs: Any,
  ):
    data: dict[str, Any] = {}
    if chease_data is not None:
      data['chease_data'] = chease_data
    if geometry_file is not None:
      data['geometry_file'] = geometry_file
    if geometry_directory is not None:
      data['geometry_directory'] = geometry_directory
    data.update(kwargs)
    super().__init__(**data)

  @pydantic.model_validator(mode='before')
  @classmethod
  def _before_validator(cls, data: Any) -> Any:
    if not isinstance(data, Mapping):
      return data
    data_dict = dict(data)
    if 'chease_data' not in data_dict:
      g_dir = data_dict.pop('geometry_directory', None)
      if not g_dir:
        g_dir = None
      g_file = data_dict.pop('geometry_file', None)
      if g_file is None:
        raise ValueError(
            "Either 'geometry_file' or 'chease_data' must be provided."
        )
      if isinstance(g_file, Mapping):
        data_dict['chease_data'] = {
            float(t): geometry_loader.load_geo_data(
                g_dir, single_file, geometry_loader.GeometrySource.CHEASE
            )
            for t, single_file in g_file.items()
        }
      else:
        data_dict['chease_data'] = geometry_loader.load_geo_data(
            g_dir, g_file, geometry_loader.GeometrySource.CHEASE
        )
    else:
      data_dict.pop('geometry_directory', None)
      data_dict.pop('geometry_file', None)
    return data_dict

  @pydantic.model_validator(mode='after')
  def _check_fields(self) -> Self:
    if not self.R_major >= self.a_minor:
      raise ValueError('a_minor must be less than or equal to R_major.')
    return self

  def _build_slice_geometry(
      self, data: CheaseData
  ) -> standard_geometry.StandardGeometry:
    intermediates = _construct_intermediates_from_chease(
        face_centers=self.get_face_centers(),
        R_major=float(self.R_major),
        a_minor=float(self.a_minor),
        B_0=float(self.B_0),
        hires_factor=self.hires_factor,
        trapped_fraction_source=self.trapped_fraction_source,
        chease_data=data,
    )
    return standard_geometry.build_standard_geometry(intermediates)

  def _build_geometry(
      self,
  ) -> (
      standard_geometry.StandardGeometry
      | Mapping[float, standard_geometry.StandardGeometry]
  ):
    if isinstance(self.chease_data, CheaseData):
      return self._build_slice_geometry(self.chease_data)
    return {
        float(t): self._build_slice_geometry(data)
        for t, data in self.chease_data.items()
    }


def _construct_intermediates_from_chease(
    *,
    chease_data: CheaseData | Mapping[str, Any],
    face_centers: np.ndarray,
    R_major: float,
    a_minor: float,
    B_0: float,
    hires_factor: int,
    trapped_fraction_source: (
        trapped_fraction_lib.TrappedFractionSource
    ) = trapped_fraction_lib.TrappedFractionSource.SAUTER,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from CHEASE data.

  Args:
    chease_data: CheaseData instance or mapping of CHEASE columns.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    R_major: Major radius (R) in meters, commonly defined in TORAX as the
      midplane average of the LCFS radius. CHEASE geometries are normalized, so
      this is used as an unnormalization factor.
    a_minor: Minor radius (a) in meters, commonly defined in TORAX as half the
      horizontal width of the LCFS.
    B_0: Vacuum toroidal magnetic field at `R_major` [T].
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.
    trapped_fraction_source: Selects how the effective trapped particle fraction
      is computed; see `trapped_fraction.TrappedFractionSource`.

  Returns:
    A StandardGeometryIntermediates instance based on the input data.
  """
  if not isinstance(chease_data, CheaseData):
    chease_data = CheaseData.model_validate(chease_data)
  # Prepare variables from CHEASE to be interpolated into our simulation
  # grid. CHEASE variables are normalized. Need to unnormalize them with
  # reference values poloidal flux and CHEASE-internal-calculated plasma
  # current.
  # Also, CHEASE is COCOS=02, which means psi_torax = 2π * psi_chease
  psiunnormfactor = R_major**2 * B_0 * 2 * np.pi
  psi = chease_data.psi_chease * psiunnormfactor
  I_p_chease = chease_data.ip_profile / constants.CONSTANTS.mu_0 * R_major * B_0

  # toroidal flux
  Phi = (chease_data.rho_tor * R_major) ** 2 * B_0 * np.pi

  # midplane radii
  R_in_chease = chease_data.r_inboard * R_major
  R_out_chease = chease_data.r_outboard * R_major
  # toroidal field flux function
  F = chease_data.t_rbphi * R_major * B_0

  int_dl_over_Bp = chease_data.int_dl_over_bp * R_major / B_0
  flux_surf_avg_1_over_R = chease_data.inv_r / R_major
  flux_surf_avg_1_over_R2 = chease_data.inv_r2 / R_major**2
  # COCOS > 10: <|\nabla \psi|> = 2\pi<R B_p>
  flux_surf_avg_grad_psi2_over_R2 = chease_data.bp2 * B_0**2 * (4 * np.pi**2)
  flux_surf_avg_grad_psi = chease_data.grad_psi * psiunnormfactor / R_major
  flux_surf_avg_grad_psi2 = (
      chease_data.grad_psi2 * psiunnormfactor**2 / R_major**2
  )
  flux_surf_avg_B2 = chease_data.b2 * B_0**2
  flux_surf_avg_1_over_B2 = chease_data.inv_b2 / B_0**2
  match trapped_fraction_source:
    case trapped_fraction_lib.TrappedFractionSource.FILE:
      if chease_data.ftrap is None:
        raise ValueError(
            'trapped_fraction_source is FILE but ftrap is None in CheaseData.'
        )
      trapped_fraction = chease_data.ftrap
    case trapped_fraction_lib.TrappedFractionSource.SAUTER:
      epsilon = (R_out_chease - R_in_chease) / (R_out_chease + R_in_chease)
      delta = (chease_data.delta_upper + chease_data.delta_bottom) / 2.0
      trapped_fraction = trapped_fraction_lib.calculate_sauter_trapped_fraction(
          epsilon=epsilon, delta=delta
      )
    case _:
      raise ValueError(
          f'Unsupported trapped_fraction_source: {trapped_fraction_source}.'
          'Supported options: '
          f'{trapped_fraction_lib.TrappedFractionSource.SAUTER.value}, '
          f'{trapped_fraction_lib.TrappedFractionSource.FILE.value}.'
      )

  rhon = np.sqrt(Phi / Phi[-1])
  vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.CHEASE,
      R_major=np.array(R_major),
      a_minor=np.array(a_minor),
      B_0=np.array(B_0),
      psi=psi,
      I_p_profile=I_p_chease,
      Phi=Phi,
      R_in=R_in_chease,
      R_out=R_out_chease,
      F=F,
      int_dl_over_Bp=int_dl_over_Bp,
      flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
      flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
      flux_surf_avg_grad_psi2_over_R2=flux_surf_avg_grad_psi2_over_R2,
      flux_surf_avg_grad_psi=flux_surf_avg_grad_psi,
      flux_surf_avg_grad_psi2=flux_surf_avg_grad_psi2,
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      trapped_fraction=trapped_fraction,
      delta_upper_face=chease_data.delta_upper,
      delta_lower_face=chease_data.delta_bottom,
      elongation=chease_data.elongation,
      vpr=vpr,
      face_centers=face_centers,
      hires_factor=hires_factor,
      diverted=None,
      connection_length_target=None,
      connection_length_divertor=None,
      angle_of_incidence_target=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
      z_magnetic_axis=None,
  )
