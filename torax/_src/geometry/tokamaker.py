# Copyright 2026 DeepMind Technologies Limited
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
"""Functions for loading and representing a TokaMaker geometry."""
from typing import Annotated, Any, Literal

import numpy as np
import pydantic
import scipy
from torax._src import constants
from torax._src.geometry import base
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.torax_pydantic import torax_pydantic
import typing_extensions

# pylint: disable=invalid-name

# Maps `TokaMaker_equilibrium.get_fsa()` keys onto field names of
# `TokaMakerEquilibrium`.
_GET_FSA_KEY_MAP = {
    'psi': 'psi',
    'q': 'q',
    'F': 'F',
    '<1/R>': 'flux_surf_avg_1_over_R',
    '<1/R^2>': 'flux_surf_avg_1_over_R2',
    'dV/dPsi': 'dV_dpsi',
    '<|grad psi|>': 'flux_surf_avg_grad_psi',
    '<|grad psi|^2>': 'flux_surf_avg_grad_psi2',
    '<Bp^2>': 'flux_surf_avg_Bp2',
    '<1/B^2>': 'flux_surf_avg_1_over_B2',
    'R_min': 'R_min',
    'R_max': 'R_max',
    'Z_min': 'Z_min',
    'Z_max': 'Z_max',
    'R_at_Zmin': 'R_at_Zmin',
    'R_at_Zmax': 'R_at_Zmax',
    'psi_axis': 'psi_axis',
    'R_axis': 'R_axis',
    'Z_axis': 'Z_axis',
    'F_axis': 'F_axis',
    'F0': 'F0',
}


class TokaMakerEquilibrium(torax_pydantic.BaseModelFrozen):
  """Flux surface averaged profiles extracted from a TokaMaker equilibrium.

  Constructed from the dict of profiles returned by
  ``TokaMaker_equilibrium.get_fsa()``, whose keys are mapped onto the attributes
  of this class by `_GET_FSA_KEY_MAP`.

  Attributes:
    psi: Poloidal flux [Wb/rad]
    q: Safety factor [dimensionless]
    F: Toroidal field flux function F = R*B_phi [m T]
    flux_surf_avg_1_over_R: <1/R> [m^-1]
    flux_surf_avg_1_over_R2: <1/R^2> [m^-2]
    dV_dpsi: dV/dpsi [m^3 (Wb/rad)^-1]
    flux_surf_avg_grad_psi: <|grad psi|> [Wb rad^-1 m^-1]
    flux_surf_avg_grad_psi2: <|grad psi|^2> [Wb^2 rad^-2 m^-2]
    flux_surf_avg_Bp2: <Bp^2> [T^2]
    flux_surf_avg_1_over_B2: <1/B^2> [T^-2]
    R_min: Minimum major radius of each flux surface [m]
    R_max: Maximum major radius of each flux surface [m]
    Z_min: Minimum height of each flux surface [m]
    Z_max: Maximum height of each flux surface [m]
    R_at_Zmin: Major radius at ``Z_min`` [m]
    R_at_Zmax: Major radius at ``Z_max`` [m]
    psi_axis: Poloidal flux on the magnetic axis [Wb/rad]
    R_axis: Major radius of the magnetic axis [m]
    Z_axis: Height of the magnetic axis [m]
    F_axis: ``F`` on the magnetic axis [m T]
    F0: Vacuum toroidal field flux function [m T]
  """

  psi: torax_pydantic.NumpyArray1D
  q: torax_pydantic.NumpyArray1D
  F: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_R: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_R2: torax_pydantic.NumpyArray1D
  dV_dpsi: torax_pydantic.NumpyArray1D
  flux_surf_avg_grad_psi: torax_pydantic.NumpyArray1D
  flux_surf_avg_grad_psi2: torax_pydantic.NumpyArray1D
  flux_surf_avg_Bp2: torax_pydantic.NumpyArray1D
  flux_surf_avg_1_over_B2: torax_pydantic.NumpyArray1D
  R_min: torax_pydantic.NumpyArray1D
  R_max: torax_pydantic.NumpyArray1D
  Z_min: torax_pydantic.NumpyArray1D
  Z_max: torax_pydantic.NumpyArray1D
  R_at_Zmin: torax_pydantic.NumpyArray1D
  R_at_Zmax: torax_pydantic.NumpyArray1D
  psi_axis: float
  R_axis: torax_pydantic.Meter
  Z_axis: float
  F_axis: float
  F0: float

  @pydantic.model_validator(mode='before')
  @classmethod
  def _conform_get_fsa_dict(cls, data: Any) -> Any:
    """Renames `get_fsa()` keys."""
    if not isinstance(data, dict):
      return data
    return {
        _GET_FSA_KEY_MAP.get(key, key): value
        for key, value in data.items()
        if key in _GET_FSA_KEY_MAP or key in cls.model_fields
    }

  @pydantic.model_validator(mode='after')
  def _validate_profiles(self) -> typing_extensions.Self:
    lengths = {
        name: len(getattr(self, name))
        for name in type(self).model_fields
        if isinstance(getattr(self, name), np.ndarray)
    }
    if len(set(lengths.values())) != 1:
      raise ValueError(f'Profiles have inconsistent lengths: {lengths}')
    if len(self.psi) < 2:
      raise ValueError('At least 2 flux surfaces are required.')
    return self


class TokaMakerConfig(base.BaseGeometryConfig):
  """Pydantic model for the TokaMaker geometry.

  The flux surface grid is set by the ``get_fsa()`` call, not by this config.
  The outermost sampled surface becomes rho_norm = 1, so ``psi_pad=0.01`` plays
  the role of ``last_surface_factor=0.99`` in the EQDSK geometry. Sample
  uniformly in sqrt(psi_norm) rather than psi_norm, since rho_norm scales as
  sqrt(psi_norm) near the axis and a psi_norm-uniform grid is coarse there.

  Attributes:
    geometry_type: Always set to 'tokamaker'.
    Ip_from_parameters: Toggles whether total plasma current is read from the
      configuration file, or from the equilibrium. If True, then the `psi`
      calculated from the equilibrium is scaled to match the desired `I_p`.
    fsa_profiles: Flux surface profiles, as returned by
      ``TokaMaker_equilibrium.get_fsa()``.
  """

  geometry_type: Annotated[
      Literal['tokamaker'], torax_pydantic.TIME_INVARIANT
  ] = 'tokamaker'
  Ip_from_parameters: Annotated[bool, torax_pydantic.TIME_INVARIANT] = True
  fsa_profiles: TokaMakerEquilibrium

  def build_geometry(self) -> standard_geometry.StandardGeometry:
    intermediates = _construct_intermediates_from_tokamaker(
        equilibrium=self.fsa_profiles,
        Ip_from_parameters=self.Ip_from_parameters,
        face_centers=self.get_face_centers(),
        hires_factor=self.hires_factor,
    )
    return standard_geometry.build_standard_geometry(intermediates)


def _construct_intermediates_from_tokamaker(
    equilibrium: TokaMakerEquilibrium,
    Ip_from_parameters: bool,
    face_centers: np.ndarray,
    hires_factor: int,
) -> standard_geometry.StandardGeometryIntermediates:
  """Constructs a StandardGeometryIntermediates from TokaMaker profiles.

  Converts TokaMaker's conventions (psi in Wb/rad, so Bp = |grad psi| / R) to
  the COCOS 11 (psi in Wb, so Bp = |grad psi| / 2*pi*R),
  i.e. psi_11 = 2*pi*psi_TM, and prepends the magnetic axis, where no flux
  surface can be traced.

  Args:
    equilibrium: Flux surface profiles from ``TokaMaker_equilibrium.get_fsa()``.
    Ip_from_parameters: If True, the Ip is taken from the parameters and the
      values in the Geometry are rescaled to match the new Ip.
    face_centers: Array of face center coordinates in normalized rho (0 to 1).
    hires_factor: Grid refinement factor for poloidal flux <--> plasma current
      calculations.

  Returns:
    A StandardGeometryIntermediates instance, to be passed to
    `build_standard_geometry`.
  """
  eq = equilibrium
  # On axis a flux surface degenerates to a point, so the poloidal field and
  # its flux surface averages vanish and B is purely toroidal.
  Btor_axis = eq.F_axis / eq.R_axis
  prepend = lambda axis_value, profile: np.concatenate(
      [np.array([axis_value]), profile]
  )

  psi = 2 * np.pi * prepend(eq.psi_axis, eq.psi)
  F = prepend(eq.F_axis, eq.F)
  # dV/dpsi = 2*pi*oint(dl/Bp), and Bp is convention independent.
  int_dl_over_Bp = prepend(0.0, np.abs(eq.dV_dpsi) / (2 * np.pi))
  flux_surf_avg_1_over_R = prepend(1.0 / eq.R_axis, eq.flux_surf_avg_1_over_R)
  flux_surf_avg_1_over_R2 = prepend(
      1.0 / eq.R_axis**2, eq.flux_surf_avg_1_over_R2
  )
  flux_surf_avg_grad_psi = prepend(0.0, 2 * np.pi * eq.flux_surf_avg_grad_psi)
  flux_surf_avg_grad_psi2 = prepend(
      0.0, 4 * np.pi**2 * eq.flux_surf_avg_grad_psi2
  )
  flux_surf_avg_grad_psi2_over_R2 = prepend(
      0.0, 4 * np.pi**2 * eq.flux_surf_avg_Bp2
  )
  # F is constant on a flux surface, so <B^2> follows exactly from <Bp^2>.
  flux_surf_avg_B2 = prepend(
      Btor_axis**2, eq.flux_surf_avg_Bp2 + eq.F**2 * eq.flux_surf_avg_1_over_R2
  )
  flux_surf_avg_1_over_B2 = prepend(
      1.0 / Btor_axis**2, eq.flux_surf_avg_1_over_B2
  )

  # Enclosed toroidal current from Ampere's law on a flux surface:
  # mu_0 I = oint(B_p dl) = <B_p^2> oint(dl/B_p).
  Ip_profile = (
      prepend(0.0, eq.flux_surf_avg_Bp2)
      * int_dl_over_Bp
      / constants.CONSTANTS.mu_0
  )

  # q = dPhi/dpsi in COCOS 11.
  q = prepend(eq.q[0], eq.q)
  Phi = scipy.integrate.cumulative_trapezoid(y=q, x=psi, initial=0.0)

  # R extrema bound the plasma, so a non-positive one means the tracer left the
  # entry unwritten; it would otherwise reach R_in / R_out and later give NaNs.
  unwritten = np.flatnonzero((eq.R_min <= 0.0) | (eq.R_max <= 0.0))
  if unwritten.size:
    i = unwritten[0]
    raise ValueError(
        f'Flux surface {i} of {len(eq.R_min)} (psi = {eq.psi[i]} Wb/rad) has a'
        f' non-positive major radius extremum (R_min = {eq.R_min[i]}, R_max ='
        f' {eq.R_max[i]}); it did not trace.'
    )

  R_in = prepend(eq.R_axis, eq.R_min)
  R_out = prepend(eq.R_axis, eq.R_max)
  a_minor_local = (eq.R_max - eq.R_min) / 2.0
  R_geo_local = (eq.R_max + eq.R_min) / 2.0

  # A traced surface can still collapse to a point, most often the innermost
  # one, where a zero minor radius makes the shape parameters non-finite.
  resolved = a_minor_local > 0.0
  if not resolved.any():
    raise ValueError(
        'Every traced flux surface has zero minor radius indicating the ' \
        'TokaMaker equilibrium is degenerate.'
    )
  index = np.arange(len(a_minor_local))
  fill = lambda profile: (
      profile
      if resolved.all()
      else np.interp(index, index[resolved], profile)
  )
  a_resolved = a_minor_local[resolved]
  elongation = fill(
      (eq.Z_max[resolved] - eq.Z_min[resolved]) / (2.0 * a_resolved)
  )
  delta_upper_face = fill(
      (R_geo_local[resolved] - eq.R_at_Zmax[resolved]) / a_resolved
  )
  delta_lower_face = fill(
      (R_geo_local[resolved] - eq.R_at_Zmin[resolved]) / a_resolved
  )
  # The axis is a point, so it takes the shape of the innermost traced surface.
  elongation = prepend(elongation[0], elongation)
  delta_upper_face = prepend(delta_upper_face[0], delta_upper_face)
  delta_lower_face = prepend(delta_lower_face[0], delta_lower_face)

  R_major = (eq.R_max[-1] + eq.R_min[-1]) / 2.0
  a_minor = (eq.R_max[-1] - eq.R_min[-1]) / 2.0
  if a_minor <= 0.0:
    raise ValueError(
        'The outermost traced flux surface has zero minor radius'
        f' (R_min = R_max = {eq.R_max[-1]}); it cannot define the plasma'
        ' boundary.'
    )
  # F0 carries the sign of the toroidal field direction, which TORAX does not
  # track: B_0 enters as a magnitude, e.g. rho = sqrt(Phi_b/(pi*B_0)).
  B_0 = np.abs(eq.F0) / R_major

  # dV/drho_norm, from dV/dpsi and dpsi/drho_norm = 2*Phi_b*rho_norm/q.
  rhon = np.sqrt(np.abs(Phi / Phi[-1]))
  vpr = 4 * np.pi * np.abs(Phi[-1]) * rhon / (F * flux_surf_avg_1_over_R2)

  return standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.TOKAMAKER,
      Ip_from_parameters=Ip_from_parameters,
      R_major=np.array(R_major),
      a_minor=np.array(a_minor),
      B_0=np.array(B_0),
      psi=psi,
      Ip_profile=Ip_profile,
      Phi=Phi,
      R_in=R_in,
      R_out=R_out,
      F=F,
      int_dl_over_Bp=int_dl_over_Bp,
      flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
      flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
      flux_surf_avg_grad_psi=flux_surf_avg_grad_psi,
      flux_surf_avg_grad_psi2=flux_surf_avg_grad_psi2,
      flux_surf_avg_grad_psi2_over_R2=flux_surf_avg_grad_psi2_over_R2,
      flux_surf_avg_B2=flux_surf_avg_B2,
      flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
      delta_upper_face=delta_upper_face,
      delta_lower_face=delta_lower_face,
      elongation=elongation,
      vpr=vpr,
      face_centers=face_centers,
      hires_factor=hires_factor,
      # The outermost sampled surface is the LCFS, so no diverted edge model.
      diverted=None,
      connection_length_target=None,
      connection_length_divertor=None,
      angle_of_incidence_target=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
      z_magnetic_axis=np.array(eq.Z_axis),
  )
