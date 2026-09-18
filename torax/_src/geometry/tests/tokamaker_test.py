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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import constants
from torax._src.geometry import pydantic_model
from torax._src.geometry import standard_geometry
from torax._src.geometry import tokamaker

# pylint: disable=invalid-name

_R_MAJOR = 6.2
_A_MINOR = 2.0
_B_0 = 5.3


def _circular_get_fsa(n_surfaces: int = 60, a_minor: float = _A_MINOR) -> dict:
  """Builds a dummy `TokaMaker_equilibrium.get_fsa()`-shaped dict for a concentric
  circular equilibrium, without using TokaMaker.

  Flux surfaces are circles of minor radius r about (R_major, 0) with a
  poloidal field uniform on each surface, so every flux surface average has a
  closed form. Quantities follow TokaMaker's conventions, i.e. poloidal flux in
  Wb/rad with B_p = |grad(psi)|/R.
  """
  r = np.linspace(a_minor / n_surfaces, a_minor, n_surfaces)
  q = 1.0 + 2.0 * (r / a_minor) ** 2
  # F = R*B_phi, constant with no diamagnetism.
  F = np.full_like(r, _R_MAJOR * _B_0)
  # q = r*B_0/(R_major*B_p) for a large aspect ratio circular surface.
  Bp = r * _B_0 / (_R_MAJOR * q)
  # With B_p uniform on the surface the averaging weight dl/B_p is uniform in
  # the poloidal angle, so <f> reduces to the plain angular mean.
  sqrt_term = np.sqrt(_R_MAJOR**2 - r**2)
  flux_surf_avg_1_over_R = 1.0 / sqrt_term
  flux_surf_avg_1_over_R2 = _R_MAJOR / sqrt_term**3
  # int(dl/B_p) over a circle of circumference 2*pi*r.
  int_dl_over_Bp = 2.0 * np.pi * r / Bp
  theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
  R_theta = _R_MAJOR + r[:, None] * np.cos(theta)[None, :]
  B2 = Bp[:, None] ** 2 + (F[:, None] / R_theta) ** 2
  # psi in Wb/rad, from dpsi/dr = R*B_p evaluated at the surface average of R.
  # Integrated from the axis, where B_p vanishes, so that the innermost traced
  # surface carries a nonzero flux as it does in a real `get_fsa()` grid.
  r_full = np.concatenate([[0.0], r])
  Bp_full = np.concatenate([[0.0], Bp])
  psi = np.cumsum(
      np.diff(r_full) * _R_MAJOR * 0.5 * (Bp_full[1:] + Bp_full[:-1])
  )
  return {
      'psi': psi,
      'q': q,
      'F': F,
      '<R>': np.full_like(r, _R_MAJOR),
      '<1/R>': flux_surf_avg_1_over_R,
      '<1/R^2>': flux_surf_avg_1_over_R2,
      # TokaMaker returns dV/dPsi = 2*pi*int(dl/B_p), negative for this
      # orientation of the normalized flux coordinate.
      'dV/dPsi': -2.0 * np.pi * int_dl_over_Bp,
      '<|grad psi|>': _R_MAJOR * Bp,
      '<|grad psi|^2>': Bp**2 * (_R_MAJOR**2 + r**2 / 2.0),
      '<Bp^2>': Bp**2,
      '<1/B^2>': np.mean(1.0 / B2, axis=1),
      'R_min': _R_MAJOR - r,
      'R_max': _R_MAJOR + r,
      'Z_min': -r,
      'Z_max': r,
      'R_at_Zmin': np.full_like(r, _R_MAJOR),
      'R_at_Zmax': np.full_like(r, _R_MAJOR),
      'psi_axis': 0.0,
      'R_axis': _R_MAJOR,
      'Z_axis': 0.0,
      'F_axis': _R_MAJOR * _B_0,
      'F0': _R_MAJOR * _B_0,
      'diverted': False,
  }


class TokaMakerGeometryTest(parameterized.TestCase):

  def test_build_geometry_from_get_fsa_dict(self):
    """A `get_fsa()` dict is accepted directly as the equilibrium input."""
    config = tokamaker.TokaMakerConfig(fsa_profiles=_circular_get_fsa())
    geo = config.build_geometry()
    self.assertEqual(geo.torax_mesh.nx, 25)

  def test_unrecognized_get_fsa_keys_are_ignored(self):
    """Extra keys, such as '<R>', do not need to be stripped by the caller."""
    fsa = _circular_get_fsa() | {'some_future_key': np.zeros(3)}
    tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()

  def test_inconsistent_profile_lengths_raises(self):
    fsa = _circular_get_fsa()
    fsa['q'] = fsa['q'][:-1]
    with self.assertRaisesRegex(ValueError, 'inconsistent lengths'):
      tokamaker.TokaMakerConfig(fsa_profiles=fsa)

  @parameterized.parameters(
      dict(name='R_major', expected=_R_MAJOR),
      dict(name='a_minor', expected=_A_MINOR),
      dict(name='B_0', expected=_B_0),
  )
  def test_reference_scalars(self, name, expected):
    """Reference scalars come from the outermost surface and F0."""
    geo = tokamaker.TokaMakerConfig(
        fsa_profiles=_circular_get_fsa()
    ).build_geometry()
    np.testing.assert_allclose(getattr(geo, name), expected, rtol=1e-12)

  def test_cocos11_unit_conversions(self):
    """TokaMaker psi is Wb/rad, so COCOS 11 quantities pick up 2*pi factors."""
    fsa = _circular_get_fsa()
    intermediates = tokamaker._construct_intermediates_from_tokamaker(  # pylint: disable=protected-access
        equilibrium=tokamaker.TokaMakerEquilibrium.from_dict(fsa),
        Ip_from_parameters=True,
        face_centers=np.linspace(0.0, 1.0, 26),
        hires_factor=4,
    )
    # Index 0 is the prescribed magnetic axis row, so compare from index 1.
    np.testing.assert_allclose(
        intermediates.psi[1:], 2 * np.pi * fsa['psi'], rtol=1e-12
    )
    np.testing.assert_allclose(
        intermediates.int_dl_over_Bp[1:],
        np.abs(fsa['dV/dPsi']) / (2 * np.pi),
        rtol=1e-12,
    )
    # <B^2> = <B_p^2> + F^2<1/R^2>, since F is constant on a flux surface.
    np.testing.assert_allclose(
        intermediates.flux_surf_avg_B2[1:],
        fsa['<Bp^2>'] + fsa['F'] ** 2 * fsa['<1/R^2>'],
        rtol=1e-12,
    )
    # The gradient averages are additionally smoothed near the axis by
    # StandardGeometryIntermediates, so only compare outside that region.
    rho_norm = np.sqrt(intermediates.Phi / intermediates.Phi[-1])[1:]
    outer = rho_norm > 2.0 * standard_geometry._RHO_SMOOTHING_LIMIT  # pylint: disable=protected-access
    for got, want in (
        (intermediates.flux_surf_avg_grad_psi, 2 * np.pi * fsa['<|grad psi|>']),
        (
            intermediates.flux_surf_avg_grad_psi2,
            4 * np.pi**2 * fsa['<|grad psi|^2>'],
        ),
        (
            intermediates.flux_surf_avg_grad_psi2_over_R2,
            4 * np.pi**2 * fsa['<Bp^2>'],
        ),
    ):
      np.testing.assert_allclose(got[1:][outer], want[outer], rtol=1e-12)

  def test_Ip_profile_matches_ampere_law(self):
    """Ip = <B_p^2> int(dl/B_p) / mu_0, with int(dl/B_p) = |dV/dPsi|/(2*pi)."""
    fsa = _circular_get_fsa()
    geo = tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()
    int_dl_over_Bp = np.abs(fsa['dV/dPsi'][-1]) / (2 * np.pi)
    expected = fsa['<Bp^2>'][-1] * int_dl_over_Bp / constants.CONSTANTS.mu_0
    np.testing.assert_allclose(geo.Ip_profile_face[-1], expected, rtol=1e-6)

  def test_shape_parameters(self):
    """Concentric circles have zero triangularity and unit elongation."""
    geo = tokamaker.TokaMakerConfig(
        fsa_profiles=_circular_get_fsa()
    ).build_geometry()
    np.testing.assert_allclose(geo.delta_face, 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.elongation_face, 1.0, rtol=1e-12)

  def test_collapsed_surface_takes_shape_from_its_neighbours(self):
    """A surface that failed to trace must not make the shape non-finite."""
    fsa = _circular_get_fsa()
    for key in ('R_min', 'R_max', 'Z_min', 'Z_max'):
      fsa[key][0] = _R_MAJOR if key.startswith('R') else 0.0  # positive, but a point
    geo = tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()
    self.assertTrue(np.all(np.isfinite(geo.elongation_face)))
    self.assertTrue(np.all(np.isfinite(geo.delta_face)))
    np.testing.assert_allclose(geo.delta_face, 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.elongation_face, 1.0, rtol=1e-12)

  def test_untraced_surface_raises(self):
    """A zeroed extremum must be rejected, not carried into R_in / R_out."""
    fsa = _circular_get_fsa()
    fsa['R_max'][40] = 0.0
    with self.assertRaisesRegex(ValueError, 'did not trace'):
      tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()

  def test_collapsed_boundary_surface_raises(self):
    """The boundary surface normalizes the grid, so it has no fallback."""
    fsa = _circular_get_fsa()
    fsa['R_min'][-1] = fsa['R_max'][-1]
    with self.assertRaisesRegex(ValueError, 'outermost traced flux surface'):
      tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()

  def test_fully_degenerate_equilibrium_raises(self):
    fsa = _circular_get_fsa()
    fsa['R_min'] = fsa['R_max'].copy()
    with self.assertRaisesRegex(ValueError, 'Every traced flux surface'):
      tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()

  def test_on_axis_values_are_prescribed(self):
    """No surface can be traced on axis, so TORAX prescribes the axis row."""
    geo = tokamaker.TokaMakerConfig(
        fsa_profiles=_circular_get_fsa()
    ).build_geometry()
    # The poloidal field vanishes on axis, so B is purely toroidal there.
    np.testing.assert_allclose(
        geo.gm5_face[0], (_R_MAJOR * _B_0 / _R_MAJOR) ** 2, rtol=1e-12
    )
    np.testing.assert_allclose(geo.Ip_profile_face[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(geo.R_in_face[0], _R_MAJOR, rtol=1e-12)
    np.testing.assert_allclose(geo.R_out_face[0], _R_MAJOR, rtol=1e-12)

  def test_reversed_toroidal_field(self):
    """F0 sets the field direction, which TORAX tracks only as a magnitude."""
    fsa = _circular_get_fsa()
    reversed_fsa = fsa | {
        'F': -fsa['F'],
        'F_axis': -fsa['F_axis'],
        'F0': -fsa['F0'],
    }
    geo = tokamaker.TokaMakerConfig(fsa_profiles=fsa).build_geometry()
    reversed_geo = tokamaker.TokaMakerConfig(
        fsa_profiles=reversed_fsa
    ).build_geometry()
    np.testing.assert_allclose(reversed_geo.B_0, _B_0, rtol=1e-12)
    for field in ('vpr', 'g0', 'g1', 'g2', 'g3', 'Phi', 'psi', 'F_face'):
      np.testing.assert_allclose(
          getattr(reversed_geo, field),
          getattr(geo, field),
          err_msg=f'Field "{field}" mismatch.',
      )

  def test_config_round_trips_through_json(self):
    """The config must serialize, since every output write dumps it to JSON."""
    config = pydantic_model.Geometry.from_dict({
        'geometry_type': 'tokamaker',
        'n_rho': 25,
        'fsa_profiles': _circular_get_fsa(),
    })
    restored = pydantic_model.Geometry.model_validate_json(
        config.model_dump_json()
    )
    geo, geo_restored = (
        config.build_provider(0.0),
        restored.build_provider(0.0),
    )
    for field in ('vpr', 'g0', 'g1', 'g2', 'g3', 'psi', 'Ip_profile_face'):
      np.testing.assert_allclose(
          getattr(geo, field),
          getattr(geo_restored, field),
          err_msg=f'Field "{field}" mismatch.',
      )

  def test_time_dependent_geometry_configs(self):
    """Equilibria at several times are interpolated by the standard provider."""
    config = pydantic_model.Geometry.from_dict({
        'geometry_type': 'tokamaker',
        'n_rho': 25,
        'geometry_configs': {
            0.0: {'fsa_profiles': _circular_get_fsa(a_minor=2.0)},
            1.0: {'fsa_profiles': _circular_get_fsa(a_minor=2.2)},
        },
    })
    provider = config.build_provider
    np.testing.assert_allclose(provider(0.0).a_minor, 2.0, rtol=1e-12)
    np.testing.assert_allclose(provider(1.0).a_minor, 2.2, rtol=1e-12)
    np.testing.assert_allclose(provider(0.5).a_minor, 2.1, rtol=1e-12)


if __name__ == '__main__':
  absltest.main()
