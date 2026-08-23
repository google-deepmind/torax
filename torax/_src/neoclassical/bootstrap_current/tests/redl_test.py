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

"""Tests for Redl bootstrap current, including old vs multi-species."""

from collections.abc import Mapping
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry as geometry_lib
from torax._src.neoclassical.bootstrap_current import redl
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.formulas import redl as redl_formulas
from torax._src.physics import collisions
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6
# Axis (face 0) is singular in ν* because ε→0; compare off-axis faces.
_OFF_AXIS = slice(1, None)


def _build_redl_state(
    plasma_composition: Mapping[str, Any] | None = None,
) -> tuple[
    runtime_params_lib.RuntimeParams, geometry_lib.Geometry, state.CoreProfiles
]:
  """Builds runtime params, geometry, and initial core profiles for Redl."""
  torax_config = model_config.ToraxConfig.from_dict({
      'profile_conditions': {},
      'numerics': {},
      'plasma_composition': dict(plasma_composition or {'Z_eff': 2.0}),
      'geometry': {
          'geometry_type': 'circular',
          'n_rho': _N_RHO,
      },
      'transport': {},
      'solver': {},
      'pedestal': {},
      'sources': {},
      'neoclassical': {
          'bootstrap_current': {'model_name': 'redl'},
      },
  })
  params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )
  runtime_params, geo = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=torax_config.numerics.t_initial,
          runtime_params_provider=params_provider,
          geometry_provider=torax_config.geometry.build_provider,
          is_initialization=True,
      )
  )
  core_profiles = initialization.initial_core_profiles(
      runtime_params,
      geo,
      source_models=torax_config.sources.build_models(),
      neoclassical_models=torax_config.neoclassical.build_models(),
  )
  return runtime_params, geo, core_profiles


def _redl_l_coefficients(
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    *,
    n_i_for_nu: jnp.ndarray,
    Z_for_nu: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Redl L31, L32, L34=L31, α, and ν_i* for a chosen ν_i* density/charge.

  The old single-fluid path used bundled main-ion density and Z_eff^4.
  The NEO multi-species path uses dens_sum and Z_ion^4.
  lnΛ_ii always uses main-ion density and Z_ion (Sauter 18e).
  """
  n_e = core_profiles.n_e
  n_i = core_profiles.n_i
  T_e = core_profiles.T_e
  T_i = core_profiles.T_i
  f_trap = formulas.calculate_f_trap(geo)
  log_lambda_ei = collisions.calculate_log_lambda_ei(
      T_e.face_value(), n_e.face_value()  # pyrefly: ignore[bad-argument-type]
  )
  log_lambda_ii = collisions.calculate_log_lambda_ii(
      T_i.face_value(), n_i.face_value(), core_profiles.Z_i_face  # pyrefly: ignore[bad-argument-type]
  )
  nu_e_star = formulas.calculate_nu_e_star(
      q=core_profiles.q_face,
      geo=geo,
      n_e=n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      T_e=T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_eff=core_profiles.Z_eff_face,
      log_lambda_ei=log_lambda_ei,
  )
  nu_i_star = formulas.calculate_nu_i_star(
      q=core_profiles.q_face,
      geo=geo,
      n_i=n_i_for_nu,
      T_i=T_i.face_value(),  # pyrefly: ignore[bad-argument-type]
      Z_i=Z_for_nu,
      log_lambda_ii=log_lambda_ii,
  )
  L31 = redl_formulas.calculate_L31(f_trap, nu_e_star, core_profiles.Z_eff_face)
  L32 = redl_formulas.calculate_L32(f_trap, nu_e_star, core_profiles.Z_eff_face)
  L34 = L31
  alpha = redl_formulas.calculate_alpha(
      f_trap, nu_i_star, core_profiles.Z_eff_face
  )
  return L31, L32, L34, alpha, nu_i_star


def _old_single_fluid_j_face(
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
    L31: jnp.ndarray,
    L32: jnp.ndarray,
    L34: jnp.ndarray,
    alpha: jnp.ndarray,
    bootstrap_multiplier: float = 1.0,
) -> jnp.ndarray:
  """Pre-NEO Redl drive: bundled p_i and ∇ln n_i (main-ion density)."""
  n_e = core_profiles.n_e
  n_i = core_profiles.n_i
  T_e = core_profiles.T_e
  T_i = core_profiles.T_i
  p_e = core_profiles.pressure_thermal_e
  p_i = core_profiles.pressure_thermal_i
  psi = core_profiles.psi
  prefactor = -geo.F_face * bootstrap_multiplier * 2 * jnp.pi / geo.B_0
  pe = p_e.face_value()
  pi = p_i.face_value()
  dpsi = psi.face_grad()
  dlnne = n_e.face_grad() / n_e.face_value()
  dlnni = n_i.face_grad() / n_i.face_value()
  dlnte = T_e.face_grad() / T_e.face_value()
  dlnti = T_i.face_grad() / T_i.face_value()
  global_coeff = jnp.concatenate([jnp.zeros(1), prefactor[1:] / dpsi[1:]])
  return global_coeff * (
      L31 * pe * dlnne
      + L31 * pi * dlnni
      + (L31 + L32) * pe * dlnte
      + (L31 + L34 * alpha) * pi * dlnti
  )


def _old_single_fluid_redl_j_face(
    geo: geometry_lib.Geometry,
    core_profiles: state.CoreProfiles,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Old Redl: ν_i* ∝ n_i Z_eff^4 and bundled ion drives.

  Returns:
    j_parallel_bootstrap_face, nu_i_star, alpha
  """
  L31, L32, L34, alpha, nu_i_star = _redl_l_coefficients(
      geo,
      core_profiles,
      n_i_for_nu=core_profiles.n_i.face_value(),
      Z_for_nu=core_profiles.Z_eff_face,
  )
  j_face = _old_single_fluid_j_face(
      geo, core_profiles, L31, L32, L34, alpha
  )
  return j_face, nu_i_star, alpha


class RedlTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.runtime_params, self.geo, self.core_profiles = _build_redl_state(
        {'Z_eff': 2.0}
    )

  def test_redl_bootstrap_current_is_correct_shape(self):
    model = redl.RedlModel()
    result = model.calculate_bootstrap_current(
        self.runtime_params, self.geo, self.core_profiles
    )
    self.assertEqual(result.j_parallel_bootstrap.shape, (_N_RHO,))
    self.assertEqual(result.j_parallel_bootstrap_face.shape, (_N_RHO + 1,))

  def test_hydrogenic_plasma_matches_old_single_fluid(self):
    """At Z_eff=1, dens_sum=n_i and Z_ion=Z_eff, so old and new agree."""
    runtime_params, geo, core_profiles = _build_redl_state({
        'main_ion': 'D',
        'Z_eff': 1.0,
    })
    np.testing.assert_allclose(
        core_profiles.Z_eff_face, 1.0, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        core_profiles.Z_i_face, 1.0, atol=_A_TOL, rtol=_R_TOL
    )

    new = redl.RedlModel().calculate_bootstrap_current(
        runtime_params, geo, core_profiles
    )
    j_old, nu_i_old, _ = _old_single_fluid_redl_j_face(geo, core_profiles)

    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=True
    )
    dens_sum = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=core_profiles.n_e.face_value()
    )
    np.testing.assert_allclose(
        dens_sum,
        core_profiles.n_i.face_value(),
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        new.j_parallel_bootstrap_face[_OFF_AXIS],
        j_old[_OFF_AXIS],
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    _, _, _, _, nu_i_new = _redl_l_coefficients(
        geo,
        core_profiles,
        n_i_for_nu=dens_sum,
        Z_for_nu=core_profiles.Z_i_face,
    )
    np.testing.assert_allclose(
        nu_i_new[_OFF_AXIS],
        nu_i_old[_OFF_AXIS],
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_multi_species_nu_i_star_uses_dens_sum_and_Z_ion(self):
    """NEO ν_i* is dens_sum Z_ion^4, not the old n_i Z_eff^4."""
    geo = self.geo
    core_profiles = self.core_profiles
    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=True
    )
    dens_sum = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=core_profiles.n_e.face_value()
    )
    n_i_face = core_profiles.n_i.face_value()
    Z_i = core_profiles.Z_i_face
    Z_eff = core_profiles.Z_eff_face

    _, _, _, _, nu_i_new = _redl_l_coefficients(
        geo, core_profiles, n_i_for_nu=dens_sum, Z_for_nu=Z_i
    )
    _, nu_i_old, _ = _old_single_fluid_redl_j_face(geo, core_profiles)

    expected_ratio = (dens_sum / n_i_face) * (Z_i / Z_eff) ** 4
    np.testing.assert_allclose(
        nu_i_new[_OFF_AXIS] / nu_i_old[_OFF_AXIS],
        expected_ratio[_OFF_AXIS],
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    # Impurities raise Z_eff above Z_ion, so the old Z_eff^4 path
    # overstates ν_i*.
    self.assertTrue(np.all(Z_eff[_OFF_AXIS] > Z_i[_OFF_AXIS]))
    self.assertTrue(np.all(nu_i_new[_OFF_AXIS] < nu_i_old[_OFF_AXIS]))
    # dens_sum includes impurities, so it exceeds bundled main-ion density.
    self.assertTrue(np.all(dens_sum[_OFF_AXIS] > n_i_face[_OFF_AXIS]))

  def test_multi_species_bootstrap_differs_from_old_at_elevated_Z_eff(self):
    """At Z_eff=2 the NEO assembly changes j_bs relative to the old path."""
    new = redl.RedlModel().calculate_bootstrap_current(
        self.runtime_params, self.geo, self.core_profiles
    )
    j_old, _, _ = _old_single_fluid_redl_j_face(self.geo, self.core_profiles)
    j_new = new.j_parallel_bootstrap_face
    self.assertTrue(np.all(np.isfinite(j_new)))
    self.assertTrue(np.all(np.isfinite(j_old)))

    rel = (j_new[_OFF_AXIS] - j_old[_OFF_AXIS]) / j_old[_OFF_AXIS]
    # The collisionality fix is a tens-of-percent effect at Z_eff=2.
    self.assertGreater(float(np.mean(np.abs(rel))), 0.05)
    np.testing.assert_array_less(np.abs(rel), np.ones_like(rel))

  def test_bootstrap_difference_is_dominated_by_collisionality(self):
    """Most of the old→new j_bs change is ν_i*→α, not the drive split."""
    geo = self.geo
    core_profiles = self.core_profiles
    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=True
    )
    dens_sum = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=core_profiles.n_e.face_value()
    )
    L31, L32, L34, alpha_new, _ = _redl_l_coefficients(
        geo,
        core_profiles,
        n_i_for_nu=dens_sum,
        Z_for_nu=core_profiles.Z_i_face,
    )
    _, _, _, alpha_old, _ = _redl_l_coefficients(
        geo,
        core_profiles,
        n_i_for_nu=core_profiles.n_i.face_value(),
        Z_for_nu=core_profiles.Z_eff_face,
    )
    j_new = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=core_profiles.n_e,
        T_e=core_profiles.T_e,
        p_e=core_profiles.pressure_thermal_e,
        ion_species=ion_species,
        psi=core_profiles.psi,
        geo=geo,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha=alpha_new,
    ).j_parallel_bootstrap_face
    j_old = _old_single_fluid_j_face(
        geo, core_profiles, L31, L32, L34, alpha_old
    )
    # New drives with old α ≈ old j_bs; old drives with new α ≈ new j_bs.
    j_new_drive_old_alpha = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=core_profiles.n_e,
        T_e=core_profiles.T_e,
        p_e=core_profiles.pressure_thermal_e,
        ion_species=ion_species,
        psi=core_profiles.psi,
        geo=geo,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha=alpha_old,
    ).j_parallel_bootstrap_face
    j_old_drive_new_alpha = _old_single_fluid_j_face(
        geo, core_profiles, L31, L32, L34, alpha_new
    )

    delta_full = j_new[_OFF_AXIS] - j_old[_OFF_AXIS]
    delta_alpha_only = j_old_drive_new_alpha[_OFF_AXIS] - j_old[_OFF_AXIS]
    delta_drive_only = j_new_drive_old_alpha[_OFF_AXIS] - j_old[_OFF_AXIS]
    rms = lambda x: float(np.sqrt(np.mean(np.square(np.asarray(x)))))
    # α (from ν_i*) explains the gap; the per-species drive split is small.
    self.assertGreater(rms(delta_alpha_only), 5.0 * rms(delta_drive_only))
    np.testing.assert_allclose(
        delta_alpha_only,
        delta_full,
        rtol=0.15,
        atol=0.0,
    )

  def test_old_path_is_more_sensitive_to_Z_eff_than_new(self):
    """Old Z_eff^4 ν_i* makes j_bs grow with Z_eff; NEO path does not."""
    mid = _N_RHO // 2  # off-axis face
    j_old = []
    j_new = []
    for zeff in (1.0, 2.0, 3.0):
      runtime_params, geo, core_profiles = _build_redl_state({'Z_eff': zeff})
      new = redl.RedlModel().calculate_bootstrap_current(
          runtime_params, geo, core_profiles
      )
      old_face, _, _ = _old_single_fluid_redl_j_face(geo, core_profiles)
      j_new.append(float(new.j_parallel_bootstrap_face[mid]))
      j_old.append(float(old_face[mid]))

    # Hydrogenic: both paths agree.
    np.testing.assert_allclose(j_new[0], j_old[0], atol=_A_TOL, rtol=_R_TOL)
    # Old |j_bs| rises with Z_eff; new stays flatter (NEO collisionality).
    self.assertGreater(abs(j_old[2]), abs(j_old[0]) * 1.2)
    self.assertLess(abs(j_new[2] - j_new[0]), abs(j_old[2] - j_old[0]))


if __name__ == '__main__':
  absltest.main()
