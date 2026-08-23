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

import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from torax._src import constants
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.neoclassical.formulas import formulas
from torax._src.physics import collisions
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6
# Axis (face 0) is singular in 1/dψ; compare off-axis faces.
_OFF_AXIS = slice(1, None)


def _make_linear_profile(
    geo: geometry.Geometry,
    center: float,
    edge: float,
) -> cell_variable.CellVariable:
  """CellVariable linear in rho_norm from ``center`` to ``edge``."""
  value = center + (edge - center) * geo.rho_norm
  return cell_variable.CellVariable(
      value=value,
      face_centers=geo.rho_face_norm,
      right_face_constraint=jnp.asarray(edge),
      right_face_grad_constraint=None,
  )


class FormulasTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'Ip': 15e6,
            'current_profile_nu': 3,
            'n_e_nbar_is_fGW': True,
            'normalize_n_e_to_nbar': True,
            'nbar': 0.85,
            'n_e': {0: {0.0: 1.5, 1.0: 1.0}},
        },
        'numerics': {},
        'plasma_composition': {
            'Z_eff': 2.0,
        },
        'geometry': {
            'geometry_type': 'chease',
            'Ip_from_parameters': False,
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
    })

    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params, self.geo = (
        build_runtime_params.get_consistent_runtime_params_and_geometry(
            t=torax_config.numerics.t_initial,
            runtime_params_provider=params_provider,
            geometry_provider=torax_config.geometry.build_provider,
            is_initialization=True,
        )
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params,
        self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    log_lambda_ei = collisions.calculate_log_lambda_ei(
        self.core_profiles.T_e.face_value(), self.core_profiles.n_e.face_value()  # pyrefly: ignore[bad-argument-type]
    )
    self.nu_e_star = formulas.calculate_nu_e_star(
        q=self.core_profiles.q_face,
        geo=self.geo,
        n_e=self.core_profiles.n_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        T_e=self.core_profiles.T_e.face_value(),  # pyrefly: ignore[bad-argument-type]
        Z_eff=self.core_profiles.Z_eff_face,
        log_lambda_ei=log_lambda_ei,
    )

    self.f_trap = formulas.calculate_f_trap(self.geo)

  def test_calculate_f_trap_positive_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.4362384616678634
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_negative_triangularity(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        delta_face=np.array(-0.2),
        epsilon_face=np.array(0.1),
    )
    result = formulas.calculate_f_trap(geo)
    expected = 0.45134158459680895
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_gradient_on_axis(self):
    grad_fn = jax.grad(
        lambda geo: jnp.sum(formulas.calculate_f_trap(geo)),
        allow_int=True,
    )
    grad_geo = grad_fn(self.geo)

    for leaf in jax.tree_util.tree_leaves(grad_geo):
      if isinstance(leaf, (jax.Array, np.ndarray)) and jnp.issubdtype(
          leaf.dtype, jnp.inexact
      ):
        chex.assert_tree_all_finite(leaf)

  def test_calculate_f_trap_simple_model(self):
    epsilon = np.array(0.1)
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        epsilon_face=epsilon,
    )
    result = formulas.calculate_f_trap(geo, f_trap_model='simple')
    expected = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_LinLiu_model(self):
    epsilon = np.array(0.1)
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        epsilon_face=epsilon,
    )
    result = formulas.calculate_f_trap(geo, f_trap_model='LinLiu')
    sqrt_eps = np.sqrt(epsilon)
    denom = np.sqrt(1.0 - epsilon**2)
    f_tu = 1.0 - (1.0 - 1.5 * sqrt_eps + 0.5 * epsilon * sqrt_eps) / denom
    f_tl = 1.0 - (1.0 - epsilon) ** 2 / (
        denom * (1.0 + 1.46 * sqrt_eps + 0.2 * epsilon)
    )
    expected = 0.75 * f_tu + 0.25 * f_tl
    np.testing.assert_allclose(result, expected)

  def test_calculate_f_trap_RABBIT_model(self):
    R_out_face = np.array([6.0, 6.5, 7.0])
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        R_out_face=R_out_face,
    )
    result = formulas.calculate_f_trap(geo, f_trap_model='RABBIT')
    R_0 = R_out_face[0]
    epsilon_eff = (R_out_face - R_0) / R_0
    expected = 1.4624256 * np.sqrt(epsilon_eff) - 0.46 * epsilon_eff**1.5
    np.testing.assert_allclose(result, expected)

  def test_trapped_fraction_from_B_uniform_field_is_zero(self):
    B = np.full((4,), 2.0)
    weights = np.ones_like(B)
    result = formulas.trapped_fraction_from_B(B, weights)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)

  def test_trapped_fraction_from_B_circular_between_LinLiu_bounds(self):
    """Numerical f_t for B∝1/R on a circle lies between Lin-Liu bounds."""
    epsilon = 0.2
    R0 = 1.0
    a = epsilon * R0
    n_theta = 128
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    R = R0 + a * np.cos(theta)
    B = 1.0 / R
    weights = R
    f_t = float(formulas.trapped_fraction_from_B(B, weights))

    sqrt_eps = np.sqrt(epsilon)
    denom = np.sqrt(1.0 - epsilon**2)
    f_tu = 1.0 - (1.0 - 1.5 * sqrt_eps + 0.5 * epsilon * sqrt_eps) / denom
    f_tl = 1.0 - (1.0 - epsilon) ** 2 / (
        denom * (1.0 + 1.46 * sqrt_eps + 0.2 * epsilon)
    )
    self.assertGreaterEqual(f_t, f_tl - 1e-3)
    self.assertLessEqual(f_t, f_tu + 1e-3)

  def test_calculate_f_trap_numerical_model(self):
    result = formulas.calculate_f_trap(
        self.geo,
        f_trap_model='numerical',
        q_face=self.core_profiles.q_face,
    )
    self.assertEqual(result.shape, self.geo.rho_face_norm.shape)
    np.testing.assert_allclose(result[0], 0.0, atol=1e-6)
    self.assertTrue(np.all(result >= 0.0))
    self.assertTrue(np.all(result <= 1.0))
    self.assertGreater(float(result[-1]), float(result[1]))

  def test_calculate_f_trap_numerical_requires_q_face(self):
    with self.assertRaisesRegex(ValueError, 'q_face'):
      formulas.calculate_f_trap(self.geo, f_trap_model='numerical')

  def test_chease_geometry_has_no_surface_B(self):
    self.assertIsNone(self.geo.B_surface_face)
    self.assertIsNone(self.geo.fsa_weight_face)

  def test_calculate_f_trap_numerical_uses_stored_B(self):
    """Uniform stored |B| gives f_t = 0 and does not require q_face."""
    n_face = self.geo.rho_face_norm.shape[0]
    n_theta = geometry.N_THETA_SURFACE
    geo = dataclasses.replace(
        self.geo,
        B_surface_face=np.ones((n_face, n_theta)),
        fsa_weight_face=np.ones((n_face, n_theta)),
    )
    result = formulas.calculate_f_trap(geo, f_trap_model='numerical')
    np.testing.assert_allclose(result, 0.0, atol=1e-6)

  def test_neo_miller_includes_poloidal_field(self):
    """Finite q gives B_p > 0, so |B| exceeds |F|/R at the outboard midplane."""
    q_face = self.core_profiles.q_face
    B, _ = formulas._neo_miller_B_and_weights(self.geo, q_face)
    # theta=0 is the outboard midplane of the Miller surface.
    B_tor_omp = np.abs(self.geo.F_face) / np.maximum(self.geo.R_out_face, 1e-30)
    np.testing.assert_array_less(B_tor_omp[1:], np.asarray(B[1:, 0]))

  def test_calculate_f_trap_unknown_model_raises(self):
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        epsilon_face=np.array(0.1),
    )
    with self.assertRaisesRegex(ValueError, 'Unknown f_trap_model'):
      formulas.calculate_f_trap(geo, f_trap_model='not_a_model')  # type: ignore[arg-type]

  def test_calculate_Z_eff_from_ion_species_matches_core_profiles(self):
    """Species-summed Z_eff matches bundled face Z_eff without fast ions."""
    ion_species = formulas.build_ion_species_from_core_profiles(
        self.core_profiles,
        subtract_fast_ions=False,
    )
    Z_eff_face = formulas.calculate_Z_eff_from_ion_species(
        self.core_profiles, ion_species
    )
    np.testing.assert_allclose(
        Z_eff_face,
        self.core_profiles.Z_eff_face,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_ion_density_and_pressure_sum_face(self):
    """dens_sum / press_sum are face sums over thermal ion species."""
    ion_species = formulas.build_ion_species_from_core_profiles(
        self.core_profiles,
        subtract_fast_ions=False,
    )
    # D + T main ions plus Ne impurity.
    self.assertLen(ion_species, 3)

    dens_sum = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=self.core_profiles.n_e.face_value()
    )
    expected_dens = sum(s.n.face_value() for s in ion_species)
    np.testing.assert_allclose(dens_sum, expected_dens, atol=_A_TOL, rtol=_R_TOL)

    press_sum = formulas.calculate_ion_pressure_sum_face(
        ion_species, placeholder=self.core_profiles.n_e.face_value()
    )
    expected_press = sum(
        formulas.make_thermal_pressure(s.n, s.T).face_value()
        for s in ion_species
    )
    np.testing.assert_allclose(
        press_sum, expected_press, atol=_A_TOL, rtol=_R_TOL
    )
    # Shared T_i ⇒ cell-center pressure is dens_sum_cell * T_i * keV_to_J.
    keV_to_J = constants.CONSTANTS.keV_to_J
    dens_cell = sum(s.n.value for s in ion_species)
    press_cell = sum(s.n.value * s.T.value * keV_to_J for s in ion_species)
    np.testing.assert_allclose(
        press_cell,
        dens_cell * self.core_profiles.T_i.value * keV_to_J,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_empty_ion_species_sums_return_placeholder_zeros(self):
    """Empty ion_species yields zeros shaped like the provided placeholder."""
    placeholder = self.core_profiles.n_e.face_value()
    dens_sum = formulas.calculate_ion_density_sum_face(
        (), placeholder=placeholder
    )
    press_sum = formulas.calculate_ion_pressure_sum_face(
        (), placeholder=placeholder
    )
    np.testing.assert_array_equal(dens_sum, np.zeros_like(placeholder))
    np.testing.assert_array_equal(press_sum, np.zeros_like(placeholder))

  def test_analytic_bootstrap_handles_empty_and_zero_density_species(self):
    """Empty ions and zero-density species use placeholders without NaN/Inf."""
    geo = self.geo
    n_e = _make_linear_profile(geo, 2.0e20, 1.0e20)
    T_e = _make_linear_profile(geo, 8.0, 2.0)
    T_i = _make_linear_profile(geo, 6.0, 1.5)
    p_e = formulas.make_thermal_pressure(n_e, T_e)
    psi = _make_linear_profile(geo, 0.0, 1.0)
    ones = jnp.ones_like(geo.rho_face_norm)
    zeros = jnp.zeros_like(geo.rho_face_norm)

    empty_result = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=n_e,
        T_e=T_e,
        p_e=p_e,
        ion_species=(),
        psi=psi,
        geo=geo,
        L31=ones,
        L32=zeros,
        L34=ones,
        alpha=0.5 * ones,
    )
    self.assertTrue(np.all(np.isfinite(empty_result.j_parallel_bootstrap_face)))

    n_zero = core_profile_helpers.make_constant_core_profile(geo, 0.0)
    n_main = _make_linear_profile(geo, 1.2e20, 0.6e20)
    zero_species_result = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=n_e,
        T_e=T_e,
        p_e=p_e,
        ion_species=(
            formulas.IonSpeciesProfiles(n=n_main, T=T_i),
            formulas.IonSpeciesProfiles(n=n_zero, T=T_i),
        ),
        psi=psi,
        geo=geo,
        L31=ones,
        L32=zeros,
        L34=ones,
        alpha=0.5 * ones,
    )
    self.assertTrue(
        np.all(np.isfinite(zero_species_result.j_parallel_bootstrap_face))
    )

  def test_subtract_fast_ions_reduces_matching_impurity_density(self):
    """subtract_fast_ions=True removes impurity density matching FastIon.species."""
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {},
        'numerics': {},
        'plasma_composition': {
            'main_ion': 'D',
            'impurity': {
                'impurity_mode': 'n_e_ratios',
                'species': {'He3': 0.05},
            },
        },
        'geometry': {
            'geometry_type': 'circular',
            'n_rho': _N_RHO,
        },
        'transport': {},
        'solver': {},
        'pedestal': {},
        'sources': {},
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
    self.assertIn('He3', core_profiles.impurity_fractions)

    n_fast = 0.01e20
    fast_ion_he3 = fast_ion_lib.FastIon(
        species='He3',
        source='ICRH',
        n=core_profile_helpers.make_constant_core_profile(geo, n_fast),
        T=core_profile_helpers.make_constant_core_profile(geo, 100.0),
    )
    # Unrelated species must not affect He3 thermal density.
    fast_ion_h = fast_ion_lib.FastIon(
        species='H',
        source='NBI',
        n=core_profile_helpers.make_constant_core_profile(geo, 0.5e20),
        T=core_profile_helpers.make_constant_core_profile(geo, 50.0),
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        fast_ions=(fast_ion_he3, fast_ion_h),
    )

    with_fast = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=False
    )
    without_fast = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=True
    )
    # Species order: main ion D, then impurity He3.
    self.assertLen(with_fast, 2)
    np.testing.assert_allclose(
        with_fast[0].n.face_value(),
        without_fast[0].n.face_value(),
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        without_fast[1].n.face_value(),
        with_fast[1].n.face_value() - n_fast,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    dens_with = formulas.calculate_ion_density_sum_face(
        with_fast, placeholder=core_profiles.n_e.face_value()
    )
    dens_without = formulas.calculate_ion_density_sum_face(
        without_fast, placeholder=core_profiles.n_e.face_value()
    )
    np.testing.assert_allclose(
        dens_without, dens_with - n_fast, atol=_A_TOL, rtol=_R_TOL
    )

  def test_analytic_bootstrap_multi_species_drive_assembly(self):
    """NEO drive assembly: per-species L31 dens and L34α from p_s ∇ln T_s."""
    geo = self.geo
    n_e = _make_linear_profile(geo, 2.0e20, 1.0e20)
    T_e = _make_linear_profile(geo, 8.0, 2.0)
    T_main = _make_linear_profile(geo, 6.0, 1.5)
    T_imp = _make_linear_profile(geo, 4.0, 2.5)
    n_main = _make_linear_profile(geo, 1.2e20, 0.6e20)
    n_imp = _make_linear_profile(geo, 0.3e20, 0.15e20)
    p_e = formulas.make_thermal_pressure(n_e, T_e)
    psi = _make_linear_profile(geo, 0.0, 1.0)

    ion_species = (
        formulas.IonSpeciesProfiles(n=n_main, T=T_main),
        formulas.IonSpeciesProfiles(n=n_imp, T=T_imp),
    )
    ones = jnp.ones_like(geo.rho_face_norm)
    zeros = jnp.zeros_like(geo.rho_face_norm)
    L31 = ones
    L32 = zeros
    L34 = ones
    alpha = 0.5 * ones

    result = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=n_e,
        T_e=T_e,
        p_e=p_e,
        ion_species=ion_species,
        psi=psi,
        geo=geo,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha=alpha,
    )

    pe = p_e.face_value()
    p_main = formulas.make_thermal_pressure(n_main, T_main).face_value()
    p_imp = formulas.make_thermal_pressure(n_imp, T_imp).face_value()
    ion_density_drive = (
        p_main
        * (n_main.face_grad() / (n_main.face_value() + constants.CONSTANTS.eps))
        + p_imp
        * (n_imp.face_grad() / (n_imp.face_value() + constants.CONSTANTS.eps))
    )
    ion_temperature_drive = (
        p_main
        * (T_main.face_grad() / (T_main.face_value() + constants.CONSTANTS.eps))
        + p_imp
        * (T_imp.face_grad() / (T_imp.face_value() + constants.CONSTANTS.eps))
    )
    dlnne = n_e.face_grad() / n_e.face_value()
    dlnte = T_e.face_grad() / T_e.face_value()
    prefactor = -geo.F_face * 2.0 * jnp.pi / geo.B_0
    dpsi = psi.face_grad()
    global_coeff = jnp.concatenate(
        [jnp.zeros(1), prefactor[1:] / dpsi[1:]]
    )
    expected_face = global_coeff * (
        L31 * pe * dlnne
        + (L31 + L32) * pe * dlnte
        + L31 * ion_density_drive
        + (L31 + alpha * L34) * ion_temperature_drive
    )
    np.testing.assert_allclose(
        result.j_parallel_bootstrap_face,
        expected_face,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    # Distinct T_s must not collapse to press_sum × ∇ln T_main.
    lumped_ti_face = global_coeff * (
        L31 * pe * dlnne
        + (L31 + L32) * pe * dlnte
        + L31 * ion_density_drive
        + (L31 + alpha * L34)
        * (p_main + p_imp)
        * (T_main.face_grad() / T_main.face_value())
    )
    self.assertGreater(
        float(
            np.max(
                np.abs(
                    result.j_parallel_bootstrap_face[_OFF_AXIS]
                    - lumped_ti_face[_OFF_AXIS]
                )
            )
        ),
        0.0,
    )

    # Splitting one density into two equal species with the same T must
    # leave dens_sum, press_sum, and ion drives unchanged.
    T_i = T_main
    n_half_a = _make_linear_profile(geo, 0.75e20, 0.375e20)
    n_half_b = _make_linear_profile(geo, 0.75e20, 0.375e20)
    combined = (
        formulas.IonSpeciesProfiles(
            n=_make_linear_profile(geo, 1.5e20, 0.75e20), T=T_i
        ),
    )
    split = (
        formulas.IonSpeciesProfiles(n=n_half_a, T=T_i),
        formulas.IonSpeciesProfiles(n=n_half_b, T=T_i),
    )
    combined_result = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=n_e,
        T_e=T_e,
        p_e=p_e,
        ion_species=combined,
        psi=psi,
        geo=geo,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha=alpha,
    )
    split_result = formulas.calculate_analytic_bootstrap_current(
        bootstrap_multiplier=1.0,
        n_e=n_e,
        T_e=T_e,
        p_e=p_e,
        ion_species=split,
        psi=psi,
        geo=geo,
        L31=L31,
        L32=L32,
        L34=L34,
        alpha=alpha,
    )
    np.testing.assert_allclose(
        split_result.j_parallel_bootstrap_face,
        combined_result.j_parallel_bootstrap_face,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_calculate_poloidal_velocity_values_are_correct(self):
    poloidal_velocity = formulas.calculate_poloidal_velocity(
        T_i=self.core_profiles.T_i,
        n_i=self.core_profiles.n_i.face_value(),
        q=self.core_profiles.q_face,
        Z_i=self.core_profiles.Z_i_face,
        B_tor=np.ones_like(self.geo.rho_face_norm),
        B_total_squared=np.ones_like(self.geo.rho_face_norm),
        geo=self.geo,
    )
    np.testing.assert_allclose(
        _POLOIDAL_VELOCITY_EXPECTED,
        poloidal_velocity.face_value(),
        atol=_A_TOL,
        rtol=_R_TOL,
    )


# Reference values from running test code in a notebook.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
# If a change is expected to these reference values, the new values can be
# copied/pasted from the logs of a failing test.
_POLOIDAL_VELOCITY_EXPECTED = np.array([
    -2803.870156792,
    -4312.356519970,
    -5962.209635814,
    -6210.168368861,
    -6481.880862200,
    -6841.083404946,
    -7254.744224101,
    -7716.079881948,
    -8119.067018654,
    -7507.588263511,
    -5112.832873848,
])

if __name__ == '__main__':
  absltest.main()
