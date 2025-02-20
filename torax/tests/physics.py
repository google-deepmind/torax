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

"""Unit tests for torax.physics."""

import dataclasses
from typing import Callable
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax import physics
from torax import state
from torax.fvm import cell_variable
from torax.geometry import circular_geometry
from torax.geometry import standard_geometry
from torax.sources import generic_current_source
from torax.sources import source_profiles
from torax.tests.test_lib import torax_refs


_trapz = jax.scipy.integrate.trapezoid


class PhysicsTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.physics` module."""

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_q_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_q_from_psi` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
        )
    )

    q_face_jax, q_cell_jax = physics.calc_q_from_psi(
        geo,
        references.psi,
        dynamic_runtime_params_slice.numerics.q_correction_factor,
    )

    # Make ground truth
    def calc_q_from_psi(runtime_params, geo):
      """Reference implementation from PINT."""
      iota = np.zeros(geo.torax_mesh.nx + 1)  # on face grid
      # We use the reference value of psi here because the original code
      # for calculating psi depends on FiPy, and we don't want to install that
      iota[1:] = np.abs(
          references.psi_face_grad[1:] / (2 * geo.Phib * geo.rho_face_norm[1:])
      )
      iota[0] = np.abs(
          references.psi_face_grad[1] / (2 * geo.Phib * geo.drho_norm)
      )
      q = 1 / iota
      q *= runtime_params.numerics.q_correction_factor

      def face_to_cell(face):
        cell = np.zeros(geo.torax_mesh.nx)
        cell[:] = 0.5 * (face[1:] + face[:-1])
        return cell

      q_cell = face_to_cell(q)
      return q, q_cell

    q_face_np, q_cell_np = calc_q_from_psi(runtime_params, geo)

    np.testing.assert_allclose(q_face_jax, q_face_np)
    np.testing.assert_allclose(q_cell_jax, q_cell_np)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_update_psi_from_j(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params
    source_runtime_params = generic_current_source.RuntimeParams()
    # Turn on the external current source.
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
            sources={
                generic_current_source.GenericCurrentSource.SOURCE_NAME: (
                    source_runtime_params
                )
            },
        )
    )
    # pylint: disable=protected-access
    if isinstance(geo, standard_geometry.StandardGeometry):
      psi = geo.psi_from_Ip
    else:
      bootstrap = source_profiles.BootstrapCurrentProfile.zero_profile(geo)
      external_current = generic_current_source.calculate_generic_current(
          mock.ANY,
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          geo=geo,
          source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
          unused_state=mock.ANY,
          unused_calculated_source_profiles=mock.ANY,
      )[0]
      currents = core_profile_setters._prescribe_currents(
          bootstrap_profile=bootstrap,
          external_current=external_current,
          dynamic_runtime_params_slice=dynamic_runtime_params_slice,
          geo=geo,
      )
      psi = core_profile_setters._update_psi_from_j(
          dynamic_runtime_params_slice.profile_conditions.Ip_tot,
          geo,
          currents.jtot_hires,
      ).value
    # pylint: enable=protected-access
    np.testing.assert_allclose(psi, references.psi.value)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_jtot_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_jtot_from_psi` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )
    # pylint: disable=invalid-name
    j, _, Ip_profile_face = physics.calc_jtot_from_psi(
        geo,
        references.psi,
    )
    # pylint: enable=invalid-name
    np.testing.assert_allclose(j, references.jtot, rtol=1e-5)

    if references.Ip_from_parameters:
      np.testing.assert_allclose(
          Ip_profile_face[-1],
          references.runtime_params.profile_conditions.Ip_tot * 1e6,
      )
    else:
      assert isinstance(geo, standard_geometry.StandardGeometry)
      np.testing.assert_allclose(
          Ip_profile_face[-1],
          geo.Ip_profile_face[-1],
      )

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_s_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_s_from_psi` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )

    s = physics.calc_s_from_psi(
        geo,
        references.psi,
    )

    np.testing.assert_allclose(s, references.s, rtol=1e-5)

  def test_fast_ion_fractional_heating_formula(self):
    """Compare `ion_heat_fraction` to a reference value."""
    # Inertial energy small compared to critical energy, all energy to ions.
    birth_energy = 1e-3
    temp_el = jnp.array(0.1, dtype=jnp.float32)
    fast_ion_mass = 1
    frac_i = physics.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 1.0, atol=1e-3)

    # Inertial energy large compared to critical energy, all energy to e-.
    birth_energy = 1e10
    frac_i = physics.fast_ion_fractional_heating_formula(
        birth_energy, temp_el, fast_ion_mass
    )
    np.testing.assert_allclose(frac_i, 0.0, atol=1e-9)

  # TODO(b/377225415): generalize to arbitrary number of ions.
  @parameterized.parameters([
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=1.0, ni=1.0, expected=1.0),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=1.0, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=2.0, Ai=4.0, ni=0.5, expected=0.5),
      dict(Aimp=20.0, Zimp=10.0, Zi=1.0, Ai=2.0, ni=0.9, expected=0.5),
      dict(Aimp=40.0, Zimp=20.0, Zi=1.0, Ai=2.0, ni=0.92, expected=0.5),
  ])
  # pylint: disable=invalid-name
  def test_calculate_weighted_Zeff(self, Aimp, Zimp, Zi, Ai, ni, expected):
    """Compare `_calculate_weighted_Zeff` to a reference value."""
    references = torax_refs.circular_references()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )
    ne = 1.0
    nimp = (ne - ni * Zi) / Zimp
    core_profiles = state.CoreProfiles(
        ne=cell_variable.CellVariable(
            value=jnp.array(ne),
            dr=jnp.array(1.0),
        ),
        ni=cell_variable.CellVariable(
            value=jnp.array(ni),
            dr=jnp.array(1.0),
        ),
        nimp=cell_variable.CellVariable(
            value=jnp.array(nimp),
            dr=jnp.array(1.0),
        ),
        temp_ion=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        temp_el=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        psi=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        psidot=cell_variable.CellVariable(
            value=jnp.array(0.0),
            dr=jnp.array(1.0),
        ),
        vloop_lcfs=jnp.array(0.0),
        currents=state.Currents.zeros(geo),
        q_face=jnp.array(0.0),
        s_face=jnp.array(0.0),
        Zi=Zi,
        Zi_face=Zi,
        Ai=Ai,
        Zimp=Zimp,
        Zimp_face=Zimp,
        Aimp=Aimp,
        nref=1e20,
    )
    # pylint: enable=invalid-name
    # pylint: disable=protected-access
    np.testing.assert_allclose(
        physics._calculate_weighted_Zeff(core_profiles), expected
    )
    # pylint: enable=protected-access

  # TODO(b/377225415): generalize to arbitrary number of ions.
  # pylint: disable=invalid-name
  @parameterized.parameters([
      dict(Zi=1.0, Zimp=10.0, Zeff=1.0, expected=1.0),
      dict(Zi=1.0, Zimp=5.0, Zeff=1.0, expected=1.0),
      dict(Zi=2.0, Zimp=10.0, Zeff=2.0, expected=0.5),
      dict(Zi=2.0, Zimp=5.0, Zeff=2.0, expected=0.5),
      dict(Zi=1.0, Zimp=10.0, Zeff=1.9, expected=0.9),
      dict(Zi=2.0, Zimp=10.0, Zeff=3.6, expected=0.4),
  ])
  def test_get_main_ion_dilution_factor(self, Zi, Zimp, Zeff, expected):
    """Unit test of `get_main_ion_dilution_factor`."""
    np.testing.assert_allclose(
        physics.get_main_ion_dilution_factor(Zi, Zimp, Zeff), expected
    )

  # pylint: enable=invalid-name

  def test_calculate_plh_scaling_factor(self):
    """Compare `calculate_plh_scaling_factor` to a reference value."""
    geo = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.0,
        hires_fac=4,
        Rmaj=6.0,
        Rmin=2.0,
        B0=5.0,
    )
    core_profiles = state.CoreProfiles(
        ne=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 2,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(2.0),
            dr=geo.drho_norm,
        ),
        ni=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 1,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(1.0),
            dr=geo.drho_norm,
        ),
        nimp=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        temp_ion=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        temp_el=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        psi=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        psidot=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        vloop_lcfs=jnp.array(0.0),
        currents=state.Currents.zeros(geo),
        q_face=jnp.array(0.0),
        s_face=jnp.array(0.0),
        Zi=1.0,
        Zi_face=1.0,
        Ai=3.0,
        Zimp=20,
        Zimp_face=20,
        Aimp=40,
        nref=1e20,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        currents=dataclasses.replace(
            core_profiles.currents,
            Ip_profile_face=jnp.ones_like(geo.rho_face_norm) * 10e6,
        ),
    )
    # pylint: disable=invalid-name
    P_LH_hi_dens, P_LH_low_dens, ne_min_P_LH = (
        physics.calculate_plh_scaling_factor(geo, core_profiles)
    )
    expected_PLH_hi_dens = (
        2.15 * 2**0.782 * 5**0.772 * 2**0.975 * 6**0.999 * (2.0141 / 3)
    )
    expected_PLH_low_dens = 0.36 * 10**0.27 * 5**1.25 * 6**1.23 * 3**0.08
    expected_ne_min_P_LH = 0.7 * 10**0.34 * 5**0.62 * 2.0**-0.95 * 3**0.4 / 10
    # pylint: enable=invalid-name
    np.testing.assert_allclose(P_LH_hi_dens / 1e6, expected_PLH_hi_dens)
    np.testing.assert_allclose(P_LH_low_dens / 1e6, expected_PLH_low_dens)
    np.testing.assert_allclose(ne_min_P_LH, expected_ne_min_P_LH)

  @parameterized.parameters([
      dict(elongation_LCFS=1.0),
      dict(elongation_LCFS=1.5),
  ])
  # pylint: disable=invalid-name
  def test_calculate_scaling_law_confinement_time(self, elongation_LCFS):
    """Compare `calculate_scaling_law_confinement_time` to reference values."""
    geo = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=elongation_LCFS,
        hires_fac=4,
        Rmaj=6.0,
        Rmin=2.0,
        B0=5.0,
    )
    core_profiles = state.CoreProfiles(
        ne=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 2,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(2.0),
            dr=geo.drho_norm,
        ),
        ni=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 2,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(1.0),
            dr=geo.drho_norm,
        ),
        nimp=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        temp_ion=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        temp_el=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        psi=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        psidot=cell_variable.CellVariable(
            value=jnp.ones_like(geo.rho_norm) * 0,
            left_face_grad_constraint=jnp.zeros(()),
            right_face_grad_constraint=None,
            right_face_constraint=jnp.array(0.0),
            dr=geo.drho_norm,
        ),
        vloop_lcfs=jnp.array(0.0),
        currents=state.Currents.zeros(geo),
        q_face=jnp.array(0.0),
        s_face=jnp.array(0.0),
        Zi=1.0,
        Zi_face=1.0,
        Ai=3.0,
        Zimp=20.0,
        Zimp_face=20.0,
        Aimp=40.0,
        nref=1e20,
    )
    core_profiles = dataclasses.replace(
        core_profiles,
        currents=dataclasses.replace(
            core_profiles.currents,
            Ip_profile_face=jnp.ones_like(geo.rho_face_norm) * 10e6,
        ),
    )
    Ploss = jnp.array(50e6)

    H89P = physics.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H89P'
    )
    H98 = physics.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H98'
    )
    H97L = physics.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H97L'
    )
    H20 = physics.calculate_scaling_law_confinement_time(
        geo, core_profiles, Ploss, 'H20'
    )

    expected_H89P = (
        0.038128
        * 10**0.85
        * 5**0.2
        * 20**0.1
        * 50**-0.5
        * 6**1.5
        * (1 / 3) ** 0.3
        * 3**0.50
        * elongation_LCFS**0.50
    )

    expected_H98 = (
        0.0562
        * 10**0.93
        * 5**0.15
        * 20**0.41
        * 50**-0.69
        * 6**1.97
        * (1 / 3) ** 0.58
        * 3**0.19
        * elongation_LCFS**0.78
    )

    expected_H97L = (
        0.023
        * 10**0.96
        * 5**0.03
        * 20**0.4
        * 50**-0.73
        * 6**1.83
        * (1 / 3) ** -0.06
        * 3**0.20
        * elongation_LCFS**0.64
    )

    expected_H20 = (
        0.053
        * 10**0.98
        * 5**0.22
        * 20**0.24
        * 50**-0.669
        * 6**1.71
        * (1 / 3) ** 0.35
        * 3**0.20
        * elongation_LCFS**0.80
    )
    # pylint: enable=invalid-name
    np.testing.assert_allclose(H89P, expected_H89P)
    np.testing.assert_allclose(H98, expected_H98)
    np.testing.assert_allclose(H97L, expected_H97L)
    np.testing.assert_allclose(H20, expected_H20)

  # pylint: disable=invalid-name
  def test_calc_Wpol(self):
    """Compare `calc_Wpol` to an analytical formula in circular geometry."""

    # Small inverse aspect ratio limit of circular geometry, such that we
    # approximate the simplest form of circular geometry where the analytical
    # Bpol formula is applicable.
    geo = circular_geometry.build_circular_geometry(
        n_rho=25,
        elongation_LCFS=1.0,
        Rmaj=100.0,
        Rmin=1.0,
        B0=5.0,
    )
    Ip_tot = 15
    # calculate high resolution jtot consistent with total current profile
    jtot_profile = (1 - geo.rho_hires_norm**2) ** 2
    denom = _trapz(jtot_profile * geo.spr_hires, geo.rho_hires_norm)
    Ctot = Ip_tot * 1e6 / denom
    jtot = jtot_profile * Ctot
    # pylint: disable=protected-access
    psi_cell_variable = core_profile_setters._update_psi_from_j(
        Ip_tot,
        geo,
        jtot,
    )
    _, _, Ip_profile_face = physics.calc_jtot_from_psi(
        geo,
        psi_cell_variable,
    )

    # Analytical formula for Bpol in circular geometry (Ampere's law)
    Bpol_bulk = (
        constants.CONSTANTS.mu0
        * Ip_profile_face[1:]
        / (2 * np.pi * geo.rho_face[1:])
    )
    Bpol = np.concatenate([np.array([0.0]), Bpol_bulk])

    expected_Wpol = _trapz(Bpol**2 * geo.vpr_face, geo.rho_face_norm) / (
        2 * constants.CONSTANTS.mu0
    )

    calculated_Wpol = physics.calc_Wpol(geo, psi_cell_variable)

    # Relatively low tolerence because the analytical formula is not exact for
    # our circular geometry, but approximates it at low inverse aspect ratio.
    np.testing.assert_allclose(calculated_Wpol, expected_Wpol, rtol=1e-3)

  # pylint: enable=invalid-name
  # pylint: enable=protected-access


if __name__ == '__main__':
  absltest.main()
