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

"""Unit tests for torax.physics.psi_calculations."""

from typing import Callable
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.geometry import standard_geometry
from torax.physics import psi_calculations
from torax.sources import generic_current_source
from torax.sources import source_profiles
from torax.tests.test_lib import torax_refs


_trapz = jax.scipy.integrate.trapezoid


class PsiCalculationsTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.physics.psi_calculations` module."""

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_q(self, references_getter: Callable[[], torax_refs.References]):
    """Compare `calc_q` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params
    _, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
        )
    )

    q_face_jax, q_cell_jax = psi_calculations.calc_q(
        geo,
        references.psi,
    )

    # Make ground truth
    def calc_q(geo):
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
      q *= geo.q_correction_factor

      def face_to_cell(face):
        cell = np.zeros(geo.torax_mesh.nx)
        cell[:] = 0.5 * (face[1:] + face[:-1])
        return cell

      q_cell = face_to_cell(q)
      return q, q_cell

    q_face_np, q_cell_np = calc_q(geo)

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
  def test_calc_jtot(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_jtot` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )
    # pylint: disable=invalid-name
    j, _, Ip_profile_face = psi_calculations.calc_jtot(
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
  def test_calc_s(self, references_getter: Callable[[], torax_refs.References]):
    """Compare `calc_s` to a reference value."""
    references = references_getter()
    geo = references.geometry_provider(
        references.runtime_params.numerics.t_initial
    )

    s = psi_calculations.calc_s(
        geo,
        references.psi,
    )

    np.testing.assert_allclose(s, references.s, rtol=1e-5)

  # pylint: disable=invalid-name
  def test_calc_Wpol(self):
    """Compare `calc_Wpol` to an analytical formula in circular geometry."""

    # Small inverse aspect ratio limit of circular geometry, such that we
    # approximate the simplest form of circular geometry where the analytical
    # Bpol formula is applicable.
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=25,
        elongation_LCFS=1.0,
        Rmaj=100.0,
        Rmin=1.0,
        B0=5.0,
    ).build_geometry()
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
    _, _, Ip_profile_face = psi_calculations.calc_jtot(
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

    calculated_Wpol = psi_calculations.calc_Wpol(geo, psi_cell_variable)

    # Relatively low tolerence because the analytical formula is not exact for
    # our circular geometry, but approximates it at low inverse aspect ratio.
    np.testing.assert_allclose(calculated_Wpol, expected_Wpol, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
