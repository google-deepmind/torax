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

from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import constants
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.geometry import standard_geometry
from torax.physics import psi_calculations
from torax.tests.test_lib import torax_refs


_trapz = jax.scipy.integrate.trapezoid


class PsiCalculationsTest(torax_refs.ReferenceValueTest):

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_q(self, references_getter: Callable[[], torax_refs.References]):
    references = references_getter()

    runtime_params = references.runtime_params
    _, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
        )
    )

    q_face_calculated, _ = psi_calculations.calc_q(
        geo,
        references.psi,
    )

    q_face_expected = references.q

    np.testing.assert_allclose(q_face_calculated, q_face_expected, rtol=1e-5)

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
    psi_cell_variable = initialization._update_psi_from_j(
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
