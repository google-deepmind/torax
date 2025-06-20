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
from torax._src import constants
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.geometry import pydantic_model as geometry_pydantic_model
from torax._src.geometry import standard_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import source_profile_builders
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import torax_refs

_trapz = jax.scipy.integrate.trapezoid


class PsiCalculationsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_q(self, references_getter: Callable[[], torax_refs.References]):
    references = references_getter()

    _, geo = references.get_dynamic_slice_and_geo()

    q_face_calculated = psi_calculations.calc_q_face(geo, references.psi)
    np.testing.assert_allclose(q_face_calculated, references.q, rtol=1e-5)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_j_total(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    references = references_getter()
    geo = references.geometry_provider(references.config.numerics.t_initial)
    # pylint: disable=invalid-name
    j, _, Ip_profile_face = psi_calculations.calc_j_total(
        geo,
        references.psi,
    )
    # pylint: enable=invalid-name
    np.testing.assert_allclose(j, references.j_total, rtol=1e-5)

    if isinstance(geo, standard_geometry.StandardGeometry):
      if not geo.Ip_from_parameters:
        np.testing.assert_allclose(
            Ip_profile_face[-1],
            geo.Ip_profile_face[-1],
            rtol=1e-6,
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
    geo = references.geometry_provider(references.config.numerics.t_initial)

    s = psi_calculations.calc_s_face(geo, references.psi)
    np.testing.assert_allclose(s, references.s, rtol=1e-5)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_psidot(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    references = references_getter()
    references.config.update_fields(
        {'sources.generic_current.mode': 'MODEL_BASED'}
    )
    source_models = references.config.sources.build_models(
        neoclassical=references.config.neoclassical
    )
    dynamic_runtime_params_slice, geo = references.get_dynamic_slice_and_geo()
    source_profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        references.config
    )
    initial_core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
    )
    # Updates the calculated source profiles with the standard source profiles.
    source_profile_builders.build_standard_source_profiles(
        static_runtime_params_slice=static_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        core_profiles=initial_core_profiles,
        source_models=source_models,
        psi_only=True,
        calculate_anyway=True,
        calculated_source_profiles=source_profiles,
    )
    conductivity = source_models.conductivity.calculate_conductivity(
        geo,
        initial_core_profiles,
    )

    psidot_calculated = psi_calculations.calculate_psidot_from_psi_sources(
        psi_sources=sum(source_profiles.psi.values()),
        sigma=conductivity.sigma,
        resistivity_multiplier=dynamic_runtime_params_slice.numerics.resistivity_multiplier,
        psi=references.psi,
        geo=geo,
    )

    psidot_expected = references.psidot

    np.testing.assert_allclose(psidot_calculated, psidot_expected, atol=1e-6)

  # pylint: disable=invalid-name
  def test_calc_Wpol(self):
    # Small inverse aspect ratio limit of circular geometry, such that we
    # approximate the simplest form of circular geometry where the analytical
    # Bpol formula is applicable.
    geo = geometry_pydantic_model.CircularConfig(
        n_rho=25,
        elongation_LCFS=1.0,
        R_major=100.0,
        a_minor=1.0,
        B_0=5.0,
    ).build_geometry()
    Ip = 15e6
    # calculate high resolution j_total consistent with total current profile
    j_total_profile = (1 - geo.rho_hires_norm**2) ** 2
    denom = _trapz(j_total_profile * geo.spr_hires, geo.rho_hires_norm)
    Ctot = Ip / denom
    j_total = j_total_profile * Ctot
    psi_cell_variable = initialization.update_psi_from_j(
        Ip,
        geo,
        j_total,
    )
    _, _, Ip_profile_face = psi_calculations.calc_j_total(
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

    # Relatively low tolerance because the analytical formula is not exact for
    # our circular geometry, but approximates it at low inverse aspect ratio.
    np.testing.assert_allclose(calculated_Wpol, expected_Wpol, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
