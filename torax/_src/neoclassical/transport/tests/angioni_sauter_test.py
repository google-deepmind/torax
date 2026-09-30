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

from absl.testing import absltest
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6

# pylint: disable=invalid-name


class AngioniSauterTest(absltest.TestCase):

  def _get_reference_runtime_params_geo_and_core_profiles(
      self,
  ) -> tuple[
      runtime_params_lib.RuntimeParams, geometry.Geometry, state.CoreProfiles
  ]:
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
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

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
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    return runtime_params, geo, core_profiles

  def test_calculate_Lmn_ion_symmetry(self):
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    n_faces = geo.F_face.shape[0]
    Kmn_e = np.ones((n_faces, 4, 4))
    Kmn_i = np.ones((n_faces, 2, 2))
    nu_e_star = np.ones(n_faces)
    nu_i_star = np.ones(n_faces)

    _, Lmn_i = angioni_sauter._calculate_Lmn(
        Kmn_e=Kmn_e,
        Kmn_i=Kmn_i,
        geo=geo,
        core_profiles=core_profiles,
        epsilon=geo.epsilon_face,
        nu_e_star=nu_e_star,
        nu_i_star=nu_i_star,
    )

    np.testing.assert_allclose(Lmn_i[:, 1, 0], -Lmn_i[:, 0, 1])
    self.assertTrue(np.all(Lmn_i[:, 1, 0] != 0.0))

  def test_calculate_Lmn_poloidal_gyroradius(self):
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    n_faces = geo.F_face.shape[0]
    Kmn_e = np.ones((n_faces, 4, 4))
    Kmn_i = np.ones((n_faces, 2, 2))
    nu_e_star = np.ones(n_faces)
    nu_i_star = np.ones(n_faces)

    Lmn_e, Lmn_i = angioni_sauter._calculate_Lmn(
        Kmn_e=Kmn_e,
        Kmn_i=Kmn_i,
        geo=geo,
        core_profiles=core_profiles,
        epsilon=geo.epsilon_face,
        nu_e_star=nu_e_star,
        nu_i_star=nu_i_star,
    )

    scale = 2.5
    scaled_geo = dataclasses.replace(geo, F_face=geo.F_face * scale)
    Lmn_e_scaled, Lmn_i_scaled = angioni_sauter._calculate_Lmn(
        Kmn_e=Kmn_e,
        Kmn_i=Kmn_i,
        geo=scaled_geo,
        core_profiles=core_profiles,
        epsilon=scaled_geo.epsilon_face,
        nu_e_star=nu_e_star,
        nu_i_star=nu_i_star,
    )
    np.testing.assert_allclose(
        Lmn_e_scaled[:, 0, 0],
        Lmn_e[:, 0, 0] * scale**2,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        Lmn_i_scaled[:, 1, 1],
        Lmn_i[:, 1, 1] * scale**2,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_angioni_sauter_against_reference_values(self):
    """Reference values generated from running Angioni-Sauter."""
    runtime_params, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )

    # Test raw Angioni-Sauter values
    result = angioni_sauter._calculate_angioni_sauter_transport(
        runtime_params, geo, core_profiles
    )
    np.testing.assert_allclose(
        result.chi_face_ion,
        _ANGIONI_SAUTER_REFERENCE_VALUES.chi_face_ion,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.chi_face_el,
        _ANGIONI_SAUTER_REFERENCE_VALUES.chi_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.d_face_el,
        _ANGIONI_SAUTER_REFERENCE_VALUES.d_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.v_face_el,
        _ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.v_face_el_ware,
        _ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el_ware,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_angioni_sauter_with_shaing_against_reference_values(self):
    """Reference values generated from Angioni-Sauter + Shaing ion correction."""
    runtime_params, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )

    # Enable Shaing ion correction
    modified_runtime_params = dataclasses.replace(
        runtime_params,
        neoclassical=dataclasses.replace(
            runtime_params.neoclassical,
            transport=angioni_sauter.AngioniSauterModelConfig(
                use_shaing_ion_correction=True
            ).build_runtime_params(),
        ),
    )

    # Test blended Angioni-Sauter + Shaing values
    result = angioni_sauter.AngioniSauterModel()._call_implementation(
        modified_runtime_params, geo, core_profiles
    )
    np.testing.assert_allclose(
        result.chi_face_ion,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.chi_face_ion,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.chi_face_el,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.chi_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.d_face_el,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.d_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.v_face_el,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.v_face_el,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.v_face_el_ware,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.v_face_el_ware,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_calculate_Kmn_populates_all_electron_elements(self):
    ftrap = np.array([0.0, 0.2, 0.5])
    ftrap_d = np.array([0.0, 0.18, 0.45])
    Z_eff = np.full_like(ftrap, 2.0)
    B2_avg_Bm2_avg = np.full_like(ftrap, 1.05)
    nu_e_star = np.array([0.01, 0.1, 1.0])
    nu_i_star = np.array([0.01, 0.1, 1.0])
    alpha_I = Z_eff - 1.0

    Kmn_e, _ = angioni_sauter._calculate_Kmn(
        ftrap=ftrap,
        ftrap_d=ftrap_d,
        Z_eff=Z_eff,
        B2_avg_Bm2_avg=B2_avg_Bm2_avg,
        nu_e_star=nu_e_star,
        nu_i_star=nu_i_star,
        alpha_I=alpha_I,
    )

    np.testing.assert_allclose(Kmn_e[:, 3, 3], Kmn_e[:, 0, 3])
    self.assertTrue(np.all(np.abs(Kmn_e[1:, 3, 3]) > 0.0))

    sauter_formulas = angioni_sauter.sauter_formulas
    expected_K23 = -sauter_formulas.calculate_L34(ftrap, nu_e_star, Z_eff)
    np.testing.assert_allclose(Kmn_e[:, 2, 3], expected_K23)
    np.testing.assert_allclose(Kmn_e[:, 3, 2], expected_K23)
    self.assertTrue(np.all(np.abs(Kmn_e[1:, 2, 3]) > 0.0))
    self.assertTrue(np.all(np.abs(Kmn_e[1:, 3, 2]) > 0.0))

  def test_calculate_Kmn_alpha_collisionality_dependence(self):
    ftrap = np.full(5, 0.5)
    ftrap_d = np.full(5, 0.4)
    Z_eff = np.full(5, 2.0)
    B2_avg_Bm2_avg = np.full(5, 1.1)
    nu_e_star = np.full(5, 0.1)
    nu_i_star = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    alpha_I = Z_eff - 1.0

    _, Kmn_i = angioni_sauter._calculate_Kmn(
        ftrap=ftrap,
        ftrap_d=ftrap_d,
        Z_eff=Z_eff,
        B2_avg_Bm2_avg=B2_avg_Bm2_avg,
        nu_e_star=nu_e_star,
        nu_i_star=nu_i_star,
        alpha_I=alpha_I,
    )

    alpha_0 = (
        -(0.62 + 1.5 * alpha_I)
        / (0.53 + alpha_I)
        * ((1.0 - ftrap) / (1.0 - 0.22 * ftrap - 0.19 * ftrap**2))
    )
    expected_alpha = (
        (alpha_0 + 0.25 * (1.0 - ftrap**2) * np.sqrt(nu_i_star))
        / (1.0 + 0.5 * np.sqrt(nu_i_star))
        + 0.315 * nu_i_star**2 * ftrap**6
    ) / (1.0 + 0.15 * nu_i_star**2 * ftrap**6)

    np.testing.assert_allclose(
        -Kmn_i[:, 0, 1], expected_alpha, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        Kmn_i[:, 1, 0], expected_alpha, atol=_A_TOL, rtol=_R_TOL
    )


# Reference values from running test code in a standalone manner.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
#
# The implementation was independently tested against NEOS up to the
# generation of the Kmn matrix.
_V_CONV = np.array([
    8.32386572e-04,
    8.32386572e-04,
    1.55741386e-04,
    6.17351839e-05,
    6.27622315e-05,
    9.40591223e-05,
    1.58896385e-04,
    3.00635993e-04,
    6.69625990e-04,
    1.89390369e-03,
    6.74650214e-03,
])
_V_WARE = np.array([
    -0.00242914,
    -0.00242914,
    -0.00265910,
    -0.00235265,
    -0.00202297,
    -0.00192882,
    -0.00208039,
    -0.00241751,
    -0.00351821,
    -0.00998076,
    -0.01108644,
])

_ANGIONI_SAUTER_REFERENCE_VALUES = transport_coeffs_lib.NeoclassicalTransport(
    chi_face_ion=np.array([
        0.5818619162839137,
        0.5818619162839137,
        0.19108337379184998,
        0.08719228857827582,
        0.07168513537507847,
        0.0794383259782818,
        0.09731044514322656,
        0.1276304426399899,
        0.1782827157417277,
        0.2548717570816584,
        0.26756778248552926,
    ]),
    chi_face_el=np.array([
        -0.1045502167705777,
        -0.1045502167705777,
        -0.02596546549147008,
        -0.009746608475807739,
        -0.007570150752927756,
        -0.008324996963023567,
        -0.010153495220952265,
        -0.01336443974304198,
        -0.019027691587194445,
        -0.02908972852387477,
        -0.034280493575281776,
    ]),
    d_face_el=np.array([
        5.46696553e-03,
        5.46696553e-03,
        1.74224858e-03,
        7.50942474e-04,
        5.91645106e-04,
        6.31345948e-04,
        7.41923956e-04,
        9.27966691e-04,
        1.21729060e-03,
        1.56775646e-03,
        1.31717999e-03,
    ]),
    v_face_el=_V_CONV + _V_WARE,
    v_face_el_ware=_V_WARE,
)

# Shaing correction only affects ions, so we can reuse the other values
_ANGIONI_SAUTER_SHAING_REFERENCE_VALUES = (
    transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=np.array([
            0.2717338312709709,
            0.32450791067261286,
            0.11474338767184676,
            0.06689101170272159,
            0.06409886503436546,
            0.07606229419855957,
            0.09573542591958789,
            0.12686160348908032,
            0.17788789174189854,
            0.2546667730988888,
            0.26749195112806423,
        ]),
        chi_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.chi_face_el,
        d_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.d_face_el,
        v_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el,
        v_face_el_ware=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el_ware,
    )
)

if __name__ == '__main__':
  absltest.main()
