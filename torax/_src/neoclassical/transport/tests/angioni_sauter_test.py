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

  def test_calculate_Kmn_banana_limit_matches_K22e_0(self):
    ftrap = np.array([0.0, 0.2, 0.5])
    ftrap_d = np.array([0.0, 0.18, 0.45])
    Z_eff = np.full_like(ftrap, 2.0)
    B2_avg_Bm2_avg = np.full_like(ftrap, 1.05)
    nu_e_star = np.zeros_like(ftrap)
    nu_i_star = np.zeros_like(ftrap)
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
    F_ftrap_d = angioni_sauter._Fmn_X(ftrap_d, Z_eff)
    expected_K22e_0 = (
        -(13.0 / 8.0 + 1.0 / (np.sqrt(2.0) * Z_eff)) * F_ftrap_d[:, 1, 1]
    )
    np.testing.assert_allclose(
        Kmn_e[:, 1, 1], expected_K22e_0, atol=_A_TOL, rtol=_R_TOL
    )
    self.assertTrue(np.all(Kmn_e[1:, 1, 1] < 0.0))


# Reference values from running test code in a standalone manner.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
#
# The implementation was independently tested against NEOS up to the
# generation of the Kmn matrix.
_V_CONV = np.array([
    8.12487624e-04,
    8.12487624e-04,
    1.52311168e-04,
    6.04049750e-05,
    6.07982420e-05,
    8.92173399e-05,
    1.46361460e-04,
    2.66734324e-04,
    5.66525965e-04,
    1.50377518e-03,
    4.87797441e-03,
])
_V_WARE = np.array([
    -0.00237107,
    -0.00237107,
    -0.00260053,
    -0.00230196,
    -0.00195967,
    -0.00182954,
    -0.00191628,
    -0.00214489,
    -0.00297652,
    -0.00792481,
    -0.00801591,
])

_ANGIONI_SAUTER_REFERENCE_VALUES = transport_coeffs_lib.NeoclassicalTransport(
    chi_face_ion=np.array([
        0.5442650558424593,
        0.5442650558424593,
        0.18032915,
        0.08241574,
        0.06569636,
        0.06845025,
        0.07693944,
        0.09001002,
        0.10795760,
        0.12556257855725886,
        0.10507996,
    ]),
    chi_face_el=np.array([
        0.00379139,
        0.00379139,
        0.00237033,
        0.00150087,
        0.00106628,
        0.00087723,
        0.00080277,
        0.00078393,
        0.00079085,
        0.00054897,
        0.00043538,
    ]),
    d_face_el=np.array([
        5.11371876e-03,
        5.11371876e-03,
        1.64419017e-03,
        7.09800793e-04,
        5.42215812e-04,
        5.44016816e-04,
        5.86608760e-04,
        6.54437829e-04,
        7.37116441e-04,
        7.72355264e-04,
        5.17286534e-04,
    ]),
    v_face_el=_V_CONV + _V_WARE,
    v_face_el_ware=_V_WARE,
)

# Shaing correction only affects ions, so we can reuse the other values
_ANGIONI_SAUTER_SHAING_REFERENCE_VALUES = (
    transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=np.array([
            0.26725217564749176,
            0.31439655758643875,
            0.10936628,
            0.06339907,
            0.05882397,
            0.06559534,
            0.07573081,
            0.08949297,
            0.10773667,
            0.1254754018560309,
            0.10505861,
        ]),
        chi_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.chi_face_el,
        d_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.d_face_el,
        v_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el,
        v_face_el_ware=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el_ware,
    )
)

if __name__ == '__main__':
  absltest.main()
