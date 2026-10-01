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

from absl.testing import absltest
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import transport_coeffs as transport_coeffs_lib

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


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
    neoclassical_model = torax_config.neoclassical.build_model()

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
        neoclassical_model=neoclassical_model,
    )

    return runtime_params, geo, core_profiles

  def test_angioni_sauter_against_reference_values(self):
    """Reference values generated from running Angioni-Sauter."""
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    neoclassical_intermediates = formulas.compute_neoclassical_intermediates(
        geo, core_profiles
    )

    # Test raw Angioni-Sauter values
    result = angioni_sauter._calculate_angioni_sauter_transport(
        geometry=geo,
        core_profiles=core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
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
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    neoclassical_intermediates = formulas.compute_neoclassical_intermediates(
        geo, core_profiles
    )

    # Enable Shaing ion correction
    transport_params = angioni_sauter.AngioniSauterModelConfig(
        use_shaing_ion_correction=True
    ).build_runtime_params()

    # Test blended Angioni-Sauter + Shaing values
    result = angioni_sauter.AngioniSauterModel()._call_implementation(
        transport_params, geo, core_profiles, neoclassical_intermediates
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


# Reference values from running test code in a standalone manner.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
#
# The implementation was independently tested against NEOS up to the
# generation of the Kmn matrix.
_V_CONV = np.array([
    3.84305595e-07,
    3.84305595e-07,
    3.95211833e-07,
    5.48471882e-07,
    9.45926210e-07,
    1.57655305e-06,
    2.51472902e-06,
    4.02218832e-06,
    6.84799094e-06,
    1.37547927e-05,
    4.23170619e-05,
])
_V_WARE = np.array([
    -0.00038114,
    -0.00038114,
    -0.00041759,
    -0.00037123,
    -0.00032066,
    -0.00030646,
    -0.0003312,
    -0.00038565,
    -0.00056229,
    -0.00159816,
    -0.00178913,
])

_ANGIONI_SAUTER_REFERENCE_VALUES = transport_coeffs_lib.NeoclassicalTransport(
    chi_face_ion=np.array([
        0.00043435,
        0.00043435,
        0.00079160,
        0.00110976,
        0.00138541,
        0.00162655,
        0.00184376,
        0.00203633,
        0.00218851,
        0.00225017,
        0.00210714,
    ]),
    chi_face_el=np.array([
        5.76205656e-05,
        5.76205656e-05,
        7.49570352e-05,
        8.02645038e-06,
        -6.46121367e-05,
        -1.16471101e-04,
        -1.56052924e-04,
        -1.95016136e-04,
        -2.42476213e-04,
        -3.79547146e-04,
        -4.30913618e-04,
    ]),
    d_face_el=np.array([
        4.16454817e-06,
        4.16454817e-06,
        7.51341748e-06,
        1.01368454e-05,
        1.20045778e-05,
        1.33603494e-05,
        1.43740438e-05,
        1.50229323e-05,
        1.50958466e-05,
        1.39879660e-05,
        1.04093537e-05,
    ]),
    v_face_el=_V_CONV + _V_WARE,
    v_face_el_ware=_V_WARE,
)

# Shaing correction only affects ions, so we can reuse the other values
_ANGIONI_SAUTER_SHAING_REFERENCE_VALUES = (
    transport_coeffs_lib.NeoclassicalTransport(
        chi_face_ion=np.array([
            0.20242597,
            0.16813795,
            0.01959750,
            0.00395964,
            0.00217907,
            0.00194080,
            0.00198582,
            0.00210808,
            0.00222910,
            0.00227534,
            0.00212033,
        ]),
        chi_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.chi_face_el,
        d_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.d_face_el,
        v_face_el=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el,
        v_face_el_ware=_ANGIONI_SAUTER_REFERENCE_VALUES.v_face_el_ware,
    )
)

if __name__ == '__main__':
  absltest.main()
