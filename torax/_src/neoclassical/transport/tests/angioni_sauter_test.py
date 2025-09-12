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
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.torax_pydantic import model_config

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class AngioniSauterTest(absltest.TestCase):

  def test_angioni_sauter_against_reference_values(self):
    """Reference values generated from Angioni-Sauter with NEOS verification."""
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
        )
    )

    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    result = angioni_sauter._calculate_angioni_sauter_transport(
        runtime_params, geo, core_profiles
    )
    np.testing.assert_allclose(
        result.chi_neo_i, _EXPECTED_CHI_NEO_I, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        result.chi_neo_e, _EXPECTED_CHI_NEO_E, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        result.D_neo_e, _EXPECTED_D_NEO_E, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        result.V_neo_e, _EXPECTED_V_NEO_E, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        result.V_neo_ware_e, _EXPECTED_V_NEO_WARE_E, atol=_A_TOL, rtol=_R_TOL
    )


# Reference values from running test code in a standalone manner.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
#
# The implementation was independently tested against NEOS up to the
# generation of the Kmn matrix.
_EXPECTED_CHI_NEO_I = np.array([
    -0.0,
    0.01225973,
    0.02229486,
    0.03122829,
    0.03897236,
    0.04574876,
    0.05185254,
    0.05726295,
    0.06153546,
    0.06325801,
    0.05919456,
])

_EXPECTED_CHI_NEO_E = np.array([
    -0.0,
    -0.00209616,
    -0.00307249,
    -0.00388121,
    -0.00455071,
    -0.00510756,
    -0.00560579,
    -0.00608657,
    -0.00658051,
    -0.00717423,
    -0.00750257,
])

_EXPECTED_D_NEO_E = np.array([
    0.0,
    0.00011798,
    0.00021212,
    0.00028574,
    0.00033818,
    0.00037626,
    0.00040472,
    0.0004229,
    0.00042484,
    0.0003935,
    0.00029244,
])

_EXPECTED_V_NEO_E = np.array([
    0.0,
    1.0494725e-05,
    1.0478798e-05,
    1.4456407e-05,
    2.5235609e-05,
    4.2482203e-05,
    6.8245361e-05,
    1.0981218e-04,
    1.8812368e-04,
    3.8060147e-04,
    1.1793068e-03,
])

_EXPECTED_V_NEO_WARE_E = np.array([
    -0.0,
    -0.00039555,
    -0.00043249,
    -0.00038382,
    -0.00033076,
    -0.00031509,
    -0.0003391,
    -0.0003929,
    -0.00056975,
    -0.00161012,
    -0.00179094,
])


if __name__ == '__main__':
  absltest.main()
