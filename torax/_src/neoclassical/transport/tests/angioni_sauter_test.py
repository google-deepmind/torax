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
    0.01220085,
    0.02223608,
    0.03117304,
    0.03891618,
    0.04568965,
    0.05179111,
    0.0572006,
    0.06147531,
    0.06320731,
    0.0591895,
])

_EXPECTED_CHI_NEO_E = np.array([
    -0.0,
    -0.00210023,
    -0.0030792,
    -0.00388683,
    -0.0045548,
    -0.00511068,
    -0.0056083,
    -0.0060884,
    -0.00658147,
    -0.00717367,
    -0.00750323,
])

_EXPECTED_D_NEO_E = np.array([
    0.0,
    0.00011698,
    0.00021105,
    0.00028474,
    0.00033721,
    0.00037529,
    0.00040377,
    0.00042199,
    0.00042404,
    0.00039292,
    0.0002924,
])

_EXPECTED_V_NEO_E = np.array([
    0.0,
    1.07951440e-05,
    1.11015003e-05,
    1.54065751e-05,
    2.65710672e-05,
    4.42853751e-05,
    7.06387381e-05,
    1.12983269e-04,
    1.92360065e-04,
    3.86372126e-04,
    1.18868626e-03,
])

_EXPECTED_V_NEO_WARE_E = np.array([
    -0.0,
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


if __name__ == '__main__':
  absltest.main()
