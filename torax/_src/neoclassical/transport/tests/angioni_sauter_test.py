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

    dynamic_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )

    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice,
        geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    model = angioni_sauter.AngioniSauterModel()
    result = model.calculate_neoclassical_transport(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geometry=geo,
        core_profiles=core_profiles,
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
    0.01195613,
    0.02174449,
    0.03062838,
    0.03880287,
    0.04665049,
    0.05457236,
    0.06273228,
    0.07096089,
    0.07800027,
    0.07973496,
])

_EXPECTED_CHI_NEO_E = np.array([
    -0.0,
    -0.00203451,
    -0.00298301,
    -0.00379278,
    -0.0045157,
    -0.00518874,
    -0.00587365,
    -0.00663125,
    -0.00753242,
    -0.00873426,
    -0.00994788,
])

_EXPECTED_D_NEO_E = np.array([
    0.0,
    0.00011474,
    0.00020632,
    0.00027948,
    0.00033561,
    0.00038208,
    0.00042377,
    0.00046043,
    0.0004862,
    0.00048063,
    0.00039108,
])

_EXPECTED_V_NEO_E = np.array([
    0.00000000e00,
    1.02011963e-05,
    1.01671795e-05,
    1.41110590e-05,
    2.52396497e-05,
    4.38906098e-05,
    7.31203214e-05,
    1.22781454e-04,
    2.21681700e-04,
    4.78413008e-04,
    1.59532648e-03,
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
