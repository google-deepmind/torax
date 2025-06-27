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
from torax._src.test_utils import torax_refs
from torax._src.torax_pydantic import torax_pydantic

_N_RHO = 10
_A_TOL = 1e-6
_R_TOL = 1e-6


class AngioniSauterTest(absltest.TestCase):

  def test_angioni_sauter_against_reference_values(self):
    """Reference values generated from Angioni-Sauter with NEOS verification."""
    reference = torax_refs.chease_references_Ip_from_chease()
    geometry_config = {
        'geometry_type': 'chease',
        'Ip_from_parameters': False,
        'n_rho': _N_RHO,
    }
    reference.config.update_fields({'geometry': geometry_config})
    torax_pydantic.set_grid(
        reference.config,
        reference.config.geometry.build_provider.torax_mesh,
        mode='force',
    )
    source_models = reference.config.sources.build_models()
    neoclassical_models = reference.config.neoclassical.build_models()
    dynamic_runtime_params_slice, geo = reference.get_dynamic_slice_and_geo()
    static_slice = build_runtime_params.build_static_params_from_config(
        reference.config
    )
    core_profiles = initialization.initial_core_profiles(
        static_slice,
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


_EXPECTED_CHI_NEO_I = np.array([
    -0.0,
    0.00509967,
    0.00863428,
    0.01176691,
    0.01466972,
    0.01749245,
    0.0203952,
    0.02349431,
    0.02687125,
    0.03045085,
    0.03423973,
])

_EXPECTED_CHI_NEO_E = np.array([
    -0.0,
    -0.00095521,
    -0.00146699,
    -0.00191542,
    -0.00230064,
    -0.00264251,
    -0.00298192,
    -0.00335926,
    -0.00381456,
    -0.00443535,
    -0.00538708,
])

_EXPECTED_D_NEO_E = np.array([
    0.00000000e00,
    7.58243986e-05,
    1.24788815e-04,
    1.61750414e-04,
    1.90600262e-04,
    2.15557205e-04,
    2.39291094e-04,
    2.62427361e-04,
    2.83449453e-04,
    2.93968052e-04,
    2.62777860e-04,
])

_EXPECTED_V_NEO_E = np.array([
    0.00000000e00,
    9.48594278e-06,
    1.30951909e-05,
    1.92213440e-05,
    2.90025056e-05,
    4.30261379e-05,
    6.35413440e-05,
    9.64522309e-05,
    1.58321980e-04,
    3.12319833e-04,
    1.03138682e-03,
])

_EXPECTED_V_NEO_WARE_E = np.array([
    -0.0,
    -0.00029796,
    -0.00031825,
    -0.00027813,
    -0.00023889,
    -0.00022863,
    -0.00024862,
    -0.00029331,
    -0.00043811,
    -0.00129203,
    -0.00155953,
])


if __name__ == '__main__':
  absltest.main()
