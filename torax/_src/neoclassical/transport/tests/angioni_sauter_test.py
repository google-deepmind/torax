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
from torax._src.neoclassical.formulas import formulas
from torax._src.neoclassical.transport import angioni_sauter
from torax._src.torax_pydantic import model_config

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

  def test_calculate_Lmn_ion_symmetry(self):
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    n_faces = geo.F_face.shape[0]
    Kmn_e = np.ones((n_faces, 4, 4))
    Kmn_i = np.ones((n_faces, 2, 2))
    log_lambda_ei = np.ones(n_faces)
    log_lambda_ii = np.ones(n_faces)

    _, Lmn_i = angioni_sauter._calculate_Lmn(
        Kmn_e=Kmn_e,
        Kmn_i=Kmn_i,
        geo=geo,
        core_profiles=core_profiles,
        log_lambda_ei=log_lambda_ei,
        log_lambda_ii=log_lambda_ii,
    )

    np.testing.assert_allclose(Lmn_i[:, 1, 0], -Lmn_i[:, 0, 1])
    self.assertTrue(np.all(Lmn_i[:, 1, 0] != 0.0))

  def test_angioni_sauter_transport_and_shaing_blending(self):
    _, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )
    neoclassical_intermediates = formulas.compute_neoclassical_intermediates(
        geo, core_profiles
    )
    raw = angioni_sauter._calculate_angioni_sauter_transport(
        geometry=geo,
        core_profiles=core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    for field in dataclasses.fields(raw):
      arr = getattr(raw, field.name)
      self.assertTrue(np.all(np.isfinite(arr)))
      np.testing.assert_allclose(arr[0], arr[1], atol=_A_TOL, rtol=_R_TOL)

    params_no_shaing = angioni_sauter.AngioniSauterModelConfig(
        use_shaing_ion_correction=False
    ).build_runtime_params()
    res_no_shaing = angioni_sauter.AngioniSauterModel()._call_implementation(
        params_no_shaing, geo, core_profiles, neoclassical_intermediates
    )
    for field in dataclasses.fields(raw):
      np.testing.assert_allclose(
          getattr(res_no_shaing, field.name),
          getattr(raw, field.name),
          atol=_A_TOL,
          rtol=_R_TOL,
      )

    params_shaing = angioni_sauter.AngioniSauterModelConfig(
        use_shaing_ion_correction=True
    ).build_runtime_params()
    res_shaing = angioni_sauter.AngioniSauterModel()._call_implementation(
        params_shaing, geo, core_profiles, neoclassical_intermediates
    )
    shaing = angioni_sauter._calculate_shaing_transport(
        runtime_params=params_shaing,
        geometry=geo,
        core_profiles=core_profiles,
        neoclassical_intermediates=neoclassical_intermediates,
    )
    alpha = angioni_sauter._calculate_blend_alpha(
        rho_face_norm=geo.rho_face_norm,
        start=params_shaing.shaing_blend_start,
        rate=params_shaing.shaing_blend_rate,
    )
    np.testing.assert_allclose(
        res_shaing.chi_face_ion,
        (1.0 - alpha) * shaing.chi_face_ion + alpha * raw.chi_face_ion,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        res_shaing.chi_face_el, raw.chi_face_el, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        res_shaing.d_face_el, raw.d_face_el, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        res_shaing.v_face_el, raw.v_face_el, atol=_A_TOL, rtol=_R_TOL
    )
    np.testing.assert_allclose(
        res_shaing.v_face_el_ware, raw.v_face_el_ware, atol=_A_TOL, rtol=_R_TOL
    )


if __name__ == '__main__':
  absltest.main()
