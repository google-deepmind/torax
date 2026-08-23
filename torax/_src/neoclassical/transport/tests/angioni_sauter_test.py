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
from torax._src.neoclassical.transport import base
from torax._src.torax_pydantic import model_config

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

  def test_angioni_sauter_against_reference_values(self):
    """Reference values generated from running Angioni-Sauter."""
    runtime_params, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )

    f_trap = formulas.calculate_f_trap(
        geo,
        runtime_params.neoclassical.f_trap_model,
        q_face=core_profiles.q_face,
    )
    ion_species = formulas.build_ion_species_from_core_profiles(
        core_profiles, subtract_fast_ions=True
    )
    Z_eff_face = formulas.calculate_Z_eff_from_ion_species(
        core_profiles, ion_species
    )
    dens_sum_face = formulas.calculate_ion_density_sum_face(
        ion_species, placeholder=core_profiles.n_e.face_value()
    )
    # Without fast ions, species-summed Z_eff matches bundled Z_eff_face.
    np.testing.assert_allclose(
        Z_eff_face, core_profiles.Z_eff_face, atol=_A_TOL, rtol=_R_TOL
    )

    # Test raw Angioni-Sauter values
    result = angioni_sauter._calculate_angioni_sauter_transport(
        runtime_params, geo, core_profiles, f_trap, Z_eff_face, dens_sum_face
    )
    np.testing.assert_allclose(
        result.chi_neo_i,
        _ANGIONI_SAUTER_REFERENCE_VALUES.chi_neo_i,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.chi_neo_e,
        _ANGIONI_SAUTER_REFERENCE_VALUES.chi_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.D_neo_e,
        _ANGIONI_SAUTER_REFERENCE_VALUES.D_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.V_neo_e,
        _ANGIONI_SAUTER_REFERENCE_VALUES.V_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.V_neo_ware_e,
        _ANGIONI_SAUTER_REFERENCE_VALUES.V_neo_ware_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )

  def test_angioni_sauter_with_shaing_against_reference_values(self):
    """Reference values generated from Angioni-Sauter + Shaing ion correction."""
    runtime_params, geo, core_profiles = (
        self._get_reference_runtime_params_geo_and_core_profiles()
    )

    # Modify runtime params to include settings for Shaing
    neoclassical_runtime_params = runtime_params.neoclassical
    neoclassical_runtime_params = dataclasses.replace(
        neoclassical_runtime_params,
        transport=angioni_sauter.AngioniSauterModelConfig(
            use_shaing_ion_correction=True
        ).build_runtime_params(),
    )
    modified_runtime_params = dataclasses.replace(
        runtime_params,
        neoclassical=neoclassical_runtime_params,
    )

    # Test blended Angioni-Sauter + Shaing values
    result = angioni_sauter.AngioniSauterModel()._call_implementation(
        modified_runtime_params, geo, core_profiles
    )
    np.testing.assert_allclose(
        result.chi_neo_i,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.chi_neo_i,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.chi_neo_e,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.chi_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.D_neo_e,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.D_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.V_neo_e,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.V_neo_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )
    np.testing.assert_allclose(
        result.V_neo_ware_e,
        _ANGIONI_SAUTER_SHAING_REFERENCE_VALUES.V_neo_ware_e,
        atol=_A_TOL,
        rtol=_R_TOL,
    )


# Reference values from running test code in a standalone manner.
# The test thus does not directly test the implementation, but rather
# guards against unexpected modifications.
#
# The implementation was independently tested against NEOS up to the
# generation of the Kmn matrix.
_ANGIONI_SAUTER_REFERENCE_VALUES = base.NeoclassicalTransport(
    chi_neo_i=np.array([
        0.012331097,
        0.012331097,
        0.022491648,
        0.031540901,
        0.039378489,
        0.046231733,
        0.052401891,
        0.057866989,
        0.062174428,
        0.063889172,
        0.059739414,
    ]),
    chi_neo_e=np.array([
        -0.002100234,
        -0.002100234,
        -0.003079202,
        -0.003886831,
        -0.004554799,
        -0.005110684,
        -0.005608297,
        -0.006088400,
        -0.006581468,
        -0.007173669,
        -0.007503227,
    ]),
    D_neo_e=np.array([
        0.000116982,
        0.000116982,
        0.000211052,
        0.000284744,
        0.000337209,
        0.000375292,
        0.000403767,
        0.000421994,
        0.000424042,
        0.000392922,
        0.000292399,
    ]),
    V_neo_e=np.array([
        1.079514416e-05,
        1.079514416e-05,
        1.110150038e-05,
        1.540657517e-05,
        2.657106723e-05,
        4.428537522e-05,
        7.063873825e-05,
        1.129832698e-04,
        1.923600656e-04,
        3.863721278e-04,
        1.188686269e-03,
    ]),
    V_neo_ware_e=np.array([
        -0.000381139,
        -0.000381139,
        -0.000417593,
        -0.000371229,
        -0.000320656,
        -0.000306465,
        -0.000331205,
        -0.000385647,
        -0.000562293,
        -0.001598163,
        -0.001789132,
    ]),
)

# Shaing correction only affects ions, so we can reuse the other values
_ANGIONI_SAUTER_SHAING_REFERENCE_VALUES = base.NeoclassicalTransport(
    chi_neo_i=np.array([
        0.203844093,
        0.171337483,
        0.030447525,
        0.026206587,
        0.035643266,
        0.044430553,
        0.051634606,
        0.057565066,
        0.062066696,
        0.063858183,
        0.059733278,
    ]),
    chi_neo_e=_ANGIONI_SAUTER_REFERENCE_VALUES.chi_neo_e,
    D_neo_e=_ANGIONI_SAUTER_REFERENCE_VALUES.D_neo_e,
    V_neo_e=_ANGIONI_SAUTER_REFERENCE_VALUES.V_neo_e,
    V_neo_ware_e=_ANGIONI_SAUTER_REFERENCE_VALUES.V_neo_ware_e,
)

if __name__ == '__main__':
  absltest.main()
