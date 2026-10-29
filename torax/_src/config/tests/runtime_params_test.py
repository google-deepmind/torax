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
from absl.testing import parameterized
import jax
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import runtime_params as core_profiles_params
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
from torax._src.torax_pydantic import interpolated_param_2d


def _create_dummy_standard_geometry() -> standard_geometry.StandardGeometry:
  """Creates a dummy StandardGeometry in memory without loading any files."""
  intermediate = standard_geometry.StandardGeometryIntermediates(
      geometry_type=geometry.GeometryType.FBT,
      R_major=6.2,
      a_minor=2.0,
      B_0=5.3,
      psi=np.arange(0, 1.0, 0.01),
      I_p_profile=np.linspace(0.0, 15e6, 100),
      Phi=np.arange(0, 1.0, 0.01),
      R_in=np.arange(1, 2, 0.01),
      R_out=np.arange(1, 2, 0.01),
      F=np.arange(0, 1.0, 0.01),
      int_dl_over_Bp=np.arange(0, 1.0, 0.01),
      flux_surf_avg_1_over_R=np.arange(0, 1.0, 0.01),
      flux_surf_avg_1_over_R2=np.arange(0, 1.0, 0.01),
      flux_surf_avg_grad_psi2=np.arange(0, 1.0, 0.01),
      flux_surf_avg_grad_psi=np.arange(0, 1.0, 0.01),
      flux_surf_avg_grad_psi2_over_R2=np.arange(0, 1.0, 0.01),
      flux_surf_avg_B2=np.arange(0, 1.0, 0.01),
      flux_surf_avg_1_over_B2=np.arange(0, 1.0, 0.01),
      trapped_fraction=np.arange(0, 1.0, 0.01),
      delta_upper_face=np.arange(0, 1.0, 0.01),
      delta_lower_face=np.arange(0, 1.0, 0.01),
      elongation=np.arange(0, 1.0, 0.01),
      vpr=np.arange(0, 1.0, 0.01),
      face_centers=interpolated_param_2d.get_face_centers(25),
      hires_factor=4,
      z_magnetic_axis=np.array(0.0),
      diverted=None,
      connection_length_target=None,
      connection_length_divertor=None,
      angle_of_incidence_target=None,
      R_OMP=None,
      R_target=None,
      B_pol_OMP=None,
  )
  return standard_geometry.build_standard_geometry(intermediate)


class RuntimeParamsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._torax_config = config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    self._torax_mesh = self._torax_config.geometry.build_provider.torax_mesh

  def test_runtime_params_can_be_input_to_jitted_function(self):
    """Tests that the params can be input to a jitted function."""

    def foo(params: runtime_params_lib.RuntimeParams):
      _ = params  # do nothing.

    foo_jitted = jax.jit(foo)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        self._torax_config
    )(
        t=self._torax_config.time_step_calculator.t_initial,
    )
    # Make sure you can call the function with runtime_params as an arg.
    foo_jitted(runtime_params)

  def test_make_i_p_consistent_from_profile_conditions(self):
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['I_p'] = 12e6
    config_dict['profile_conditions']['I_p_source'] = 'profile_conditions'
    torax_config = config.ToraxConfig.from_dict(config_dict)
    raw_runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.time_step_calculator.t_initial)
    raw_geo = _create_dummy_standard_geometry()
    runtime_params, geo = runtime_params_lib.make_I_p_consistent(
        raw_runtime_params, raw_geo
    )
    assert isinstance(geo, standard_geometry.StandardGeometry)
    self.assertEqual(
        runtime_params.profile_conditions.I_p_source,
        core_profiles_params.IpSource.PROFILE_CONDITIONS,
    )
    np.testing.assert_allclose(geo.I_p_profile_face[-1], 12e6)
    expected_scale = 12e6 / raw_geo.I_p_profile_face[-1]
    np.testing.assert_allclose(
        geo.psi_from_I_p, raw_geo.psi_from_I_p * expected_scale
    )

  def test_make_i_p_consistent_from_geometry(self):
    raw_runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        self._torax_config
    )(t=self._torax_config.time_step_calculator.t_initial)
    raw_runtime_params = dataclasses.replace(
        raw_runtime_params,
        profile_conditions=dataclasses.replace(
            raw_runtime_params.profile_conditions,
            I_p_source=core_profiles_params.IpSource.GEOMETRY,
        ),
    )
    raw_geo = _create_dummy_standard_geometry()
    runtime_params, geo = runtime_params_lib.make_I_p_consistent(
        raw_runtime_params, raw_geo
    )
    assert isinstance(geo, standard_geometry.StandardGeometry)
    self.assertEqual(
        runtime_params.profile_conditions.I_p_source,
        core_profiles_params.IpSource.GEOMETRY,
    )
    np.testing.assert_allclose(
        runtime_params.profile_conditions.I_p, raw_geo.I_p_profile_face[-1]
    )
    np.testing.assert_allclose(
        geo.I_p_profile_face[-1], raw_geo.I_p_profile_face[-1]
    )


if __name__ == '__main__':
  absltest.main()
