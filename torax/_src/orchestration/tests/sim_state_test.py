# Copyright 2026 DeepMind Technologies Limited
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
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import sim_state
from torax._src.sources import source_profiles
from torax._src.test_utils import core_profile_helpers
from torax._src.time_step_calculator import time_step_calculator_state

# pylint: disable=invalid-name


class SimStateTest(parameterized.TestCase):

  def _make_geometry(self, **overrides) -> standard_geometry.StandardGeometry:
    defaults = dict(
        geometry_type=geometry.GeometryType.FBT,
        Ip_from_parameters=True,
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        psi=np.linspace(0, 1.0, 10),
        Ip_profile=np.linspace(0, 1e6, 10),
        Phi=np.linspace(0, 1.0, 10),
        R_in=np.linspace(4.0, 4.2, 10),
        R_out=np.linspace(8.0, 8.4, 10),
        F=np.linspace(30.0, 33.0, 10),
        int_dl_over_Bp=np.linspace(0.01, 1.0, 10),
        flux_surf_avg_1_over_R=np.linspace(0.1, 0.2, 10),
        flux_surf_avg_1_over_R2=np.linspace(0.01, 0.04, 10),
        flux_surf_avg_grad_psi2=np.linspace(0.01, 1.0, 10),
        flux_surf_avg_grad_psi=np.linspace(0.01, 1.0, 10),
        flux_surf_avg_grad_psi2_over_R2=np.linspace(0.01, 1.0, 10),
        flux_surf_avg_B2=np.linspace(25.0, 30.0, 10),
        flux_surf_avg_1_over_B2=np.linspace(0.03, 0.04, 10),
        trapped_fraction=np.linspace(0.0, 0.5, 10),
        delta_upper_face=np.linspace(0.0, 0.3, 10),
        delta_lower_face=np.linspace(0.0, 0.3, 10),
        elongation=np.linspace(1.0, 1.7, 10),
        vpr=np.linspace(0.01, 1.0, 10),
        face_centers=np.linspace(0, 1.0, 5),
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
    defaults.update(overrides)
    intermediates = standard_geometry.StandardGeometryIntermediates(**defaults)  # pyrefly: ignore[bad-argument-type]
    return standard_geometry.build_standard_geometry(intermediates)

  def _make_sim_state(self, geo: geometry.Geometry) -> sim_state.SimState:
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    core_transport = state.CoreTransport.zeros(geo)

    sp = source_profiles.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(geo),
        qei=source_profiles.QeiInfo.zeros(geo),
    )

    return sim_state.SimState(
        t=jnp.array(0.0),
        dt=jnp.array(0.1),
        core_profiles=core_profiles,
        core_transport=core_transport,
        core_sources=sp,
        edge_outputs=None,
        geometry=geo,
        solver_numeric_outputs=state.SolverNumericOutputs(
            outer_solver_iterations=1,
            solver_error_state=1,
            inner_solver_iterations=1,
            sawtooth_crash=False,
        ),
        time_step_calculator_state=time_step_calculator_state.TimeStepCalculatorState(),
    )

  def test_has_nan_no_nan(self):
    geo = self._make_geometry()
    s = self._make_sim_state(geo)
    self.assertFalse(s.has_nan())

  def test_has_nan_detects_nan_in_core_profiles(self):
    geo = self._make_geometry()
    s = self._make_sim_state(geo)
    T_e = s.core_profiles.T_e
    new_value = T_e.value.at[0].set(jnp.nan)
    new_T_e = dataclasses.replace(T_e, value=new_value)
    new_core_profiles = dataclasses.replace(s.core_profiles, T_e=new_T_e)
    s = dataclasses.replace(s, core_profiles=new_core_profiles)
    self.assertTrue(s.has_nan())

  def test_has_nan_ignores_geometry_optional_inputs(self):
    geo = self._make_geometry(
        connection_length_target=np.nan,
        connection_length_divertor=np.nan,
        angle_of_incidence_target=np.nan,
        R_OMP=np.nan,
        R_target=np.nan,
        B_pol_OMP=np.nan,
    )
    s = self._make_sim_state(geo)
    self.assertFalse(s.has_nan())

  def test_has_nan_ignores_z_magnetic_axis(self):
    geo = self._make_geometry(z_magnetic_axis=np.nan)
    s = self._make_sim_state(geo)
    self.assertFalse(s.has_nan())


if __name__ == '__main__':
  absltest.main()
