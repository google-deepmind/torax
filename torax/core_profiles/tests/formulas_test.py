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
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax import state
from torax.config import runtime_params as runtime_params_lib
from torax.config import runtime_params_slice
from torax.core_profiles import formulas
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.tests.test_lib import default_sources
from torax.tests.test_lib import torax_refs


class FormulasTest(parameterized.TestCase):
  """Unit tests for the `core_profiles.formulas` module."""

  def setUp(self):
    super().setUp()
    runtime_params = runtime_params_lib.GeneralRuntimeParams()
    self.geo = geometry_pydantic_model.CircularConfig().build_geometry()
    geo_provider = geometry_provider.ConstantGeometryProvider(self.geo)
    source_models_builder = default_sources.get_default_sources_builder()
    source_models = source_models_builder()
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=source_models_builder.runtime_params,
        )
    )
    static_slice = runtime_params_slice.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    self.core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

  def test_compute_pressure(self):
    """Test that pressure is computed correctly."""

    def _make_constant_core_profile(
        value: float,
    ) -> cell_variable.CellVariable:
      return cell_variable.CellVariable(
          value=value * np.ones_like(self.geo.rho_norm),
          left_face_grad_constraint=np.zeros(()),
          left_face_constraint=None,
          right_face_grad_constraint=None,
          right_face_constraint=jax.numpy.array(value),
          dr=self.geo.drho_norm,
      )

    # Override all densities and temperatures to constant values
    core_profiles = dataclasses.replace(
        self.core_profiles,
        temp_ion=_make_constant_core_profile(1.0),
        temp_el=_make_constant_core_profile(2.0),
        ne=_make_constant_core_profile(3.0),
        ni=_make_constant_core_profile(2.5),
        nimp=_make_constant_core_profile(0.25),
    )
    # pylint: disable=protected-access
    p_el, p_ion, p_tot = formulas.compute_pressure(core_profiles)
    # pylint: enable=protected-access
    # Make sure that we are grabbing the values from the face grid.
    self.assertEqual(p_el.shape, self.geo.rho_face.shape)
    # Ignore boundary condition terms and just check formula sanity.
    np.testing.assert_allclose(
        p_el, 6 * constants.CONSTANTS.keV2J * core_profiles.nref
    )
    np.testing.assert_allclose(
        p_ion,
        2.75 * constants.CONSTANTS.keV2J * core_profiles.nref,
    )
    np.testing.assert_allclose(
        p_tot,
        8.75 * constants.CONSTANTS.keV2J * core_profiles.nref,
    )

  def test_compute_stored_thermal_energy(self):
    """Test that stored thermal energy is computed correctly."""
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    p_el = np.ones_like(geo.rho_face)
    p_ion = 2 * np.ones_like(geo.rho_face)
    p_tot = p_el + p_ion
    # pylint: disable=protected-access
    wth_el, wth_ion, wth_tot = formulas.compute_stored_thermal_energy(
        p_el, p_ion, p_tot, geo
    )
    # pylint: enable=protected-access

    volume = np.trapezoid(geo.vpr_face, geo.rho_face_norm)

    np.testing.assert_allclose(wth_el, 1.5 * p_el[0] * volume)
    np.testing.assert_allclose(wth_ion, 1.5 * p_ion[0] * volume)
    np.testing.assert_allclose(wth_tot, 1.5 * p_tot[0] * volume)

  def test_calculate_greenwald_fraction(self):
    """Test that Greenwald fraction is calculated correctly."""
    ne = 1.0

    core_profiles = mock.create_autospec(
        state.CoreProfiles,
        instance=True,
        nref=1e20,
        currents=mock.create_autospec(
            state.Currents,
            instance=True,
            Ip_total=np.pi * 1e6,
        ),
    )
    geo = mock.create_autospec(
        geometry.Geometry,
        instance=True,
        Rmin=1.0,
    )

    # pylint: disable=protected-access
    fgw_ne_volume_avg_calculated = formulas.calculate_greenwald_fraction(
        ne, core_profiles, geo
    )

    fgw_ne_volume_avg_expected = 1.0

    np.testing.assert_allclose(
        fgw_ne_volume_avg_calculated, fgw_ne_volume_avg_expected
    )
    # pylint: enable=protected-access

if __name__ == '__main__':
  absltest.main()
