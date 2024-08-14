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

"""Unit tests for torax.config.runtime_params_slice."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax import geometry
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.sources import electron_density_sources
from torax.sources import external_current_source
from torax.sources import formula_config
from torax.sources import runtime_params as sources_params_lib
from torax.stepper import runtime_params as stepper_params_lib
from torax.transport_model import runtime_params as transport_params_lib


class RuntimeParamsSliceTest(parameterized.TestCase):
  """Unit tests for the `runtime_params_slice` module."""

  def setUp(self):
    super().setUp()
    self._geo = geometry.build_circular_geometry()

  def test_dynamic_slice_can_be_input_to_jitted_function(self):
    """Tests that the slice can be input to a jitted function."""

    def foo(
        runtime_params_slice: runtime_params_slice_lib.DynamicRuntimeParamsSlice,
    ):
      _ = runtime_params_slice  # do nothing.

    foo_jitted = jax.jit(foo)
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    runtime_params_provider = runtime_params.make_provider(self._geo.torax_mesh)
    dynamic_slice = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params_provider, geo=self._geo,
    )
    # Make sure you can call the function with dynamic_slice as an arg.
    foo_jitted(dynamic_slice)

  def test_time_dependent_provider_is_time_dependent(self):
    """Tests that the runtime_params slice provider is time dependent."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti_bound_right={0.0: 2.0, 4.0: 4.0},
        ),
    )
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {},
        stepper_getter=stepper_params_lib.RuntimeParams,
        torax_mesh=self._geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(t=1.0, geo=self._geo)
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ti_bound_right, 2.5
    )
    dynamic_runtime_params_slice = provider(t=2.0, geo=self._geo)
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ti_bound_right, 3.0
    )

  def test_boundary_conditions_are_time_dependent(self):
    """Tests that the boundary conditions are time dependent params."""
    # All of the following parameters are time-dependent fields, but they can
    # be initialized in different ways.
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti_bound_right={0.0: 2.0, 4.0: 4.0},
            Te_bound_right=4.5,  # not time-dependent.
            ne_bound_right=({5.0: 6.0, 7.0: 8.0}, 'step'),
        ),
    )
    runtime_params_provider = runtime_params.make_provider(self._geo.torax_mesh)
    np.testing.assert_allclose(
        runtime_params_slice_lib.build_dynamic_runtime_params_slice(
            runtime_params_provider, t=2.0, geo=self._geo,
        ).profile_conditions.Ti_bound_right,
        3.0,
    )
    np.testing.assert_allclose(
        runtime_params_slice_lib.build_dynamic_runtime_params_slice(
            runtime_params_provider, t=4.0, geo=self._geo,
        ).profile_conditions.Te_bound_right,
        4.5,
    )
    np.testing.assert_allclose(
        runtime_params_slice_lib.build_dynamic_runtime_params_slice(
            runtime_params_provider,
            t=6.0,
            geo=self._geo,
        ).profile_conditions.ne_bound_right,
        6.0,
    )

  def test_pedestal_is_time_dependent(self):
    """Tests that the pedestal runtime params are time dependent."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal={0.0: True, 1.0: False},
            Tiped={0.0: 0.0, 1.0: 1.0},
            Teped={0.0: 1.0, 1.0: 2.0},
            neped={0.0: 2.0, 1.0: 3.0},
            Ped_top={0.0: 3.0, 1.0: 5.0},
        ),
    )
    runtime_params_provider = runtime_params.make_provider(self._geo.torax_mesh)
    # Check at time 0.
    dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params_provider, t=0.0, geo=self._geo,
    )
    profile_conditions = dcs.profile_conditions
    np.testing.assert_allclose(profile_conditions.set_pedestal, True)
    np.testing.assert_allclose(profile_conditions.Tiped, 0.0)
    np.testing.assert_allclose(profile_conditions.Teped, 1.0)
    np.testing.assert_allclose(profile_conditions.neped, 2.0)
    np.testing.assert_allclose(profile_conditions.Ped_top, 3.0)
    # And check after the time limit.
    dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params_provider, t=1.0, geo=self._geo,
    )
    profile_conditions = dcs.profile_conditions
    np.testing.assert_allclose(profile_conditions.set_pedestal, False)
    np.testing.assert_allclose(profile_conditions.Tiped, 1.0)
    np.testing.assert_allclose(profile_conditions.Teped, 2.0)
    np.testing.assert_allclose(profile_conditions.neped, 3.0)
    np.testing.assert_allclose(profile_conditions.Ped_top, 5.0)

  def test_source_formula_config_has_time_dependent_params(self):
    """Tests that the source formula config params are time-dependent."""
    with self.subTest('default_ne_sources'):
      # Check that the runtime params for the default ne sources are
      # time-dependent.
      runtime_params = general_runtime_params.GeneralRuntimeParams()
      runtime_params_provider = runtime_params.make_provider(
          self._geo.torax_mesh
      )
      dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
          runtime_params=runtime_params_provider,
          sources={
              'gas_puff_source': electron_density_sources.GasPuffRuntimeParams(
                  puff_decay_length={0.0: 0.0, 1.0: 4.0},
                  S_puff_tot={0.0: 0.0, 1.0: 5.0},
              ),
              'pellet_source': electron_density_sources.PelletRuntimeParams(
                  pellet_width={0.0: 0.0, 1.0: 1.0},
                  pellet_deposition_location={0.0: 0.0, 1.0: 2.0},
                  S_pellet_tot={0.0: 0.0, 1.0: 3.0},
              ),
              'nbi_particle_source': (
                  electron_density_sources.NBIParticleRuntimeParams(
                      nbi_particle_width={0.0: 0.0, 1.0: 6.0},
                      nbi_deposition_location={0.0: 0.0, 1.0: 7.0},
                      S_nbi_tot={0.0: 0.0, 1.0: 8.0},
                  )
              ),
          },
          t=0.5,
          geo=self._geo,
      )
      pellet_source = dcs.sources['pellet_source']
      gas_puff_source = dcs.sources['gas_puff_source']
      nbi_particle_source = dcs.sources['nbi_particle_source']
      assert isinstance(
          pellet_source,
          electron_density_sources.DynamicPelletRuntimeParams,
      )
      assert isinstance(
          gas_puff_source,
          electron_density_sources.DynamicGasPuffRuntimeParams,
      )
      assert isinstance(
          nbi_particle_source,
          electron_density_sources.DynamicNBIParticleRuntimeParams,
      )
      print(pellet_source.pellet_width)
      print(type(pellet_source.pellet_width))
      np.testing.assert_allclose(pellet_source.pellet_width, 0.5)
      np.testing.assert_allclose(pellet_source.pellet_deposition_location, 1.0)
      np.testing.assert_allclose(pellet_source.S_pellet_tot, 1.5)
      np.testing.assert_allclose(gas_puff_source.puff_decay_length, 2.0)
      np.testing.assert_allclose(gas_puff_source.S_puff_tot, 2.5)
      np.testing.assert_allclose(nbi_particle_source.nbi_particle_width, 3.0)
      np.testing.assert_allclose(
          nbi_particle_source.nbi_deposition_location, 3.5
      )
      np.testing.assert_allclose(nbi_particle_source.S_nbi_tot, 4.0)

    with self.subTest('exponential_formula'):
      runtime_params = general_runtime_params.GeneralRuntimeParams()
      runtime_params_provider = runtime_params.make_provider(
          self._geo.torax_mesh
      )
      dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
          runtime_params=runtime_params_provider,
          sources={
              'gas_puff_source': sources_params_lib.RuntimeParams(
                  formula=formula_config.Exponential(
                      total={0.0: 0.0, 1.0: 1.0},
                      c1={0.0: 0.0, 1.0: 2.0},
                      c2={0.0: 0.0, 1.0: 3.0},
                  )
              ),
          },
          t=0.25,
          geo=self._geo,
      )
      gas_puff_source = dcs.sources['gas_puff_source']
      assert isinstance(
          gas_puff_source.formula,
          formula_config.DynamicExponential,
      )
      np.testing.assert_allclose(
          gas_puff_source.formula.total, 0.25
      )
      np.testing.assert_allclose(gas_puff_source.formula.c1, 0.5)
      np.testing.assert_allclose(
          gas_puff_source.formula.c2, 0.75
      )

    with self.subTest('gaussian_formula'):
      runtime_params = general_runtime_params.GeneralRuntimeParams()
      runtime_params_provider = runtime_params.make_provider(
          self._geo.torax_mesh
      )
      dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
          runtime_params=runtime_params_provider,
          sources={
              'gas_puff_source': sources_params_lib.RuntimeParams(
                  formula=formula_config.Gaussian(
                      total={0.0: 0.0, 1.0: 1.0},
                      c1={0.0: 0.0, 1.0: 2.0},
                      c2={0.0: 0.0, 1.0: 3.0},
                  )
              ),
          },
          t=0.25,
          geo=self._geo,
      )
      gas_puff_source = dcs.sources['gas_puff_source']
      assert isinstance(
          gas_puff_source.formula, formula_config.DynamicGaussian
      )
      np.testing.assert_allclose(
          gas_puff_source.formula.total, 0.25
      )
      np.testing.assert_allclose(gas_puff_source.formula.c1, 0.5)
      np.testing.assert_allclose(
          gas_puff_source.formula.c2, 0.75
      )

  def test_wext_in_dynamic_runtime_params_cannot_be_negative(self):
    """Tests that wext cannot be negative."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dcs_provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport_getter=transport_params_lib.RuntimeParams,
        sources_getter=lambda: {
            'jext': external_current_source.RuntimeParams(
                wext={0.0: 1.0, 1.0: -1.0}
            ),
        },
        stepper_getter=stepper_params_lib.RuntimeParams,
        torax_mesh=self._geo.torax_mesh,
    )
    # While wext is positive, this should be fine.
    dcs = dcs_provider(t=0.0, geo=self._geo,)
    jext = dcs.sources['jext']
    assert isinstance(
        jext, external_current_source.DynamicRuntimeParams
    )
    np.testing.assert_allclose(jext.wext, 1.0)
    # Even 0 should be fine.
    dcs = dcs_provider(t=0.5, geo=self._geo,)
    jext = dcs.sources['jext']
    assert isinstance(
        jext, external_current_source.DynamicRuntimeParams
    )
    np.testing.assert_allclose(jext.wext, 0.0)
    # But negative values will cause an error.
    with self.assertRaises(jax.lib.xla_client.XlaRuntimeError):
      dcs_provider(t=1.0, geo=self._geo,)

  @parameterized.parameters(
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'Ti',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'Ti',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'Te',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'Te',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          None,
          np.array([1.125, 1.375, 1.625, 1.875]),
          2.0,
          'ne',
      ),
      (
          {0: {0.0: 1.0, 1.0: 2.0}},
          3.0,
          np.array([1.125, 1.375, 1.625, 1.875]),
          3.0,
          'ne',
      ),
  )
  def test_profile_conditions_set_electron_temperature_and_boundary_condition(
      self,
      var,
      var_boundary_condition,
      expected_var,
      expected_var_boundary_condition,
      var_name,
  ):
    """Tests that the profile conditions can set the electron temperature."""
    profile_conditions = profile_conditions_lib.ProfileConditions()
    boundary_var_name = var_name + '_bound_right'
    temperatures = {
        var_name: var,
        boundary_var_name: var_boundary_condition,
    }
    profile_conditions = dataclasses.replace(profile_conditions, **temperatures)
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions,
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    runtime_params_provider = runtime_params.make_provider(geo.torax_mesh)
    dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params=runtime_params_provider,
        t=0.0,
        geo=geo,
    )
    np.testing.assert_allclose(
        getattr(dcs.profile_conditions, var_name), expected_var
    )
    self.assertEqual(
        getattr(dcs.profile_conditions, boundary_var_name),
        expected_var_boundary_condition,
    )

  @parameterized.product(
      ne_bound_right=[None, 1.0,],
      ne_bound_right_is_fGW=[True, False,],
      ne_is_fGW=[True, False,],
  )
  def test_profile_conditions_set_electron_density_and_boundary_condition(
      self,
      ne_bound_right,
      ne_bound_right_is_fGW,  # pylint: disable=invalid-name
      ne_is_fGW,  # pylint: disable=invalid-name
  ):
    """Tests that the profile conditions can set the electron temperature."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne_bound_right=ne_bound_right,
            ne_bound_right_is_fGW=ne_bound_right_is_fGW,
            ne_is_fGW=ne_is_fGW,
        ),
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    runtime_params_provider = runtime_params.make_provider(self._geo.torax_mesh)

    dcs = runtime_params_slice_lib.build_dynamic_runtime_params_slice(
        runtime_params=runtime_params_provider,
        t=0.0,
        geo=geo,
    )

    if ne_bound_right is None:
      # If the boundary condition was not set, it should inherit the fGW flag.
      self.assertEqual(
          dcs.profile_conditions.ne_bound_right_is_fGW,
          ne_is_fGW,
      )
      # If the boundary condition was set check it is not absolute.
      self.assertFalse(dcs.profile_conditions.ne_bound_right_is_absolute)
    else:
      self.assertEqual(
          dcs.profile_conditions.ne_bound_right_is_fGW,
          ne_bound_right_is_fGW,
      )
      self.assertTrue(dcs.profile_conditions.ne_bound_right_is_absolute)


if __name__ == '__main__':
  absltest.main()
