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
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice as runtime_params_slice_lib
from torax.geometry import geometry
from torax.pedestal_model import set_tped_nped
from torax.sources import electron_density_sources
from torax.sources import generic_current_source
from torax.stepper import runtime_params as stepper_params_lib
from torax.tests.test_lib import default_sources
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
    dynamic_slice = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )(
        t=runtime_params.numerics.t_initial,
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
        transport=transport_params_lib.RuntimeParams(),
        sources={},
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self._geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )
    np.testing.assert_allclose(
        dynamic_runtime_params_slice.profile_conditions.Ti_bound_right, 2.5
    )
    dynamic_runtime_params_slice = provider(
        t=2.0,
    )
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
    np.testing.assert_allclose(
        runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            torax_mesh=self._geo.torax_mesh,
        )(
            t=2.0,
        ).profile_conditions.Ti_bound_right,
        3.0,
    )
    np.testing.assert_allclose(
        runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            torax_mesh=self._geo.torax_mesh,
        )(
            t=4.0,
        ).profile_conditions.Te_bound_right,
        4.5,
    )
    np.testing.assert_allclose(
        runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            torax_mesh=self._geo.torax_mesh,
        )(
            t=6.0,
        ).profile_conditions.ne_bound_right,
        6.0,
    )

  def test_pedestal_is_time_dependent(self):
    """Tests that the pedestal runtime params are time dependent."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal={0.0: True, 1.0: False},
        )
    )
    pedestal_runtime_params = set_tped_nped.RuntimeParams(
        Tiped={0.0: 0.0, 1.0: 1.0},
        Teped={0.0: 1.0, 1.0: 2.0},
        neped={0.0: 2.0, 1.0: 3.0},
        rho_norm_ped_top={0.0: 3.0, 1.0: 5.0},
    )
    # Check at time 0.
    dcs_provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params,
        pedestal=pedestal_runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )

    dcs = dcs_provider(
        t=0.0,
    )
    profile_conditions = dcs.profile_conditions
    dynamic_pedestal_runtime_params = dcs.pedestal
    assert isinstance(
        dynamic_pedestal_runtime_params,
        set_tped_nped.DynamicRuntimeParams,
    )
    np.testing.assert_allclose(profile_conditions.set_pedestal, True)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.Tiped, 0.0)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.Teped, 1.0)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.neped, 2.0)
    np.testing.assert_allclose(
        dynamic_pedestal_runtime_params.rho_norm_ped_top, 3.0
    )
    # And check after the time limit.
    dcs = dcs_provider(
        t=1.0,
    )
    profile_conditions = dcs.profile_conditions
    dynamic_pedestal_runtime_params = dcs.pedestal
    assert isinstance(
        dynamic_pedestal_runtime_params,
        set_tped_nped.DynamicRuntimeParams,
    )
    np.testing.assert_allclose(profile_conditions.set_pedestal, False)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.Tiped, 1.0)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.Teped, 2.0)
    np.testing.assert_allclose(dynamic_pedestal_runtime_params.neped, 3.0)
    np.testing.assert_allclose(
        dynamic_pedestal_runtime_params.rho_norm_ped_top, 5.0
    )

  def test_source_formula_config_has_time_dependent_params(self):
    """Tests that the source formula config params are time-dependent."""
    with self.subTest('default_ne_sources'):
      # Check that the runtime params for the default ne sources are
      # time-dependent.
      runtime_params = general_runtime_params.GeneralRuntimeParams()
      dcs = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          sources={
              electron_density_sources.GasPuffSource.SOURCE_NAME: (
                  electron_density_sources.GasPuffRuntimeParams(
                      puff_decay_length={0.0: 0.0, 1.0: 4.0},
                      S_puff_tot={0.0: 0.0, 1.0: 5.0},
                  )
              ),
              electron_density_sources.PelletSource.SOURCE_NAME: (
                  electron_density_sources.PelletRuntimeParams(
                      pellet_width={0.0: 0.0, 1.0: 1.0},
                      pellet_deposition_location={0.0: 0.0, 1.0: 2.0},
                      S_pellet_tot={0.0: 0.0, 1.0: 3.0},
                  )
              ),
              electron_density_sources.GenericParticleSource.SOURCE_NAME: (
                  electron_density_sources.GenericParticleSourceRuntimeParams(
                      particle_width={0.0: 0.0, 1.0: 6.0},
                      deposition_location={0.0: 0.0, 1.0: 7.0},
                      S_tot={0.0: 0.0, 1.0: 8.0},
                  )
              ),
          },
          torax_mesh=self._geo.torax_mesh,
      )(
          t=0.5,
      )
      pellet_source = dcs.sources[
          electron_density_sources.PelletSource.SOURCE_NAME
      ]
      gas_puff_source = dcs.sources[
          electron_density_sources.GasPuffSource.SOURCE_NAME
      ]
      generic_particle_source = dcs.sources[
          electron_density_sources.GenericParticleSource.SOURCE_NAME
      ]
      assert isinstance(
          pellet_source,
          electron_density_sources.DynamicPelletRuntimeParams,
      )
      assert isinstance(
          gas_puff_source,
          electron_density_sources.DynamicGasPuffRuntimeParams,
      )
      assert isinstance(
          generic_particle_source,
          electron_density_sources.DynamicParticleRuntimeParams,
      )
      print(pellet_source.pellet_width)
      print(type(pellet_source.pellet_width))
      np.testing.assert_allclose(pellet_source.pellet_width, 0.5)
      np.testing.assert_allclose(pellet_source.pellet_deposition_location, 1.0)
      np.testing.assert_allclose(pellet_source.S_pellet_tot, 1.5)
      np.testing.assert_allclose(gas_puff_source.puff_decay_length, 2.0)
      np.testing.assert_allclose(gas_puff_source.S_puff_tot, 2.5)
      np.testing.assert_allclose(generic_particle_source.particle_width, 3.0)
      np.testing.assert_allclose(
          generic_particle_source.deposition_location, 3.5
      )
      np.testing.assert_allclose(generic_particle_source.S_tot, 4.0)

  def test_wext_in_dynamic_runtime_params_cannot_be_negative(self):
    """Tests that wext cannot be negative."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    dcs_provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources={
            generic_current_source.GenericCurrentSource.SOURCE_NAME: (
                generic_current_source.RuntimeParams(wext={0.0: 1.0, 1.0: -1.0})
            ),
        },
        stepper=stepper_params_lib.RuntimeParams(),
        torax_mesh=self._geo.torax_mesh,
    )
    # While wext is positive, this should be fine.
    dcs = dcs_provider(
        t=0.0,
    )
    generic_current = dcs.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    assert isinstance(
        generic_current, generic_current_source.DynamicRuntimeParams
    )
    np.testing.assert_allclose(generic_current.wext, 1.0)
    # Even 0 should be fine.
    dcs = dcs_provider(
        t=0.5,
    )
    generic_current = dcs.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    assert isinstance(
        generic_current, generic_current_source.DynamicRuntimeParams
    )
    np.testing.assert_allclose(generic_current.wext, 0.0)
    # But negative values will cause an error.
    with self.assertRaises(RuntimeError):
      dcs_provider(
          t=1.0,
      )

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
    dcs = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
    )(
        t=0.0,
    )
    np.testing.assert_allclose(
        getattr(dcs.profile_conditions, var_name), expected_var
    )
    self.assertEqual(
        getattr(dcs.profile_conditions, boundary_var_name),
        expected_var_boundary_condition,
    )

  @parameterized.product(
      ne_bound_right=[
          None,
          1.0,
      ],
      ne_bound_right_is_fGW=[
          True,
          False,
      ],
      ne_is_fGW=[
          True,
          False,
      ],
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

    dcs = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
    )(
        t=0.0,
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

  def test_update_dynamic_slice_provider_updates_runtime_params(
      self,
  ):
    """Tests that the dynamic slice provider can be updated."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti_bound_right={0.0: 1.0, 1.0: 2.0},
        ),
    )
    geo = geometry.build_circular_geometry(n_rho=4)
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.profile_conditions.Ti_bound_right, 1.0)

    # Update something in runtime params.
    runtime_params.profile_conditions.Ti_bound_right = {0.0: 2.0, 1.0: 4.0}
    # Check pre-update that nothing has changed.
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.profile_conditions.Ti_bound_right, 1.0)
    # Check post-update that the change is reflected.
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.profile_conditions.Ti_bound_right, 2.0)

  def test_update_dynamic_slice_provider_updates_sources(
      self,
  ):
    """Tests that the dynamic slice provider can be updated."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = default_sources.get_default_sources_builder()
    source_models_builder.runtime_params[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ].Iext = 1.0
    geo = geometry.build_circular_geometry(n_rho=4)
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        sources=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    dcs = provider(
        t=0.0,
    )
    for key in source_models_builder.runtime_params.keys():
      self.assertIn(key, dcs.sources)

    # Update an interpolated variable.
    source_models_builder.runtime_params[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ].Iext = 2.0

    # Check pre-update that nothing has changed.
    dcs = provider(
        t=0.0,
    )
    for key in source_models_builder.runtime_params.keys():
      self.assertIn(key, dcs.sources)
    generic_current = dcs.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    assert isinstance(
        generic_current, generic_current_source.DynamicRuntimeParams
    )
    self.assertEqual(generic_current.Iext, 1.0)

    # Update any interpolated variables and check that the change is reflected.
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        sources=source_models_builder.runtime_params,
        torax_mesh=geo.torax_mesh,
    )
    dcs = provider(
        t=0.0,
    )
    for key in source_models_builder.runtime_params.keys():
      self.assertIn(key, dcs.sources)
    generic_current = dcs.sources[
        generic_current_source.GenericCurrentSource.SOURCE_NAME
    ]
    assert isinstance(
        generic_current, generic_current_source.DynamicRuntimeParams
    )
    self.assertEqual(generic_current.Iext, 2.0)

  def test_update_dynamic_slice_provider_updates_transport(
      self,
  ):
    """Tests that the dynamic slice provider can be updated."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    transport = transport_params_lib.RuntimeParams(De_inner=1.0)
    geo = geometry.build_circular_geometry(n_rho=4)
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
        transport=transport,
    )
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.transport.De_inner, 1.0)

    # Update something in transport.
    transport.De_inner = 2.0
    # Check pre-update that nothing has changed.
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.transport.De_inner, 1.0)
    # Check post-update that the change is reflected.
    provider = runtime_params_slice_lib.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        torax_mesh=geo.torax_mesh,
        transport=transport,
    )
    dcs = provider(
        t=0.0,
    )
    self.assertEqual(dcs.transport.De_inner, 2.0)

  def test_static_runtime_params_slice_hash_same_for_same_params(self):
    """Tests that the hash is the same for the same static params."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = default_sources.get_default_sources_builder()
    static_slice1 = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )
    static_slice2 = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )
    self.assertEqual(hash(static_slice1), hash(static_slice2))

  def test_static_runtime_params_slice_hash_different_for_different_params(
      self,
  ):
    """Test that the hash changes when the static params change."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    source_models_builder = default_sources.get_default_sources_builder()
    static_slice1 = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )
    runtime_params_mod = dataclasses.replace(
        runtime_params,
        numerics=dataclasses.replace(
            runtime_params.numerics,
            ion_heat_eq=not runtime_params.numerics.ion_heat_eq,
        ),
    )
    static_slice2 = runtime_params_slice_lib.build_static_runtime_params_slice(
        runtime_params=runtime_params_mod,
        source_runtime_params=source_models_builder.runtime_params,
        torax_mesh=self._geo.torax_mesh,
    )
    self.assertNotEqual(hash(static_slice1), hash(static_slice2))


if __name__ == '__main__':
  absltest.main()
