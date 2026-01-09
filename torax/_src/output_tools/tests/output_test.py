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
import json
import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
from jax import numpy as jnp
from jax import tree_util
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import config_loader
from torax._src.core_profiles import initialization
from torax._src.edge import extended_lengyel_solvers
from torax._src.edge import extended_lengyel_standalone
from torax._src.fvm import cell_variable
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.orchestration import run_simulation
from torax._src.orchestration import sim_state
from torax._src.output_tools import impurity_radiation
from torax._src.output_tools import output
from torax._src.output_tools import post_processing
from torax._src.solver import jax_root_finding
from torax._src.sources import source_profiles as source_profiles_lib
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_sources
from torax._src.test_utils import paths
from torax._src.torax_pydantic import model_config
import xarray as xr

SequenceKey = tree_util.SequenceKey
GetAttrKey = tree_util.GetAttrKey
DictKey = tree_util.DictKey


class StateHistoryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'T_i_right_bc': 27.7,
            'T_e_right_bc': {0.0: 42.0, 1.0: 0.0001},
            'n_e_right_bc': ({0.0: 0.1e20, 1.0: 2.0e20}, 'step'),
        },
        'numerics': {},
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': default_sources.get_default_source_config(),
        'solver': {},
        'transport': {
            'model_name': 'constant',
            'chi_i': 2.0,
        },
        'pedestal': {},
    })
    # Make some dummy source profiles that could have come from these sources.
    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    ones = jnp.ones_like(self.geo.rho)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        self.torax_config
    )(t=0.0)
    self.source_profiles = source_profiles_lib.SourceProfiles(
        bootstrap_current=bootstrap_current_base.BootstrapCurrent.zeros(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        T_i={
            'fusion': ones,
        },
        T_e={
            'bremsstrahlung': -ones,
            'ohmic': ones * 5,
            'fusion': ones,
            'generic_heat': 3 * ones,
            'ecrh': 7 * ones,
        },
        n_e={},
        psi={},
    )
    source_models = self.torax_config.sources.build_models()
    neoclassical_models = self.torax_config.neoclassical.build_models()
    self.core_profiles = initialization.initial_core_profiles(
        runtime_params=runtime_params,
        geo=self.geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )
    self.core_transport = state.CoreTransport.zeros(self.geo)
    self.source_models = source_models
    # Setup a state history object.
    t = jnp.array(0.0)
    dt = jnp.array(0.1)
    self.sim_state = sim_state.SimState(
        core_profiles=self.core_profiles,
        core_transport=self.core_transport,
        core_sources=self.source_profiles,
        t=t,
        dt=dt,
        solver_numeric_outputs=state.SolverNumericOutputs(
            outer_solver_iterations=1,
            solver_error_state=1,
            inner_solver_iterations=1,
            sawtooth_crash=False,
        ),
        geometry=self.geo,
        edge_outputs=None,
    )
    sim_error = state.SimError.NO_ERROR
    self._output_state = post_processing.PostProcessedOutputs.zeros(self.geo)

    self.history = output.StateHistory(
        sim_error=sim_error,
        state_history=[self.sim_state],
        post_processed_outputs_history=(self._output_state,),
        torax_config=self.torax_config,
    )

  def test_core_transport_is_saved(self):
    """Tests that the core transport is saved correctly."""
    output_xr = self.history.simulation_output_to_xr()
    core_transport_dataset = output_xr.children[output.PROFILES].dataset
    self.assertIn(output.CHI_TURB_I, core_transport_dataset.data_vars)
    # Spot check one of the expected variables.
    self.assertEqual(
        core_transport_dataset[output.CHI_TURB_I].values.shape,
        (1, len(self.geo.rho_face_norm)),
    )

  def test_geometry_is_saved(self):
    """Tests that the geometry is saved correctly."""
    # Construct a second state with a slightly different geometry.
    self.sim_state_t2 = dataclasses.replace(
        self.sim_state,
        geometry=dataclasses.replace(
            self.sim_state.geometry, R_major=self.sim_state.geometry.R_major * 2
        ),
    )
    state_history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[self.sim_state, self.sim_state_t2],
        post_processed_outputs_history=(
            self._output_state,
            self._output_state,
        ),
        torax_config=self.torax_config,
    )
    output_xr = state_history.simulation_output_to_xr()
    saved_rmaj = output_xr.children[output.SCALARS].dataset.data_vars['R_major']
    np.testing.assert_allclose(
        saved_rmaj.values[0, ...], self.sim_state.geometry.R_major
    )
    np.testing.assert_allclose(
        saved_rmaj.values[1, ...], self.sim_state_t2.geometry.R_major
    )

  def test_geometry_is_saved_with_cell_and_face_vars_merged(self):
    """Tests that the geometry is saved correctly."""
    output_xr = self.history.simulation_output_to_xr()
    # Check that the face and cell var for "volume" was merged.
    self.assertIn(
        'volume', output_xr.children[output.PROFILES].dataset.data_vars
    )
    self.assertNotIn(
        'volume_face', output_xr.children[output.PROFILES].dataset.data_vars
    )
    volume_values = (
        output_xr.children[output.PROFILES].dataset.data_vars['volume'].values
    )
    chex.assert_shape(
        volume_values,
        (1, len(self.geo.rho_norm) + 2),
    )
    np.testing.assert_equal(volume_values[0, 0], self.geo.volume_face[0])
    np.testing.assert_equal(volume_values[0, -1], self.geo.volume_face[-1])
    np.testing.assert_equal(volume_values[0, 1:-1], self.geo.volume)

  def test_state_history_saves_ion_el_source(self):
    """Tests that an ion electron source is saved correctly."""
    output_xr = self.history.simulation_output_to_xr()
    profiles_dataset = output_xr.children[output.PROFILES].dataset
    self.assertIn('p_alpha_i', profiles_dataset.data_vars)
    self.assertIn('p_alpha_e', profiles_dataset.data_vars)
    np.testing.assert_allclose(
        profiles_dataset.data_vars['p_alpha_i'].values[0, ...],
        self.source_profiles.T_i['fusion'],
    )
    np.testing.assert_allclose(
        profiles_dataset.data_vars['p_alpha_e'].values[0, ...],
        self.source_profiles.T_e['fusion'],
    )

  def test_state_history_to_xr(self):
    """Smoke test the `StateHistory.simulation_output_to_xr` method."""
    self.history.simulation_output_to_xr()

  def test_load_core_profiles_from_xr(self):
    """Test serialising and deserialising core profiles consistency."""
    # Output to an xr.DataTree and save to disk.
    data_tree_to_save = self.history.simulation_output_to_xr()
    path = os.path.join(self.create_tempdir().full_path, 'state.nc')
    data_tree_to_save.to_netcdf(path)

    loaded_data_tree = output.load_state_file(path)
    xr.testing.assert_equal(loaded_data_tree, data_tree_to_save)

  def test_expected_keys_in_child_nodes(self):
    data_tree = self.history.simulation_output_to_xr()
    expected_child_keys = [
        output.PROFILES,
        output.SCALARS,
        output.NUMERICS,
    ]
    for key in expected_child_keys:
      self.assertIn(key, data_tree.children)

  def test_concat_datatrees(self):
    """Test helper for concatenating two xr.DataTrees.

    The two datasets have the same structure, one top level dataset and
    two child nodes. Expect the concat to have the same structure and be concat
    of all three datasets.
    Where the two datasets have the same time index, the concat should drop the
    duplicate time step from the second dataset.
    """
    ds_tree1 = xr.Dataset(
        {
            'key': xr.DataArray(
                np.array([1, 2, 3]),
                coords={output.TIME: [1, 2, 3]},
            ),
        },
    )
    ds_tree2 = xr.Dataset(
        {
            'key': xr.DataArray(
                np.array([4, 5, 6, 7]),
                coords={output.TIME: [3, 4, 5, 6]},
            ),
        },
    )
    ds_expected = xr.Dataset(
        {
            'key': xr.DataArray(
                np.array([1, 2, 3, 5, 6, 7]),
                coords={output.TIME: [1, 2, 3, 4, 5, 6]},
            ),
        },
    )
    tree_expected = xr.DataTree(
        dataset=ds_expected.copy(),
        children={
            'a': xr.DataTree(dataset=ds_expected.copy()),
            'b': xr.DataTree(dataset=ds_expected.copy()),
        },
    )
    tree1 = xr.DataTree(
        dataset=ds_tree1.copy(),
        children={
            'a': xr.DataTree(dataset=ds_tree1.copy()),
            'b': xr.DataTree(dataset=ds_tree1.copy()),
        },
    )
    tree2 = xr.DataTree(
        dataset=ds_tree2.copy(),
        children={
            'a': xr.DataTree(dataset=ds_tree2.copy()),
            'b': xr.DataTree(dataset=ds_tree2.copy()),
        },
    )
    xr.testing.assert_equal(
        output.concat_datatrees(tree1, tree2),
        tree_expected,
    )

  def test_config_is_saved(self):
    """Tests that the config is saved correctly."""
    output_xr = self.history.simulation_output_to_xr()
    config_dict = json.loads(output_xr.attrs[output.CONFIG])
    self.assertEqual(config_dict['transport']['model_name'], 'constant')
    # Indexing: ['0.0'][1][1][0] = at time 0, at second rho coordinate,
    # get the value list, and the first value
    self.assertEqual(
        config_dict['transport']['chi_i']['value']['0.0'][1][1][0], 2.0
    )
    # Default values are expected to be set in the saved config
    self.assertEqual(
        config_dict['transport']['chi_e']['value']['0.0'][1][1][0], 1.0
    )

  def test_config_round_trip(self):
    """Tests that the serialization/deserialization of the config is correct."""
    output_xr = self.history.simulation_output_to_xr()
    config_dict = json.loads(output_xr.dataset.attrs[output.CONFIG])
    rebuilt_torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertEqual(rebuilt_torax_config, self.torax_config)

  def test_cell_plus_boundaries_output(self):
    torax_state = self.sim_state
    T_e = cell_variable.CellVariable(  # pylint: disable=invalid-name
        value=jnp.ones_like(self.geo.rho),
        dr=self.geo.drho_norm,
        right_face_constraint=2,
        left_face_constraint=18,
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )
    # pylint: disable=invalid-name
    Z_impurity = jnp.ones_like(self.geo.rho) * 2
    Z_impurity_face = jnp.ones_like(self.geo.rho_face) * 3
    # pylint: enable=invalid-name
    core_profiles = core_profile_helpers.make_zero_core_profiles(
        self.geo,
        T_e=T_e,
        Z_impurity=Z_impurity,
        Z_impurity_face=Z_impurity_face,
    )
    torax_state = dataclasses.replace(torax_state, core_profiles=core_profiles)
    post_processed_outputs = post_processing.PostProcessedOutputs.zeros(
        self.geo
    )
    state_history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[torax_state, torax_state],
        post_processed_outputs_history=(
            post_processed_outputs,
            post_processed_outputs,
        ),
        torax_config=self.torax_config,
    )
    output_xr = state_history.simulation_output_to_xr()
    np.testing.assert_equal(
        output_xr.children[output.PROFILES]
        .dataset.data_vars[output.Z_IMPURITY]
        .values,
        np.array([[3, 2, 2, 2, 2, 3], [3, 2, 2, 2, 2, 3]]),
    )
    np.testing.assert_equal(
        output_xr.children[output.PROFILES]
        .dataset.data_vars[output.T_E]
        .values,
        np.array([[18, 1, 1, 1, 1, 2], [18, 1, 1, 1, 1, 2]]),
    )

  def test_output_profiles_are_correct_shape(self):
    output_xr = self.history.simulation_output_to_xr()
    profile_output_dataset = output_xr.children[output.PROFILES].dataset
    self.assertCountEqual(
        profile_output_dataset.coords,
        {
            output.TIME,
            output.RHO_NORM,
            output.RHO_FACE_NORM,
            output.RHO_CELL_NORM,
        },
    )
    for data_var, data_array in profile_output_dataset.data_vars.items():
      # Check the shape of the underlying data.
      data_array_shape = data_array.values.shape
      self.assertLen(
          data_array_shape,
          2,
          msg=f'Data var {data_var} has incorrect shape {data_array_shape}.',
      )
      data_array_dims = data_array.dims
      self.assertEqual(data_array_dims[0], output.TIME)
      self.assertIn(
          data_array_dims[1],
          [output.RHO_NORM, output.RHO_FACE_NORM, output.RHO_CELL_NORM],
      )

  def test_output_scalars_are_correct_shape(self):
    output_xr = self.history.simulation_output_to_xr()
    scalar_output_dataset = output_xr.children[output.SCALARS].dataset
    # Check coordinates are inherited from top level dataset.
    expected_coords = {
        output.TIME,
        output.RHO_NORM,
        output.RHO_FACE_NORM,
        output.RHO_CELL_NORM,
        output.MAIN_ION,
    }

    self.assertCountEqual(
        scalar_output_dataset.coords,
        expected_coords,
    )
    for data_var, data_array in scalar_output_dataset.data_vars.items():
      # Check that none of the dims are spatial.
      for dim in data_array.dims:
        self.assertNotIn(
            dim,
            [output.RHO_NORM, output.RHO_FACE_NORM, output.RHO_CELL_NORM],
            msg=f'Data var {data_var} in scalars has spatial dim {dim}.',
        )
      data_array_dims = data_array.dims
      if data_array_dims:
        self.assertIn(output.TIME, data_array_dims)

  def test_impurity_radiation_output(self):
    test_data_dir = paths.test_data_dir()
    torax_config = config_loader.build_torax_config_from_file(
        os.path.join(
            test_data_dir,
            'test_iterhybrid_predictor_corrector_mavrin_impurity_radiation.py',
        )
    )

    output_xr, _ = run_simulation.run_simulation(torax_config)

    self.assertIn(
        impurity_radiation.RADIATION_OUTPUT_NAME, output_xr.profiles.data_vars
    )
    self.assertIn(
        impurity_radiation.DENSITY_OUTPUT_NAME, output_xr.profiles.data_vars
    )
    self.assertIn(
        impurity_radiation.Z_OUTPUT_NAME, output_xr.profiles.data_vars
    )
    total_impurity_from_species = (
        output_xr.profiles.radiation_impurity_species.sel(
            impurity_symbol='Ne'
        ).values
        + output_xr.profiles.radiation_impurity_species.sel(
            impurity_symbol='W'
        ).values
    )
    total_impurity_radiation = np.abs(
        output_xr.profiles.p_impurity_radiation_e.values
    )
    np.testing.assert_allclose(
        total_impurity_from_species,
        total_impurity_radiation,
    )

  def test_state_history_with_extended_lengyel_outputs_fixed_point(self):
    """Tests that extended Lengyel edge outputs are saved correctly."""

    # Create dummy ExtendedLengyelOutputs
    extended_lengyel_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        q_parallel=jnp.array(1.0),
        q_perpendicular_target=jnp.array(2.0),
        T_e_separatrix=jnp.array(3.0),
        T_e_target=jnp.array(4.0),
        pressure_neutral_divertor=jnp.array(5.0),
        alpha_t=jnp.array(0.5),
        Z_eff_separatrix=jnp.array(1.5),
        seed_impurity_concentrations={'Ar': jnp.array(0.01)},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=extended_lengyel_solvers.FixedPointOutcome.SUCCESS,
        ),
        calculated_enrichment={'Ar': jnp.array(1.0)},
    )

    sim_state_with_edge = dataclasses.replace(
        self.sim_state,
        edge_outputs=extended_lengyel_outputs,
    )

    history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[sim_state_with_edge],
        post_processed_outputs_history=(self._output_state,),
        torax_config=self.torax_config,
    )

    # Verify edge outputs are stored in the history object
    self.assertEqual(history._edge_outputs[0], extended_lengyel_outputs)

    # Verify that conversion to xarray works and contains edge data
    output_xr = history.simulation_output_to_xr()
    self.assertIsNotNone(output_xr)

    self.assertIn(output.EDGE, output_xr.children)
    edge_dataset = output_xr.children[output.EDGE].dataset

    # Check standard fields
    self.assertIn('q_parallel', edge_dataset.data_vars)
    self.assertIn('T_e_target', edge_dataset.data_vars)

    # Check extended fields
    self.assertIn('alpha_t', edge_dataset.data_vars)
    self.assertIn('Z_eff_separatrix', edge_dataset.data_vars)
    self.assertIn('seed_impurity_concentrations', edge_dataset.data_vars)
    self.assertIn('solver_physics_outcome', edge_dataset.data_vars)
    self.assertIn('calculated_enrichment', edge_dataset.data_vars)
    self.assertIn('fixed_point_outcome', edge_dataset.data_vars)

    # Verify values match
    np.testing.assert_allclose(edge_dataset['alpha_t'].values, np.array([0.5]))
    np.testing.assert_allclose(
        edge_dataset['seed_impurity_concentrations'].sel(impurity='Ar').values,
        np.array([0.01]),
    )

  def test_state_history_with_extended_lengyel_outputs_newton(self):
    """Tests that extended Lengyel edge outputs are saved correctly."""

    # Create dummy ExtendedLengyelOutputs
    extended_lengyel_outputs = extended_lengyel_standalone.ExtendedLengyelOutputs(
        q_parallel=jnp.array(1.0),
        q_perpendicular_target=jnp.array(2.0),
        T_e_separatrix=jnp.array(3.0),
        T_e_target=jnp.array(4.0),
        pressure_neutral_divertor=jnp.array(5.0),
        alpha_t=jnp.array(0.5),
        Z_eff_separatrix=jnp.array(1.5),
        seed_impurity_concentrations={'Ar': jnp.array(0.01)},
        solver_status=extended_lengyel_solvers.ExtendedLengyelSolverStatus(
            physics_outcome=extended_lengyel_solvers.PhysicsOutcome.SUCCESS,
            numerics_outcome=jax_root_finding.RootMetadata(
                iterations=jnp.array(10),
                # Use a vector residual to test the reduction logic
                residual=jnp.array([1e-6, 3e-6]),
                error=jnp.array(0),
                last_tau=jnp.array(1.0),
            ),
        ),
        calculated_enrichment={'Ar': jnp.array(1.0)},
    )

    sim_state_with_edge = dataclasses.replace(
        self.sim_state,
        edge_outputs=extended_lengyel_outputs,
    )

    history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[sim_state_with_edge],
        post_processed_outputs_history=(self._output_state,),
        torax_config=self.torax_config,
    )

    # Verify edge outputs are stored in the history object
    self.assertEqual(history._edge_outputs[0], extended_lengyel_outputs)

    # Verify that conversion to xarray works and contains edge data
    output_xr = history.simulation_output_to_xr()
    self.assertIsNotNone(output_xr)

    self.assertIn(output.EDGE, output_xr.children)
    edge_dataset = output_xr.children[output.EDGE].dataset

    # Check standard fields
    self.assertIn('q_parallel', edge_dataset.data_vars)
    self.assertIn('T_e_target', edge_dataset.data_vars)

    # Check extended fields
    self.assertIn('alpha_t', edge_dataset.data_vars)
    self.assertIn('Z_eff_separatrix', edge_dataset.data_vars)
    self.assertIn('seed_impurity_concentrations', edge_dataset.data_vars)
    self.assertIn('calculated_enrichment', edge_dataset.data_vars)
    self.assertIn('solver_physics_outcome', edge_dataset.data_vars)
    self.assertIn('solver_iterations', edge_dataset.data_vars)
    self.assertIn('solver_residual', edge_dataset.data_vars)
    self.assertIn('solver_error', edge_dataset.data_vars)

    # Verify values match
    np.testing.assert_allclose(edge_dataset['alpha_t'].values, np.array([0.5]))
    np.testing.assert_allclose(
        edge_dataset['seed_impurity_concentrations'].sel(impurity='Ar').values,
        np.array([0.01]),
    )
    np.testing.assert_allclose(
        edge_dataset['solver_iterations'].values, np.array([10])
    )
    # Check that solver_residual is reduced to a scalar per time step
    self.assertEqual(edge_dataset['solver_residual'].dims, (output.TIME,))
    # Mean of abs([1e-6, 3e-6]) is 2e-6
    np.testing.assert_allclose(
        edge_dataset['solver_residual'].values, np.array([2e-6])
    )

  def test_status_attribute_completed(self):
    """Test that status attribute is set to 'completed' for successful runs."""
    output_xr = self.history.simulation_output_to_xr()
    self.assertIn('sim_status', output_xr.numerics)
    self.assertEqual(
        output_xr.numerics.sim_status, state.SimStatus.COMPLETED.value
    )
    self.assertIn('sim_error', output_xr.numerics)
    self.assertEqual(
        output_xr.numerics.sim_error, state.SimError.NO_ERROR.value
    )

  def test_status_attribute_error(self):
    """Test that status attribute is set to 'error' when sim has errors."""
    # Create a history with an error state
    history_with_error = output.StateHistory(
        state_history=[self.sim_state],
        post_processed_outputs_history=(self._output_state,),
        sim_error=state.SimError.NAN_DETECTED,
        torax_config=self.torax_config,
    )
    output_xr = history_with_error.simulation_output_to_xr()
    self.assertIn('sim_status', output_xr.numerics)
    self.assertEqual(output_xr.numerics.sim_status, state.SimStatus.ERROR.value)
    self.assertIn('sim_error', output_xr.numerics)
    self.assertEqual(
        output_xr.numerics.sim_error, state.SimError.NAN_DETECTED.value
    )

  def test_main_ion_fractions_output_matches_config(self):
    """Tests that main_ion_fractions in core_profiles matches config expectations.

    This is a non-trivial case with time-varying scalar inputs for 2 main ions
    (D and T) at different time points.
    """
    # Create a config with 2 main ions with time-varying fractions
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'T_i_right_bc': 27.7,
            'T_e_right_bc': {0.0: 42.0, 1.0: 0.0001},
        },
        'numerics': {},
        'plasma_composition': {
            'main_ion': {
                'D': {0.0: 0.5, 1.0: 0.3},
                'T': {0.0: 0.5, 1.0: 0.7},
            },
        },
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': default_sources.get_default_source_config(),
        'solver': {},
        'transport': {'model_name': 'constant'},
        'pedestal': {},
    })

    geo = torax_config.geometry.build_provider(t=0.0)
    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )

    # Create core profiles at two different times
    runtime_params_t0 = runtime_params_provider(t=0.0)
    runtime_params_t1 = runtime_params_provider(t=1.0)

    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    core_profiles_t0 = initialization.initial_core_profiles(
        runtime_params=runtime_params_t0,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    core_profiles_t1 = initialization.initial_core_profiles(
        runtime_params=runtime_params_t1,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    # Verify the fractions in core_profiles match the config at t=0
    np.testing.assert_allclose(
        core_profiles_t0.main_ion_fractions['D'],
        0.5,
        err_msg='D fraction at t=0 should be 0.5',
    )
    np.testing.assert_allclose(
        core_profiles_t0.main_ion_fractions['T'],
        0.5,
        err_msg='T fraction at t=0 should be 0.5',
    )

    # Verify the fractions in core_profiles match the config at t=1
    np.testing.assert_allclose(
        core_profiles_t1.main_ion_fractions['D'],
        0.3,
        err_msg='D fraction at t=1 should be 0.3',
    )
    np.testing.assert_allclose(
        core_profiles_t1.main_ion_fractions['T'],
        0.7,
        err_msg='T fraction at t=1 should be 0.7',
    )

    # Verify fractions sum to 1 at both times
    sum_t0 = (
        core_profiles_t0.main_ion_fractions['D']
        + core_profiles_t0.main_ion_fractions['T']
    )
    sum_t1 = (
        core_profiles_t1.main_ion_fractions['D']
        + core_profiles_t1.main_ion_fractions['T']
    )
    np.testing.assert_allclose(
        sum_t0, 1.0, err_msg='Fractions at t=0 should sum to 1'
    )
    np.testing.assert_allclose(
        sum_t1, 1.0, err_msg='Fractions at t=1 should sum to 1'
    )

    # Verify that both D and T species are present in the dict
    self.assertIn('D', core_profiles_t0.main_ion_fractions)
    self.assertIn('T', core_profiles_t0.main_ion_fractions)
    self.assertIn('D', core_profiles_t1.main_ion_fractions)
    self.assertIn('T', core_profiles_t1.main_ion_fractions)

  def test_main_ion_fractions_xr_output_matches_config(self):
    """Tests that main_ion_fractions in core_profiles matches config expectations.

    This is a non-trivial case with time-varying scalar inputs for 2 main ions
    (D and T) at different time points.
    """
    # Create a config with 2 main ions with time-varying fractions
    torax_config = model_config.ToraxConfig.from_dict({
        'profile_conditions': {
            'T_i_right_bc': 27.7,
            'T_e_right_bc': {0.0: 42.0, 1.0: 0.0001},
        },
        'numerics': {},
        'plasma_composition': {
            'main_ion': {
                'D': {0.0: 0.5, 1.0: 0.3},
                'T': {0.0: 0.5, 1.0: 0.7},
            },
        },
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': default_sources.get_default_source_config(),
        'solver': {},
        'transport': {'model_name': 'constant'},
        'pedestal': {},
    })

    geo = torax_config.geometry.build_provider(t=0.0)
    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )

    # Create core profiles at two different times
    runtime_params_t0 = runtime_params_provider(t=0.0)
    runtime_params_t1 = runtime_params_provider(t=1.0)

    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()

    core_profiles_t0 = initialization.initial_core_profiles(
        runtime_params=runtime_params_t0,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    core_profiles_t1 = initialization.initial_core_profiles(
        runtime_params=runtime_params_t1,
        geo=geo,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
    )

    sim_state_t0 = dataclasses.replace(
        self.sim_state,
        core_profiles=core_profiles_t0,
        geometry=geo,
        t=jnp.array(0.0),
    )

    sim_state_t1 = dataclasses.replace(
        self.sim_state,
        core_profiles=core_profiles_t1,
        geometry=geo,
        t=jnp.array(1.0),
    )

    history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=[sim_state_t0, sim_state_t1],
        post_processed_outputs_history=(self._output_state, self._output_state),
        torax_config=torax_config,
    )

    output_xr = history.simulation_output_to_xr()

    # Check that main_ion_fractions is present in scalars
    scalars_dataset = output_xr.children[output.SCALARS].dataset
    self.assertIn('main_ion_fractions', scalars_dataset.data_vars)

    fractions_xr = scalars_dataset['main_ion_fractions']

    # Verify values at t=0
    np.testing.assert_allclose(
        fractions_xr.sel(main_ion='D', time=0.0).values, 0.5
    )
    np.testing.assert_allclose(
        fractions_xr.sel(main_ion='T', time=0.0).values, 0.5
    )

    # Verify values at t=1
    np.testing.assert_allclose(
        fractions_xr.sel(main_ion='D', time=1.0).values, 0.3
    )
    np.testing.assert_allclose(
        fractions_xr.sel(main_ion='T', time=1.0).values, 0.7
    )


if __name__ == '__main__':
  absltest.main()
