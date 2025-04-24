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
from jax import numpy as jnp
from jax import tree_util
import numpy as np
from torax import output
from torax import state
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.fvm import cell_variable
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import core_profile_helpers
from torax.tests.test_lib import default_sources
from torax.torax_pydantic import model_config
import xarray as xr


SequenceKey = tree_util.SequenceKey
GetAttrKey = tree_util.GetAttrKey
DictKey = tree_util.DictKey


class StateHistoryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.torax_config = model_config.ToraxConfig.from_dict({
        'runtime_params': {
            'profile_conditions': {
                'Ti_bound_right': 27.7,
                'Te_bound_right': {0.0: 42.0, 1.0: 0.0001},
                'ne_bound_right': ({0.0: 0.1, 1.0: 2.0}, 'step'),
            },
        },
        'geometry': {'geometry_type': 'circular', 'n_rho': 4},
        'sources': default_sources.get_default_source_config(),
        'stepper': {},
        'transport': {
            'transport_model': 'constant',
            'chii_const': 2.0,
        },
        'pedestal': {},
    })
    # Make some dummy source profiles that could have come from these sources.
    self.geo = self.torax_config.geometry.build_provider(t=0.0)
    print(self.geo.rho_norm)
    ones = jnp.ones_like(self.geo.rho)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            self.torax_config
        )(t=0.0)
    )
    self.source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            self.geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(self.geo),
        temp_ion={
            'fusion_heat_source': ones,
        },
        temp_el={
            'bremsstrahlung_heat_sink': -ones,
            'ohmic_heat_source': ones * 5,
            'fusion_heat_source': ones,
        },
        ne={},
        psi={},
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        self.torax_config
    )
    source_models = source_models_lib.SourceModels(
        sources=self.torax_config.sources.source_model_config
    )

    self.core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=self.geo,
        source_models=source_models,
    )
    self.core_transport = state.CoreTransport.zeros(self.geo)
    self.source_models = source_models
    # Setup a state history object.
    t = jnp.array(0.0)
    dt = jnp.array(0.1)
    self.sim_state = state.ToraxSimState(
        core_profiles=self.core_profiles,
        core_transport=self.core_transport,
        core_sources=self.source_profiles,
        t=t,
        dt=dt,
        stepper_numeric_outputs=state.StepperNumericOutputs(
            outer_stepper_iterations=1,
            stepper_error_state=1,
            inner_solver_iterations=1,
        ),
        geometry=self.geo,
    )
    sim_error = state.SimError.NO_ERROR
    self._output_state = state.PostProcessedOutputs.zeros(self.geo)

    self.history = output.StateHistory(
        sim_error=sim_error,
        state_history=(self.sim_state,),
        post_processed_outputs_history=(self._output_state,),
        source_models=self.source_models,
        torax_config=self.torax_config,
    )

  def test_geometry_is_saved(self):
    """Tests that the geometry is saved correctly."""
    # Construct a second state with a slightly different geometry.
    self.sim_state_t2 = dataclasses.replace(
        self.sim_state,
        geometry=dataclasses.replace(
            self.sim_state.geometry, Rmaj=self.sim_state.geometry.Rmaj * 2
        ),
    )
    state_history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=(self.sim_state, self.sim_state_t2),
        post_processed_outputs_history=(
            self._output_state,
            self._output_state,
        ),
        source_models=self.source_models,
        torax_config=self.torax_config,
    )
    output_xr = state_history.simulation_output_to_xr()
    print(output_xr.children[output.GEOMETRY].dataset.data_vars)
    saved_rmaj = output_xr.children[output.GEOMETRY].dataset.data_vars['Rmaj']
    np.testing.assert_allclose(
        saved_rmaj.values[0, ...], self.sim_state.geometry.Rmaj
    )
    np.testing.assert_allclose(
        saved_rmaj.values[1, ...], self.sim_state_t2.geometry.Rmaj
    )

  def test_state_history_saves_ion_el_source(self):
    """Tests that an ion electron source is saved correctly."""
    output_xr = self.history.simulation_output_to_xr()
    sources_dataset = output_xr.children[output.CORE_SOURCES].dataset
    self.assertIn('fusion_heat_source_ion', sources_dataset.data_vars)
    self.assertIn('fusion_heat_source_el', sources_dataset.data_vars)
    np.testing.assert_allclose(
        sources_dataset.data_vars['fusion_heat_source_ion'].values[0, ...],
        self.source_profiles.temp_ion['fusion_heat_source'],
    )
    np.testing.assert_allclose(
        sources_dataset.data_vars['fusion_heat_source_el'].values[0, ...],
        self.source_profiles.temp_el['fusion_heat_source'],
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

    loaded_data_tree = output.safe_load_dataset(path)
    xr.testing.assert_equal(loaded_data_tree, data_tree_to_save)

  def test_expected_keys_in_child_nodes(self):
    data_tree = self.history.simulation_output_to_xr()
    expected_child_keys = [
        output.CORE_PROFILES,
        output.CORE_TRANSPORT,
        output.CORE_SOURCES,
        output.POST_PROCESSED_OUTPUTS,
        output.GEOMETRY,
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
    config_dict = json.loads(output_xr.dataset.attrs[output.CONFIG])
    self.assertEqual(config_dict['transport']['transport_model'], 'constant')
    self.assertEqual(config_dict['transport']['chii_const']['value'][1][0], 2.0)
    # Default values are expected to be set in the saved config
    self.assertEqual(config_dict['transport']['chie_const']['value'][1][0], 1.0)

  def test_config_round_trip(self):
    """Tests that the serialization/deserialization of the config is correct."""
    output_xr = self.history.simulation_output_to_xr()
    config_dict = json.loads(output_xr.dataset.attrs[output.CONFIG])
    rebuilt_torax_config = model_config.ToraxConfig.from_dict(config_dict)
    self.assertEqual(rebuilt_torax_config, self.torax_config)

  def test_cell_plus_boundaries_output(self):
    sim_state = self.sim_state
    temp_el = cell_variable.CellVariable(
        value=jnp.ones_like(self.geo.rho),
        dr=self.geo.drho_norm,
        right_face_constraint=2,
        left_face_constraint=18,
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )
    # pylint: disable=invalid-name
    Zimp = jnp.ones_like(self.geo.rho) * 2
    Zimp_face = jnp.ones_like(self.geo.rho_face) * 3
    # pylint: enable=invalid-name
    core_profiles = core_profile_helpers.make_zero_core_profiles(
        self.geo, temp_el=temp_el, Zimp=Zimp, Zimp_face=Zimp_face
    )
    sim_state = dataclasses.replace(sim_state, core_profiles=core_profiles)
    post_processed_outputs = state.PostProcessedOutputs.zeros(self.geo)
    state_history = output.StateHistory(
        sim_error=state.SimError.NO_ERROR,
        state_history=(sim_state, sim_state),
        post_processed_outputs_history=(
            post_processed_outputs,
            post_processed_outputs,
        ),
        source_models=self.source_models,
        torax_config=self.torax_config,
    )
    output_xr = state_history.simulation_output_to_xr()
    np.testing.assert_equal(
        output_xr.children[output.CORE_PROFILES]
        .dataset.data_vars['Zimp']
        .values,
        np.array([[3, 2, 2, 2, 2, 3], [3, 2, 2, 2, 2, 3]]),
    )
    np.testing.assert_equal(
        output_xr.children[output.CORE_PROFILES]
        .dataset.data_vars['temp_el']
        .values,
        np.array([[18, 1, 1, 1, 1, 2], [18, 1, 1, 1, 1, 2]]),
    )


if __name__ == '__main__':
  absltest.main()
