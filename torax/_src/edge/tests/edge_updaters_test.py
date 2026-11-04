# Copyright 2025 DeepMind Technologies Limited
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
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles.plasma_composition import electron_density_ratios
from torax._src.edge import base as edge_base
from torax._src.edge import runtime_params as edge_runtime_params
from torax._src.edge import updaters
from torax._src.output_tools import output_keys
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import config
import xarray as xr

# pylint: disable=invalid-name

_TARGET_EDGE_RATIO = 0.05
_INITIAL_EDGE_RATIO = 0.02
_INITIAL_AXIS_RATIO = 0.01
_UNCHANGED_RATIO = 0.015


class UpdateRuntimeParamsFromEdgeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    config_dict = default_configs.get_default_config_dict()
    config_dict['profile_conditions']['n_e_right_bc'] = 0.5
    config_dict['profile_conditions']['n_e_right_bc_is_f_GW'] = True
    config_dict['plasma_composition']['impurity'] = {
        'impurity_mode': 'n_e_ratios',
        'species': {
            'N': {0: _INITIAL_AXIS_RATIO, 1: _INITIAL_EDGE_RATIO},
            'Ne': {0: _UNCHANGED_RATIO, 1: _UNCHANGED_RATIO},
        },
    }
    torax_config = config.ToraxConfig.from_dict(config_dict)
    provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    self.runtime_params = provider(t=0.0)
    self.edge_outputs = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(0.123),
        T_i_right_bc=jnp.array(0.456),
        n_e_right_bc=jnp.array(3.45e19),
        impurity_right_bc={'N': jnp.array(_TARGET_EDGE_RATIO)},
    )

  def _with_edge_flags(
      self,
      *,
      update_T_e: bool = False,
      update_T_i: bool = False,
      update_n_e: bool = False,
      update_n_impurities: bool = False,
  ):
    return dataclasses.replace(
        self.runtime_params,
        edge=edge_runtime_params.RuntimeParams(
            update_T_e=update_T_e,
            update_T_i=update_T_i,
            update_n_e=update_n_e,
            update_n_impurities=update_n_impurities,
        ),
    )

  def test_none_edge_outputs_is_noop(self):
    params = self._with_edge_flags(
        update_T_e=True,
        update_T_i=True,
        update_n_e=True,
        update_n_impurities=True,
    )
    updated = updaters.update_runtime_params(params, None)
    self.assertIs(updated, params)

  def test_invalid_for_core_update_skips_all_updates(self):
    params = self._with_edge_flags(
        update_T_e=True,
        update_T_i=True,
        update_n_e=True,
        update_n_impurities=True,
    )
    invalid_outputs = dataclasses.replace(
        self.edge_outputs, valid_for_core_update=jnp.array(False)
    )
    updated = updaters.update_runtime_params(params, invalid_outputs)

    np.testing.assert_allclose(
        updated.profile_conditions.T_e_right_bc,
        params.profile_conditions.T_e_right_bc,
    )
    np.testing.assert_allclose(
        updated.profile_conditions.T_i_right_bc,
        params.profile_conditions.T_i_right_bc,
    )
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 0.5)
    self.assertTrue(updated.profile_conditions.n_e_right_bc_is_f_GW)
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    assert isinstance(
        params.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios['N'],
        params.plasma_composition.impurity.n_e_ratios['N'],
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios_face['N'],
        params.plasma_composition.impurity.n_e_ratios_face['N'],
    )

  def test_update_T_e(self):
    params = self._with_edge_flags(update_T_e=True)
    updated = updaters.update_runtime_params(params, self.edge_outputs)

    # Electron temperature boundary condition is updated.
    np.testing.assert_allclose(updated.profile_conditions.T_e_right_bc, 0.123)
    # Ion temperature boundary condition remains unchanged.
    np.testing.assert_allclose(
        updated.profile_conditions.T_i_right_bc,
        params.profile_conditions.T_i_right_bc,
    )
    # Density boundary conditions remain unchanged.
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 0.5)
    self.assertTrue(updated.profile_conditions.n_e_right_bc_is_f_GW)
    # Impurity conditions remain unchanged.
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    assert isinstance(
        params.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )

  def test_update_T_i(self):
    params = self._with_edge_flags(update_T_i=True)
    updated = updaters.update_runtime_params(params, self.edge_outputs)

    # Electron temperature boundary condition remains unchanged.
    np.testing.assert_allclose(
        updated.profile_conditions.T_e_right_bc,
        params.profile_conditions.T_e_right_bc,
    )
    # Ion temperature boundary condition is updated.
    np.testing.assert_allclose(updated.profile_conditions.T_i_right_bc, 0.456)
    # Density boundary conditions remain unchanged.
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 0.5)
    self.assertTrue(updated.profile_conditions.n_e_right_bc_is_f_GW)
    # Impurity conditions remain unchanged.
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )

  def test_update_n_e(self):
    params = self._with_edge_flags(update_n_e=True)
    updated = updaters.update_runtime_params(params, self.edge_outputs)

    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 3.45e19)
    self.assertFalse(updated.profile_conditions.n_e_right_bc_is_f_GW)
    # Electron temperature boundary condition remains unchanged.
    np.testing.assert_allclose(
        updated.profile_conditions.T_e_right_bc,
        params.profile_conditions.T_e_right_bc,
    )
    # Ion temperature boundary condition remains unchanged.
    np.testing.assert_allclose(
        updated.profile_conditions.T_i_right_bc,
        params.profile_conditions.T_i_right_bc,
    )
    # Impurity conditions remain unchanged.
    assert isinstance(
        updated.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    assert isinstance(
        params.plasma_composition.impurity,
        electron_density_ratios.RuntimeParams,
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios['N'],
        params.plasma_composition.impurity.n_e_ratios['N'],
    )
    np.testing.assert_allclose(
        updated.plasma_composition.impurity.n_e_ratios_face['N'],
        params.plasma_composition.impurity.n_e_ratios_face['N'],
    )

  def test_update_n_impurities_scales_profile(self):
    params = self._with_edge_flags(update_n_impurities=True)
    initial_impurity_params = params.plasma_composition.impurity
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )

    updated = updaters.update_runtime_params(params, self.edge_outputs)
    updated_impurity_params = updated.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )

    scaling_factor = _TARGET_EDGE_RATIO / _INITIAL_EDGE_RATIO
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['N'],
        initial_impurity_params.n_e_ratios['N'] * scaling_factor,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['N'],
        initial_impurity_params.n_e_ratios_face['N'] * scaling_factor,
        rtol=1e-5,
    )
    # 'Ne' is not in edge_outputs.impurity_right_bc, so it remains unchanged.
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['Ne'],
        initial_impurity_params.n_e_ratios['Ne'],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios_face['Ne'],
        initial_impurity_params.n_e_ratios_face['Ne'],
        rtol=1e-5,
    )

  def test_combined_runtime_params(self):
    params = dataclasses.replace(
        self.runtime_params,
        edge=edge_runtime_params.CombinedRuntimeParams(
            update_T_e=True,
            update_T_i=True,
            update_n_e=True,
            update_n_impurities=True,
            sub_models={
                'te': self._with_edge_flags(update_T_e=True).edge,
                'ti': self._with_edge_flags(update_T_i=True).edge,
                'ne': self._with_edge_flags(update_n_e=True).edge,
                'imp': self._with_edge_flags(update_n_impurities=True).edge,
            },
        ),
    )
    updated = updaters.update_runtime_params(params, self.edge_outputs)
    np.testing.assert_allclose(updated.profile_conditions.T_e_right_bc, 0.123)
    np.testing.assert_allclose(updated.profile_conditions.T_i_right_bc, 0.456)
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 3.45e19)
    updated_impurity_params = updated.plasma_composition.impurity
    initial_impurity_params = params.plasma_composition.impurity
    assert isinstance(
        updated_impurity_params, electron_density_ratios.RuntimeParams
    )
    assert isinstance(
        initial_impurity_params, electron_density_ratios.RuntimeParams
    )
    scaling_factor = _TARGET_EDGE_RATIO / _INITIAL_EDGE_RATIO
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['N'],
        initial_impurity_params.n_e_ratios['N'] * scaling_factor,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        updated_impurity_params.n_e_ratios['Ne'],
        initial_impurity_params.n_e_ratios['Ne'],
        rtol=1e-5,
    )

  def test_none_bc_fields_are_noop(self):
    params = self._with_edge_flags(
        update_T_e=True,
        update_T_i=False,
        update_n_e=False,
        update_n_impurities=False,
    )
    partial_outputs = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(0.123),
    )
    updated = updaters.update_runtime_params(params, partial_outputs)
    np.testing.assert_allclose(updated.profile_conditions.T_e_right_bc, 0.123)
    np.testing.assert_allclose(
        updated.profile_conditions.T_i_right_bc,
        params.profile_conditions.T_i_right_bc,
    )
    np.testing.assert_allclose(updated.profile_conditions.n_e_right_bc, 0.5)
    self.assertTrue(updated.profile_conditions.n_e_right_bc_is_f_GW)


class EdgeModelOutputsDataTreeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.ds = xr.Dataset(
        data_vars={
            output_keys.T_E_RIGHT_BC: ((), 100.0),
            output_keys.T_I_RIGHT_BC: ((), 150.0),
            output_keys.N_E_RIGHT_BC: ((), 2.0e19),
            output_keys.IMPURITY_RIGHT_BC: (('impurity',), [0.03]),
        },
        coords={'impurity': ['Ne']},
    )
    self.datatree = xr.DataTree(dataset=self.ds)

  def test_from_datatree_success(self):
    parsed = edge_base.EdgeModelOutputs.from_datatree(self.datatree)
    self.assertIsNotNone(parsed.T_e_right_bc)
    self.assertIsNotNone(parsed.T_i_right_bc)
    self.assertIsNotNone(parsed.n_e_right_bc)
    assert parsed.T_e_right_bc is not None
    assert parsed.T_i_right_bc is not None
    assert parsed.n_e_right_bc is not None
    np.testing.assert_allclose(parsed.T_e_right_bc, 100.0)
    np.testing.assert_allclose(parsed.T_i_right_bc, 150.0)
    np.testing.assert_allclose(parsed.n_e_right_bc, 2.0e19)
    self.assertIn('Ne', parsed.impurity_right_bc)
    np.testing.assert_allclose(parsed.impurity_right_bc['Ne'], 0.03)
    self.assertTrue(bool(parsed.valid_for_core_update))

  def test_from_datatree_nan_sets_valid_for_core_update_false(self):
    ds_nan = xr.Dataset(
        data_vars={
            output_keys.T_E_RIGHT_BC: ((), np.nan),
            output_keys.T_I_RIGHT_BC: ((), 150.0),
        }
    )
    parsed = edge_base.EdgeModelOutputs.from_datatree(
        xr.DataTree(dataset=ds_nan)
    )
    self.assertFalse(bool(parsed.valid_for_core_update))

  def test_from_datatree_with_explicit_valid_for_core_update_false(self):
    ds_invalid = xr.Dataset(
        data_vars={
            output_keys.T_E_RIGHT_BC: ((), 100.0),
            output_keys.T_I_RIGHT_BC: ((), 150.0),
            output_keys.N_E_RIGHT_BC: ((), 2.0e19),
            output_keys.VALID_FOR_CORE_UPDATE: ((), False),
        }
    )
    parsed = edge_base.EdgeModelOutputs.from_datatree(
        xr.DataTree(dataset=ds_invalid)
    )
    self.assertFalse(bool(parsed.valid_for_core_update))

    orig = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(50.0),
        T_i_right_bc=jnp.array(50.0),
        n_e_right_bc=jnp.array(1.0e19),
        valid_for_core_update=jnp.array(True),
    )
    replaced = orig.replace_bcs_from_datatree(xr.DataTree(dataset=ds_invalid))
    self.assertFalse(bool(replaced.valid_for_core_update))

  def test_from_datatree_none_raises(self):
    with self.assertRaises(ValueError):
      edge_base.EdgeModelOutputs.from_datatree(
          None  # pyrefly: ignore[bad-argument-type]
      )

  def test_from_datatree_missing_impurity_coord_raises(self):
    ds_bad = xr.Dataset(
        data_vars={output_keys.IMPURITY_RIGHT_BC: (('dim',), [0.03])}
    )
    with self.assertRaises(ValueError):
      edge_base.EdgeModelOutputs.from_datatree(xr.DataTree(dataset=ds_bad))

  def test_replace_bcs_from_datatree_success(self):
    outputs = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(10.0),
        T_i_right_bc=jnp.array(20.0),
        n_e_right_bc=jnp.array(1.0e19),
        impurity_right_bc={'Ne': jnp.array(0.01)},
    )
    replaced = outputs.replace_bcs_from_datatree(self.datatree)
    self.assertIsNotNone(replaced.T_e_right_bc)
    self.assertIsNotNone(replaced.T_i_right_bc)
    self.assertIsNotNone(replaced.n_e_right_bc)
    assert replaced.T_e_right_bc is not None
    assert replaced.T_i_right_bc is not None
    assert replaced.n_e_right_bc is not None
    np.testing.assert_allclose(replaced.T_e_right_bc, 100.0)
    np.testing.assert_allclose(replaced.T_i_right_bc, 150.0)
    np.testing.assert_allclose(replaced.n_e_right_bc, 2.0e19)
    np.testing.assert_allclose(replaced.impurity_right_bc['Ne'], 0.03)
    self.assertTrue(bool(replaced.valid_for_core_update))

  def test_replace_bcs_from_datatree_missing_expected_bc_raises(self):
    outputs = edge_base.EdgeModelOutputs(
        T_e_right_bc=jnp.array(10.0),
        T_i_right_bc=jnp.array(20.0),
    )
    empty_dt = xr.DataTree(dataset=xr.Dataset())
    with self.assertRaises(ValueError):
      outputs.replace_bcs_from_datatree(empty_dt)

  def test_replace_bcs_from_datatree_missing_expected_impurity_raises(self):
    outputs = edge_base.EdgeModelOutputs(
        impurity_right_bc={'Ar': jnp.array(0.01)},
    )
    with self.assertRaises(ValueError):
      outputs.replace_bcs_from_datatree(self.datatree)

  def test_combined_edge_outputs_replace_bcs_success(self):
    sub = edge_base.EdgeModelOutputs(T_e_right_bc=jnp.array(10.0))
    combined = edge_base.CombinedEdgeOutputs(
        T_e_right_bc=jnp.array(10.0),
        sub_models={'sub1': sub},
    )
    combined_dt = xr.DataTree(
        dataset=self.ds,
        children={'sub1': xr.DataTree(dataset=self.ds)},
    )
    replaced = combined.replace_bcs_from_datatree(combined_dt)
    self.assertIsNotNone(replaced.T_e_right_bc)
    assert replaced.T_e_right_bc is not None
    np.testing.assert_allclose(replaced.T_e_right_bc, 100.0)
    sub1_out = replaced.sub_models['sub1']
    self.assertIsNotNone(sub1_out.T_e_right_bc)
    assert sub1_out.T_e_right_bc is not None
    np.testing.assert_allclose(sub1_out.T_e_right_bc, 100.0)

  def test_combined_edge_outputs_missing_sub_model_raises(self):
    sub = edge_base.EdgeModelOutputs(T_e_right_bc=jnp.array(10.0))
    combined = edge_base.CombinedEdgeOutputs(
        T_e_right_bc=jnp.array(10.0),
        sub_models={'sub1': sub},
    )
    with self.assertRaises(ValueError):
      combined.replace_bcs_from_datatree(self.datatree)


if __name__ == '__main__':
  absltest.main()
