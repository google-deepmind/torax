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
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import math_utils
from torax import state
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.fvm import cell_variable
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import generic_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles as source_profiles_lib
from torax.tests.test_lib import core_profile_helpers
from torax.tests.test_lib import default_configs
from torax.tests.test_lib import torax_refs
from torax.torax_pydantic import model_config


def make_zero_core_profiles(
    geo: geometry.Geometry,
) -> state.CoreProfiles:
  """Returns a dummy CoreProfiles object."""
  zero_cell_variable = cell_variable.CellVariable(
      value=jnp.zeros_like(geo.rho),
      dr=geo.drho_norm,
      right_face_constraint=jnp.ones(()),
      right_face_grad_constraint=None,
  )
  return state.CoreProfiles(
      currents=state.Currents.zeros(geo),
      temp_ion=zero_cell_variable,
      temp_el=zero_cell_variable,
      psi=zero_cell_variable,
      psidot=zero_cell_variable,
      ne=zero_cell_variable,
      ni=zero_cell_variable,
      nimp=zero_cell_variable,
      q_face=jnp.zeros_like(geo.rho_face),
      s_face=jnp.zeros_like(geo.rho_face),
      nref=jnp.array(0.0),
      vloop_lcfs=jnp.array(0.0),
      Zi=jnp.zeros_like(geo.rho),
      Zi_face=jnp.zeros_like(geo.rho_face),
      Ai=jnp.zeros(()),
      Zimp=jnp.zeros_like(geo.rho),
      Zimp_face=jnp.zeros_like(geo.rho_face),
      Aimp=jnp.zeros(()),
  )


class StateTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_sanity_check(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Make sure State.sanity_check can be called."""
    references = references_getter()
    source_models = source_models_lib.SourceModels(
        sources=references.config.sources.source_model_config
    )
    dynamic_runtime_params_slice, geo = references.get_dynamic_slice_and_geo()
    static_slice = build_runtime_params.build_static_params_from_config(
        references.config)
    basic_core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    basic_core_profiles.sanity_check()

  def test_nan_check(self):
    t = jnp.array(0.0)
    dt = jnp.array(0.1)
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    source_profiles = source_profiles_lib.SourceProfiles(
        j_bootstrap=source_profiles_lib.BootstrapCurrentProfile.zero_profile(
            geo
        ),
        qei=source_profiles_lib.QeiInfo.zeros(geo),
    )
    dummy_cell_variable = cell_variable.CellVariable(
        value=jnp.zeros_like(geo.rho),
        dr=geo.drho_norm,
        right_face_constraint=jnp.ones(()),
        right_face_grad_constraint=None,
    )
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    sim_state = state.ToraxSimState(
        core_profiles=core_profiles,
        core_transport=state.CoreTransport.zeros(geo),
        core_sources=source_profiles,
        t=t,
        dt=dt,
        solver_numeric_outputs=state.SolverNumericOutputs(
            outer_solver_iterations=1,
            solver_error_state=1,
            inner_solver_iterations=1,
        ),
        geometry=geo,
    )
    post_processed_outputs = state.PostProcessedOutputs.zeros(geo)

    with self.subTest('no NaN'):
      error = sim_state.check_for_errors()
      self.assertEqual(error, state.SimError.NO_ERROR)
      error = state.check_for_errors(sim_state, post_processed_outputs)
      self.assertEqual(error, state.SimError.NO_ERROR)

    with self.subTest('NaN in BC'):
      core_profiles = dataclasses.replace(
          core_profiles,
          temp_ion=dataclasses.replace(
              core_profiles.temp_ion,
              right_face_constraint=jnp.array(jnp.nan),
          ),
      )
      new_sim_state_core_profiles = dataclasses.replace(
          sim_state, core_profiles=core_profiles
      )
      error = new_sim_state_core_profiles.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = state.check_for_errors(
          new_sim_state_core_profiles, post_processed_outputs
      )
      self.assertEqual(error, state.SimError.NAN_DETECTED)

    with self.subTest('NaN in post processed outputs'):
      new_post_processed_outputs = dataclasses.replace(
          post_processed_outputs,
          P_external_tot=jnp.array(jnp.nan),
      )
      error = new_post_processed_outputs.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = state.check_for_errors(sim_state, new_post_processed_outputs)
      self.assertEqual(error, state.SimError.NAN_DETECTED)

    with self.subTest('NaN in one element of source array'):
      nan_array = np.zeros_like(geo.rho)
      nan_array[-1] = np.nan
      j_bootstrap = dataclasses.replace(
          sim_state.core_sources.j_bootstrap,
          j_bootstrap=nan_array,
      )
      new_core_sources = dataclasses.replace(
          sim_state.core_sources, j_bootstrap=j_bootstrap
      )
      new_sim_state_sources = dataclasses.replace(
          sim_state, core_sources=new_core_sources
      )
      error = new_sim_state_sources.check_for_errors()
      self.assertEqual(error, state.SimError.NAN_DETECTED)
      error = state.check_for_errors(
          new_sim_state_sources, post_processed_outputs
      )
      self.assertEqual(error, state.SimError.NAN_DETECTED)


class InitialStatesTest(parameterized.TestCase):

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    config = default_configs.get_default_config_dict()
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    config['profile_conditions'] = {
        'Ti_bound_right': 27.7,
        'Te_bound_right': {0.0: 42.0, 1.0: 0.001},
        'ne_bound_right': ({0.0: 0.1, 1.0: 2.0}, 'step'),
        'normalize_to_nbar': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dynamic_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    np.testing.assert_allclose(
        core_profiles.temp_ion.right_face_constraint, 27.7
    )
    np.testing.assert_allclose(
        core_profiles.temp_el.right_face_constraint, 42.0
    )
    np.testing.assert_allclose(core_profiles.ne.right_face_constraint, 0.1)

  def test_core_profiles_quasineutrality_check(self):
    """Tests core_profiles quasineutrality check on initial state."""
    torax_config = model_config.ToraxConfig.from_dict(
        default_configs.get_default_config_dict()
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=dynamic_runtime_params_slice_provider,
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    assert core_profiles.quasineutrality_satisfied()
    core_profiles = dataclasses.replace(
        core_profiles,
        Zi=core_profiles.Zi * 2.0,
    )
    assert not core_profiles.quasineutrality_satisfied()

  @parameterized.parameters([
      dict(geometry_name='circular'),
      dict(geometry_name='chease'),
  ])
  def test_initial_psi_from_j(
      self,
      geometry_name: str,
  ):
    """Tests expected behaviour of initial psi and current options."""
    config = default_configs.get_default_config_dict()
    config['geometry']['geometry_type'] = geometry_name
    config['sources'] = {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_current': {},
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    config1 = dict(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        nu=2,
        ne_bound_right=0.5,
    )
    config2 = dict(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        nu=2,
        ne_bound_right=0.5,
    )
    config3 = dict(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        nu=2,
        ne_bound_right=0.5,
    )
    # Needed to generate psi for bootstrap calculation
    config3_helper = dict(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        nu=2,
        ne_bound_right=0.5,
    )
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )

    torax_config.update_fields({'profile_conditions': config1})
    dcs1, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                torax_config
            ),
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles1 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs1,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    torax_config.update_fields({'profile_conditions': config2})
    dcs2, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                torax_config
            ),
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles2 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs2,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    new_source_config = {
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
        'generic_current': {
            'fext': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    torax_config.update_fields({
        'profile_conditions': config3,
        'sources': new_source_config,
    })
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dcs3, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                torax_config
            ),
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles3 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs3,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    new_source_config = {
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
        'generic_current': {
            'fext': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    torax_config.update_fields({
        'profile_conditions': config3_helper,
        'sources': new_source_config,
    })
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    dcs3_helper, geo = (
        build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
            t=torax_config.numerics.t_initial,
            dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
                torax_config
            ),
            geometry_provider=torax_config.geometry.build_provider,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles3_helper = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs3_helper,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    # calculate total and Ohmic current profiles arising from nu=2
    jformula = (1 - geo.rho_norm**2) ** 2
    denom = jax.scipy.integrate.trapezoid(jformula * geo.spr, geo.rho_norm)
    ctot = torax_config.profile_conditions.Ip_tot.value[0] * 1e6 / denom
    jtot_formula = jformula * ctot
    johm_formula = jtot_formula * (
        1
        - dcs1.sources[
            generic_current_source.GenericCurrentSource.SOURCE_NAME
        ].fext  # pytype: disable=attribute-error
    )

    # Calculate bootstrap current for config3 which doesn't zero it out
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
        dynamic_runtime_params_slice=dcs3,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles3_helper,
    )
    f_bootstrap = bootstrap_profile.I_bootstrap / (
        torax_config.profile_conditions.Ip_tot.value[0] * 1e6
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles1.currents.jtot,
        core_profiles2.currents.jtot,
    )

    np.testing.assert_allclose(
        core_profiles1.currents.external_current_source
        + core_profiles1.currents.johm,
        jtot_formula,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles1.currents.johm,
        johm_formula,
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        core_profiles2.currents.johm,
        johm_formula,
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles2.currents.jtot_face,
        math_utils.cell_to_face(
            jtot_formula,
            geo,
            preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
        ),
    )
    np.testing.assert_allclose(
        core_profiles3.currents.johm,
        jtot_formula * (1 - f_bootstrap),
        rtol=1e-12,
        atol=1e-12,
    )

  def test_initial_psi_from_geo_noop_circular(self):
    """Tests expected behaviour of initial psi and current options."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_from_j': False,
        'ne_bound_right': 0.5,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources.source_model_config
    )
    geo = torax_config.geometry.build_provider(torax_config.numerics.t_initial)
    dcs1 = build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles1 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs1,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    torax_config.update_fields(
        {'profile_conditions.initial_psi_from_j': True}
    )
    dcs2 = build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles2 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs2,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    np.testing.assert_allclose(
        core_profiles1.currents.jtot, core_profiles2.currents.jtot
    )

  def test_core_profiles_negative_values_check(self):
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    core_profiles = core_profile_helpers.make_zero_core_profiles(geo)
    with self.subTest('no negative values'):
      self.assertFalse(core_profiles.negative_temperature_or_density())
    with self.subTest('negative temp_ion triggers'):
      new_core_profiles = dataclasses.replace(
          core_profiles,
          temp_ion=dataclasses.replace(
              core_profiles.temp_ion,
              value=jnp.array(-1.0),
          ),
      )
      self.assertTrue(new_core_profiles.negative_temperature_or_density())
    with self.subTest('negative psi does not trigger'):
      new_core_profiles = dataclasses.replace(
          core_profiles,
          psi=dataclasses.replace(
              core_profiles.psi,
              value=jnp.array(-1.0),
          ),
      )
      self.assertFalse(new_core_profiles.negative_temperature_or_density())


if __name__ == '__main__':
  absltest.main()
