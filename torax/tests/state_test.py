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
import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import math_utils
from torax import state
from torax.config import build_runtime_params
from torax.config import config_args
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import geometry_provider
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import generic_current_source
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import torax_refs


class StateTest(torax_refs.ReferenceValueTest):

  def setUp(self):
    super().setUp()

    # Make a State object in history mode, output by scan
    self.history_length = 2
    self.sources = sources_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=self.sources.source_model_config
    )

    def make_hist(geo, dynamic_runtime_params_slice, static_slice):
      initial_counter = jnp.array(0)

      def scan_f(counter: jax.Array, _) -> tuple[jax.Array, state.CoreProfiles]:
        core_profiles = initialization.initial_core_profiles(
            dynamic_runtime_params_slice=dynamic_runtime_params_slice,
            static_runtime_params_slice=static_slice,
            geo=geo,
            source_models=source_models,
        )
        # Make one variable in the history track the value of the counter
        value = jnp.ones_like(core_profiles.temp_ion.value) * counter
        core_profiles = config_args.recursive_replace(
            core_profiles, temp_ion={'value': value}
        )
        return counter + 1, core_profiles.history_elem()

      _, history = jax.lax.scan(
          scan_f,
          initial_counter,
          xs=None,
          length=self.history_length,
      )
      return history

    def make_history(runtime_params, geo_provider):
      dynamic_runtime_params_slice, geo = (
          torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
              runtime_params,
              geo_provider,
              sources=self.sources,
          )
      )
      static_slice = build_runtime_params.build_static_runtime_params_slice(
          runtime_params=runtime_params,
          sources=self.sources,
          torax_mesh=geo.torax_mesh,
      )
      # Bind non-JAX arguments so it can be jitted
      bound = functools.partial(
          make_hist,
          geo,
          dynamic_runtime_params_slice,
          static_slice,
      )
      return jax.jit(bound)()

    self._make_history = make_history

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
    sources = sources_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            references.runtime_params,
            references.geometry_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=references.runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    basic_core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    basic_core_profiles.sanity_check()

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_index(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Test State.index."""
    references = references_getter()
    history = self._make_history(
        references.runtime_params, references.geometry_provider
    )

    for i in range(self.history_length):
      self.assertEqual(i, history.index(i).temp_ion.value[0])


class InitialStatesTest(parameterized.TestCase):
  """Unit tests for the `torax.updaters` module."""

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            Ti_bound_right=27.7,
            Te_bound_right={0.0: 42.0, 1.0: 0.0},
            ne_bound_right=({0.0: 0.1, 1.0: 2.0}, 'step'),
            normalize_to_nbar=False,
        ),
    )
    sources = sources_pydantic_model.Sources.from_dict({})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    geo_provider = geometry_provider.ConstantGeometryProvider(
        geometry_pydantic_model.CircularConfig().build_geometry()
    )
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
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
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    sources = sources_pydantic_model.Sources.from_dict({})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    geo_provider = geometry_provider.ConstantGeometryProvider(
        geometry_pydantic_model.CircularConfig().build_geometry()
    )
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
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
    config1 = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_j_is_total_current=True,
            initial_psi_from_j=True,
            nu=2,
            ne_bound_right=0.5,
        ),
    )
    config2 = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_j_is_total_current=False,
            initial_psi_from_j=True,
            nu=2,
            ne_bound_right=0.5,
        ),
    )
    config3 = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_j_is_total_current=False,
            initial_psi_from_j=True,
            nu=2,
            ne_bound_right=0.5,
        ),
    )
    # Needed to generate psi for bootstrap calculation
    config3_helper = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_j_is_total_current=True,
            initial_psi_from_j=True,
            nu=2,
            ne_bound_right=0.5,
        ),
    )
    geo_provider = geometry_pydantic_model.Geometry.from_dict(
        {'geometry_type': geometry_name}
    ).build_provider
    sources = sources_pydantic_model.Sources.from_dict({
        'j_bootstrap': {
            'bootstrap_mult': 0.0,
        },
        'generic_current_source': {},
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dcs1, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            config1,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config1,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles1 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs1,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    dcs2, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            config2,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config2,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles2 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs2,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    sources = sources_pydantic_model.Sources.from_dict({
        'j_bootstrap': {
            'bootstrap_mult': 1.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
        'generic_current_source': {
            'fext': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dcs3, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            config3,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config3,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles3 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs3,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )
    sources = sources_pydantic_model.Sources.from_dict({
        'j_bootstrap': {
            'bootstrap_mult': 0.0, 'mode': runtime_params_lib.Mode.MODEL_BASED
        },
        'generic_current_source': {
            'fext': 0.0, 'mode': runtime_params_lib.Mode.MODEL_BASED
        }
    })
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    dcs3_helper, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            config3_helper,
            geo_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config3_helper,
        sources=sources,
        torax_mesh=geo.torax_mesh,
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
    ctot = config1.profile_conditions.Ip_tot * 1e6 / denom
    jtot_formula = jformula * ctot
    johm_formula = jtot_formula * (
        1
        - dcs1.sources[
            generic_current_source.GenericCurrentSource.SOURCE_NAME
        ].fext  # pytype: disable=attribute-error
    )

    # Calculate bootstrap current for config3 which doesn't zero it out
    source_models = source_models_lib.SourceModels(
        sources_pydantic_model.Sources.from_dict({}).source_model_config
    )
    bootstrap_profile = source_models.j_bootstrap.get_bootstrap(
        dynamic_runtime_params_slice=dcs3,
        static_runtime_params_slice=static_slice,
        geo=geo,
        core_profiles=core_profiles3_helper,
    )
    f_bootstrap = bootstrap_profile.I_bootstrap / (
        config3.profile_conditions.Ip_tot * 1e6
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
    sources = sources_pydantic_model.Sources.from_dict({})
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    config1 = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_psi_from_j=False,
            ne_bound_right=0.5,
        ),
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    dcs1 = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        config1,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )(
        t=config1.numerics.t_initial,
    )
    config2 = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            initial_psi_from_j=True,
            ne_bound_right=0.5,
        ),
    )
    dcs2 = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        config2,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )(
        t=config2.numerics.t_initial,
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config1,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles1 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs1,
        static_runtime_params_slice=static_slice,
        geo=geometry_pydantic_model.CircularConfig().build_geometry(),
        source_models=source_models,
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=config2,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    core_profiles2 = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dcs2,
        static_runtime_params_slice=static_slice,
        geo=geometry_pydantic_model.CircularConfig().build_geometry(),
        source_models=source_models,
    )
    np.testing.assert_allclose(
        core_profiles1.currents.jtot, core_profiles2.currents.jtot
    )


if __name__ == '__main__':
  absltest.main()
