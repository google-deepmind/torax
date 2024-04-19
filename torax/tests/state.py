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

"""Unit tests for torax.state and torax.core_profile_setters."""

import functools
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax import config as config_lib
from torax import config_slice
from torax import core_profile_setters
from torax import geometry
from torax import state
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import torax_refs


class StateTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.state` module."""

  def setUp(self):
    super().setUp()

    # Make a State object in history mode, output by scan
    self.history_length = 2

    def make_hist(config, geo):
      initial_counter = jnp.array(0)

      def scan_f(counter: jax.Array, _) -> tuple[jax.Array, state.CoreProfiles]:
        core_profiles = core_profile_setters.initial_core_profiles(
            config_slice.build_static_config_slice(config),
            config_slice.build_dynamic_config_slice(config),
            geo,
        )
        # Make one variable in the history track the value of the counter
        value = jnp.ones_like(core_profiles.temp_ion.value) * counter
        core_profiles = config_lib.recursive_replace(
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

    def make_history(config, geo):
      # Bind non-JAX arguments so it can be jitted
      bound = functools.partial(make_hist, config, geo)
      return jax.jit(bound)()

    self._make_history = make_history

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_sanity_check(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Make sure State.sanity_check can be called."""
    references = references_getter()
    basic_core_profiles = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(references.config),
        config_slice.build_dynamic_config_slice(references.config),
        references.geo,
    )
    basic_core_profiles.sanity_check()

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_index(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Test State.index."""
    references = references_getter()
    history = self._make_history(references.config, references.geo)

    for i in range(self.history_length):
      self.assertEqual(i, history.index(i).temp_ion.value[0])

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_project(
      self,
      references_getter: Callable[[], torax_refs.References],
  ):
    """Test State.project."""
    references = references_getter()
    history = self._make_history(references.config, references.geo)

    seed = 20230421
    rng_state = jax.random.PRNGKey(seed)
    del seed  # Make sure seed isn't accidentally re-used
    weights = jax.random.normal(rng_state, (self.history_length,))
    del rng_state  # Make sure rng_state isn't accidentally re-used

    expected = jnp.dot(weights, jnp.arange(self.history_length))

    projected = history.project(weights)

    actual = projected.temp_ion.value[0]

    np.testing.assert_allclose(expected, actual)


class InitialStatesTest(parameterized.TestCase):
  """Unit tests for the `torax.core_profile_setters` module."""

  def test_initial_boundary_condition_from_time_dependent_params(self):
    """Tests that the initial boundary conditions are set from the config."""
    # Boundary conditions can be time-dependent, but when creating the initial
    # core profiles, we want to grab the boundary condition params at time 0.
    config = config_lib.Config(
        profile_conditions=config_lib.ProfileConditions(
            Ti_bound_right=27.7,
            Te_bound_right={0.0: 42.0, 1.0: 0.0},
            ne_bound_right=config_lib.InterpolationParam(
                {0.0: 0.1, 1.0: 2.0},
                interpolation_mode=config_lib.InterpolationMode.STEP,
            ),
        ),
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config),
        config_slice.build_dynamic_config_slice(config),
        geometry.build_circular_geometry(config),
    )
    np.testing.assert_allclose(
        core_profiles.temp_ion.right_face_constraint, 27.7
    )
    np.testing.assert_allclose(
        core_profiles.temp_el.right_face_constraint, 42.0
    )
    np.testing.assert_allclose(core_profiles.ne.right_face_constraint, 0.1)

  @parameterized.parameters([
      dict(geo_builder=geometry.build_circular_geometry),
      dict(geo_builder=geometry.build_chease_geometry),
  ])
  def test_initial_psi_from_j(
      self, geo_builder: Callable[[config_lib.Config], geometry.Geometry]
  ):
    """Tests expected behaviour of initial psi and current options."""
    config1 = config_lib.Config(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        nu=2,
        numerics=config_lib.Numerics(
            bootstrap_mult=0,
        ),
    )
    config2 = config_lib.Config(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        nu=2,
        numerics=config_lib.Numerics(
            bootstrap_mult=0,
        ),
    )
    config3 = config_lib.Config(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        nu=2,
        fext=0.0,
        numerics=config_lib.Numerics(
            bootstrap_mult=1,
        ),
    )
    # Needed to generate psi for bootstrap calculation
    config3_helper = config_lib.Config(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        nu=2,
        fext=0.0,
        numerics=config_lib.Numerics(
            bootstrap_mult=0,
        ),
    )
    geo = geo_builder(config1)
    core_profiles1 = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config1),
        config_slice.build_dynamic_config_slice(config1),
        geo=geo,
    )
    core_profiles2 = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config2),
        config_slice.build_dynamic_config_slice(config2),
        geo=geo,
    )
    core_profiles3 = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config3),
        config_slice.build_dynamic_config_slice(config3),
        geo=geo,
    )
    core_profiles3_helper = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config3_helper),
        config_slice.build_dynamic_config_slice(config3_helper),
        geo=geo,
    )

    # calculate total and Ohmic current profiles arising from nu=2
    jformula_face = (1 - geo.r_face_norm**2) ** 2
    denom = jax.scipy.integrate.trapezoid(
        jformula_face * geo.spr_face, geo.r_face
    )
    ctot = config1.profile_conditions.Ip * 1e6 / denom
    jtot_formula_face = jformula_face * ctot
    johm_formula_face = jtot_formula_face * (1 - config1.fext)

    # Calculate bootstrap current for config3 which doesn't zero it out
    source_models = source_models_lib.SourceModels()
    bootstrap_profile = source_models.j_bootstrap.get_value(
        dynamic_config_slice=config_slice.build_dynamic_config_slice(config3),
        geo=geo,
        temp_ion=core_profiles3.temp_ion,
        temp_el=core_profiles3.temp_el,
        ne=core_profiles3.ne,
        ni=core_profiles3.ni,
        jtot_face=core_profiles3_helper.currents.jtot_face,
        psi=core_profiles3_helper.psi,
    )
    f_bootstrap = bootstrap_profile.I_bootstrap / (
        config3.profile_conditions.Ip * 1e6
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles1.currents.jtot,
        core_profiles2.currents.jtot,
    )

    np.testing.assert_allclose(
        core_profiles1.currents.jext_face + core_profiles1.currents.johm_face,
        jtot_formula_face,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles1.currents.johm_face,
        johm_formula_face,
    )
    np.testing.assert_allclose(
        core_profiles2.currents.johm_face,
        johm_formula_face,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        core_profiles2.currents.jtot_face,
        jtot_formula_face,
    )
    np.testing.assert_allclose(
        core_profiles3.currents.johm_face,
        jtot_formula_face * (1 - f_bootstrap),
        rtol=1e-12,
        atol=1e-12,
    )

  def test_initial_psi_from_geo_noop_circular(self):
    """Tests expected behaviour of initial psi and current options."""
    config1 = config_lib.Config(
        initial_psi_from_j=False,
    )
    config2 = config_lib.Config(
        initial_psi_from_j=True,
    )
    core_profiles1 = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config1),
        config_slice.build_dynamic_config_slice(config1),
        geometry.build_circular_geometry(config1),
    )
    core_profiles2 = core_profile_setters.initial_core_profiles(
        config_slice.build_static_config_slice(config2),
        config_slice.build_dynamic_config_slice(config2),
        geometry.build_circular_geometry(config2),
    )
    np.testing.assert_allclose(
        core_profiles1.currents.jtot, core_profiles2.currents.jtot
    )


if __name__ == '__main__':
  absltest.main()
