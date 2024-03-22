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

"""Tests for bootstrap_current_source."""

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax import config as config_lib
from torax import config_slice
from torax import geometry
from torax import initial_states
from torax.sources import bootstrap_current_source
from torax.sources import source as source_lib
from torax.sources import source_config
from torax.sources import source_profiles
from torax.sources.tests import test_lib
from torax.time_step_calculator import fixed_time_step_calculator


class BootstrapCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for BootstrapCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=bootstrap_current_source.BootstrapCurrentSource,
        unsupported_types=[
            source_config.SourceType.FORMULA_BASED,
            source_config.SourceType.ZERO,
        ],
        expected_affected_mesh_states=(
            source_lib.AffectedMeshStateAttribute.PSI,
        ),
    )

  def test_source_value(self):
    source = bootstrap_current_source.BootstrapCurrentSource()
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    ts_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    sim_state = initial_states.get_initial_sim_state(
        config=config,
        geo=geo,
        time_step_calculator=ts_calculator,
        sources=source_profiles.Sources(j_bootstrap=source),
    )
    self.assertIsNotNone(
        source.get_value(
            dynamic_config_slice=(
                config_slice.build_dynamic_config_slice(config)
            ),
            geo=geo,
            temp_ion=sim_state.mesh_state.temp_ion,
            temp_el=sim_state.mesh_state.temp_el,
            ne=sim_state.mesh_state.ne,
            ni=sim_state.mesh_state.ni,
            jtot_face=sim_state.mesh_state.currents.jtot_face,
            psi=sim_state.mesh_state.psi,
        )
    )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    source = bootstrap_current_source.BootstrapCurrentSource()
    config = config_lib.Config()
    geo = geometry.build_circular_geometry(config)
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    face = source_lib.ProfileType.FACE.get_profile_shape(geo)
    fake_profile = bootstrap_current_source.BootstrapCurrentProfile(
        sigma=jnp.zeros(cell),
        j_bootstrap=jnp.ones(cell),
        j_bootstrap_face=jnp.zeros(face),
        I_bootstrap=jnp.zeros((1,)),
    )
    np.testing.assert_allclose(
        source.get_profile_for_affected_state(
            fake_profile,
            source_lib.AffectedMeshStateAttribute.PSI.value,
            geo,
        ),
        jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_profile_for_affected_state(
            fake_profile,
            source_lib.AffectedMeshStateAttribute.TEMP_ION.value,
            geo,
        ),
        jnp.zeros(cell),
    )


if __name__ == '__main__':
  absltest.main()
