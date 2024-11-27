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
from torax import geometry
from torax.sources import bootstrap_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source as source_lib
from torax.sources import source_profiles
from torax.sources.tests import test_lib


class BootstrapCurrentSourceTest(test_lib.SourceTestCase):
  """Tests for BootstrapCurrentSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=bootstrap_current_source.BootstrapCurrentSource,
        runtime_params_class=bootstrap_current_source.RuntimeParams,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
        expected_affected_core_profiles=(source_lib.AffectedCoreProfile.PSI,),
    )

  def test_expected_mesh_states(self):
    # This function is reimplemented here as BootstrapCurrentSource does not
    # appear in source_models, which the parent class uses to build the source
    source = bootstrap_current_source.BootstrapCurrentSource()
    self.assertSameElements(
        source.affected_core_profiles,
        self._expected_affected_core_profiles,
    )

  def test_extraction_of_relevant_profile_from_output(self):
    """Tests that the relevant profile is extracted from the output."""
    source = bootstrap_current_source.BootstrapCurrentSource()
    geo = geometry.build_circular_geometry()
    cell = source_lib.ProfileType.CELL.get_profile_shape(geo)
    face = source_lib.ProfileType.FACE.get_profile_shape(geo)
    fake_profile = source_profiles.BootstrapCurrentProfile(
        sigma=jnp.zeros(cell),
        sigma_face=jnp.zeros(face),
        j_bootstrap=jnp.ones(cell),
        j_bootstrap_face=jnp.zeros(face),
        I_bootstrap=jnp.zeros((1,)),
    )
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.PSI.value,
            geo,
        ),
        jnp.ones(cell),
    )
    # For unrelated states, this should just return all 0s.
    np.testing.assert_allclose(
        source.get_source_profile_for_affected_core_profile(
            fake_profile,
            source_lib.AffectedCoreProfile.TEMP_ION.value,
            geo,
        ),
        jnp.zeros(cell),
    )


if __name__ == '__main__':
  absltest.main()
