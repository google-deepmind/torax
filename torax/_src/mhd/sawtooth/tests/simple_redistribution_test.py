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

"""Tests for the simple_redistribution module."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.mhd.sawtooth import simple_redistribution
from torax._src.physics import psi_calculations
from torax._src.torax_pydantic import model_config

# Set jax_enable_x64 to True to ensure high precision for tests.
jax.config.update('jax_enable_x64', True)


class SimpleRedistributionTest(parameterized.TestCase):

  @parameterized.product(
      evolve_ion_heat=[True, False],
      evolve_electron_heat=[True, False],
      evolve_density=[True, False],
  )
  def test_simple_redistribution_with_evolving_profiles(
      self, evolve_ion_heat, evolve_electron_heat, evolve_density
  ):
    """Tests that SimpleRedistribution works with all evolving profiles."""
    config_dict = {
        'numerics': {
            'evolve_ion_heat': evolve_ion_heat,
            'evolve_electron_heat': evolve_electron_heat,
            'evolve_density': evolve_density,
            'evolve_current': True,
        },
        'profile_conditions': {  # Set up to ensure q[0] < 1
            'Ip': 15e6,
            'initial_j_is_total_current': True,
            'initial_psi_from_j': True,
            'current_profile_nu': 3,
        },
        'plasma_composition': {},
        'geometry': {'geometry_type': 'circular', 'n_rho': 10},
        'pedestal': {},
        'sources': {},
        'solver': {},
        'transport': {},
        'mhd': {
            'sawtooth': {
                'trigger_model': {'model_name': 'simple'},
                'redistribution_model': {
                    'model_name': 'simple',
                    'flattening_factor': 1.01,
                    'mixing_radius_multiplier': 1.5,
                },
            }
        },
    }

    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    assert torax_config.mhd is not None
    assert torax_config.mhd.sawtooth is not None

    redistribution_model = (
        torax_config.mhd.sawtooth.redistribution_model.build_redistribution_model()
    )
    self.assertIsInstance(
        redistribution_model, simple_redistribution.SimpleRedistribution
    )
    runtime_params_provider = (
        build_runtime_params.RuntimeParamsProvider.from_config(torax_config)
    )
    geo_provider = torax_config.geometry.build_provider

    runtime_params_t = runtime_params_provider(t=0.0)
    geo_t = geo_provider(t=0.0)

    core_profiles_t = initialization.initial_core_profiles(
        runtime_params=runtime_params_t,
        geo=geo_t,
        source_models=torax_config.sources.build_models(),
        neoclassical_models=torax_config.neoclassical.build_models(),
    )

    # Find the q=1 surface radius to pass to the model
    q_face_before = core_profiles_t.q_face
    self.assertLess(
        q_face_before[0],
        1.0,
        'Initial q-profile must be below 1 for this test.',
    )
    rho_norm_q1 = np.interp(
        1.0,
        q_face_before,
        geo_t.rho_face_norm,
    )

    # Call the redistribution model
    redistributed_core_profiles = redistribution_model(
        jnp.asarray(rho_norm_q1),
        runtime_params_t,
        geo_t,
        core_profiles_t,
    )

    # Main check: Ensure no errors were raised.
    # Also, perform a basic check to ensure redistribution occurred.
    q_face_after = psi_calculations.calc_q_face(
        geo_t, redistributed_core_profiles.psi
    )
    self.assertGreater(
        q_face_after[0],
        q_face_before[0],
        'On-axis q should increase after redistribution.',
    )
    self.assertGreaterEqual(
        q_face_after[0],
        1.0,
        'On-axis q should be at least 1.0 after redistribution.',
    )


if __name__ == '__main__':
  absltest.main()
