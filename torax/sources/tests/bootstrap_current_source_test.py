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
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax import state
from torax.config import numerics
from torax.config import plasma_composition
from torax.config import runtime_params_slice
from torax.fvm import cell_variable
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import bootstrap_current_source
from torax.sources import runtime_params
from torax.sources import source_profiles


class BootstrapCurrentSourceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    n_rho = 10
    self.source_name = (
        bootstrap_current_source.BootstrapCurrentSource.SOURCE_NAME
    )
    self.geo = geometry_pydantic_model.CircularConfig(
        n_rho=n_rho
    ).build_geometry()
    dynamic_bootstap_params = bootstrap_current_source.DynamicRuntimeParams(
        prescribed_values=mock.ANY,
        bootstrap_mult=1.0,
    )
    self.dynamic_params = mock.create_autospec(
        runtime_params_slice.DynamicRuntimeParamsSlice,
        instance=True,
        sources={self.source_name: dynamic_bootstap_params},
        plasma_composition=mock.create_autospec(
            plasma_composition.PlasmaComposition,
            instance=True,
            Zeff_face=jnp.ones_like(self.geo.rho_face),
        ),
        numerics=mock.create_autospec(
            numerics.Numerics,
            instance=True,
            nref=100,
        ),
    )
    self.core_profiles = mock.create_autospec(
        state.CoreProfiles,
        temp_ion=cell_variable.CellVariable(
            value=jnp.linspace(400, 700, n_rho), dr=self.geo.drho_norm
        ),
        temp_el=cell_variable.CellVariable(
            value=jnp.linspace(4000, 7000, n_rho), dr=self.geo.drho_norm
        ),
        psi=cell_variable.CellVariable(
            value=jnp.linspace(9000, 4000, n_rho), dr=self.geo.drho_norm
        ),
        ne=cell_variable.CellVariable(
            value=jnp.linspace(100, 200, n_rho), dr=self.geo.drho_norm
        ),
        ni=cell_variable.CellVariable(
            value=jnp.linspace(100, 200, n_rho), dr=self.geo.drho_norm
        ),
        Zi_face=np.linspace(1000, 2000, n_rho + 1),
        q_face=np.linspace(1, 5, n_rho + 1),
    )

  def test_get_bootstrap(self):
    source = bootstrap_current_source.BootstrapCurrentSource()

    static_bootstap_params = runtime_params.StaticRuntimeParams(
        mode=runtime_params.Mode.MODEL_BASED.value,
        is_explicit=False,
    )

    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={source.SOURCE_NAME: static_bootstap_params},
    )
    bootstrap_profile = source.get_bootstrap(
        dynamic_runtime_params_slice=self.dynamic_params,
        static_runtime_params_slice=static_params,
        geo=self.geo,
        core_profiles=self.core_profiles,
    )
    self.assertIsInstance(
        bootstrap_profile, source_profiles.BootstrapCurrentProfile
    )
    self.assertEqual(bootstrap_profile.sigma.shape, self.geo.rho.shape)
    self.assertEqual(
        bootstrap_profile.sigma_face.shape, self.geo.rho_face.shape
    )
    self.assertEqual(bootstrap_profile.j_bootstrap.shape, self.geo.rho.shape)
    self.assertEqual(
        bootstrap_profile.j_bootstrap_face.shape, self.geo.rho_face.shape
    )
    self.assertEqual(bootstrap_profile.I_bootstrap.shape, ())

    # Check that the values aren't zeros
    self.assertFalse(jnp.all(bootstrap_profile.sigma == 0))
    self.assertFalse(jnp.all(bootstrap_profile.sigma_face == 0))
    self.assertFalse(jnp.all(bootstrap_profile.j_bootstrap == 0))
    self.assertFalse(jnp.all(bootstrap_profile.j_bootstrap_face == 0))
    self.assertFalse(jnp.all(bootstrap_profile.I_bootstrap == 0))

  def test_get_bootstrap_with_zero_mode(self):
    source = bootstrap_current_source.BootstrapCurrentSource()
    static_bootstap_params = runtime_params.StaticRuntimeParams(
        mode=runtime_params.Mode.ZERO.value,
        is_explicit=False,
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={source.SOURCE_NAME: static_bootstap_params},
    )
    bootstrap_profile = source.get_bootstrap(
        self.dynamic_params,
        static_params,
        self.geo,
        self.core_profiles,
    )
    self.assertIsInstance(
        bootstrap_profile, source_profiles.BootstrapCurrentProfile
    )
    self.assertEqual(bootstrap_profile.sigma.shape, self.geo.rho.shape)
    self.assertEqual(
        bootstrap_profile.sigma_face.shape, self.geo.rho_face.shape
    )
    self.assertEqual(bootstrap_profile.j_bootstrap.shape, self.geo.rho.shape)
    self.assertEqual(
        bootstrap_profile.j_bootstrap_face.shape, self.geo.rho_face.shape
    )
    self.assertEqual(bootstrap_profile.I_bootstrap.shape, ())

    # Check that the sigma values aren't zeros
    self.assertFalse(jnp.all(bootstrap_profile.sigma == 0))
    self.assertFalse(jnp.all(bootstrap_profile.sigma_face == 0))
    # Check that the j_bootstrap values are zeros
    self.assertTrue(jnp.all(bootstrap_profile.j_bootstrap == 0))
    self.assertTrue(jnp.all(bootstrap_profile.j_bootstrap_face == 0))
    self.assertTrue(jnp.all(bootstrap_profile.I_bootstrap == 0))

  def test_prescribed_mode_not_supported(self):
    source = bootstrap_current_source.BootstrapCurrentSource()
    static_bootstap_params = runtime_params.StaticRuntimeParams(
        mode=runtime_params.Mode.PRESCRIBED.value,
        is_explicit=False,
    )
    static_params = mock.create_autospec(
        runtime_params_slice.StaticRuntimeParamsSlice,
        sources={source.SOURCE_NAME: static_bootstap_params},
    )
    with self.assertRaisesRegex(NotImplementedError, 'Prescribed mode'):
      source.get_bootstrap(
          self.dynamic_params,
          static_params,
          mock.ANY,
          mock.ANY,
      )

  def test_raise_error_on_get_value(self):
    source = bootstrap_current_source.BootstrapCurrentSource()
    with self.assertRaisesRegex(
        NotImplementedError, 'Call `get_bootstrap` instead.'
    ):
      source.get_value(
          mock.ANY,
          mock.ANY,
          mock.ANY,
          mock.ANY,
          None,
      )

  def test_raise_error_on_get_source_profile_for_affected_core_profile(self):
    source = bootstrap_current_source.BootstrapCurrentSource()
    with self.assertRaisesRegex(
        NotImplementedError, 'Call `get_bootstrap` instead.'
    ):
      source.get_source_profile_for_affected_core_profile(
          mock.ANY,
          mock.ANY,
          mock.ANY,
      )


if __name__ == '__main__':
  absltest.main()
