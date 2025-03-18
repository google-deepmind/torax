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

<<<<<<< HEAD:torax/transport_model/tests/bohm_gyrobohm_test.py
from absl.testing import absltest
import jax.numpy as jnp

from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.transport_model import bohm_gyrobohm
from torax import state


class RuntimeParamsTest(absltest.TestCase):

  def test_runtime_params_builds_dynamic_params(self):
    runtime_params = bohm_gyrobohm.RuntimeParams()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = runtime_params.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)


class BohmGyroBohmOutputTest(absltest.TestCase):

  def test_output_includes_new_fields(self):
    # Use the CircularConfig to build a geometry.
    geo = geometry_pydantic_model.CircularConfig().build_geometry()

    # Build dynamic runtime parameters.
    runtime_params = bohm_gyrobohm.RuntimeParams()
    provider = runtime_params.make_provider(geo.torax_mesh)
    dynamic_params = provider.build_dynamic_params(t=0.0)

    # Create dummy core profiles with the necessary attributes.
    dummy_shape = geo.rho_face.shape

    class DummyProfile:
      def __init__(self, value, grad):
        self._value = value
        self._grad = grad

      def face_value(self):
        return self._value

      def face_grad(self):
        return self._grad

    class DummyCoreProfiles:
      def __init__(self):
        # Create dummy arrays with the same shape as geo.rho_face.
        self.ne = DummyProfile(jnp.ones(dummy_shape), jnp.full(dummy_shape, 0.1))
        self.temp_el = DummyProfile(jnp.full(dummy_shape, 2.0), jnp.full(dummy_shape, 0.2))
        self.q_face = jnp.full(dummy_shape, 2.0)

    dummy_core_profiles = DummyCoreProfiles()

    # Create a dummy dynamic runtime parameters slice.
    # This object needs attributes: transport, numerics (with nref), and plasma_composition.
    class DummyNumerics:
      nref = 1.0

    class DummyMainIon:
      avg_A = 2.0

    class DummyPlasmaComposition:
      main_ion = DummyMainIon()

    class DummySlice:
      pass

    dummy_slice = DummySlice()
    dummy_slice.transport = dynamic_params
    dummy_slice.numerics = DummyNumerics()
    dummy_slice.plasma_composition = DummyPlasmaComposition()

    # Instantiate and call the BohmGyroBohm model.
    model = bohm_gyrobohm.BohmGyroBohmTransportModel()
    transport_out = model._call_implementation(
        dynamic_runtime_params_slice=dummy_slice,
        geo=geo,
        core_profiles=dummy_core_profiles,
        pedestal_model_outputs=None,
    )

    # Verify the new fields are set and have the expected shape.
    self.assertIsNotNone(transport_out.chi_e_bohm)
    self.assertIsNotNone(transport_out.chi_e_gyrobohm)
    self.assertIsNotNone(transport_out.chi_i_bohm)
    self.assertIsNotNone(transport_out.chi_i_gyrobohm)
    self.assertEqual(transport_out.chi_e_bohm.shape, dummy_shape)
    self.assertEqual(transport_out.chi_e_gyrobohm.shape, dummy_shape)
    self.assertEqual(transport_out.chi_i_bohm.shape, dummy_shape)
    self.assertEqual(transport_out.chi_i_gyrobohm.shape, dummy_shape)


if __name__ == '__main__':
  absltest.main()
=======
"""Transport model tests."""
>>>>>>> origin/main:torax/transport_model/tests/__init__.py
