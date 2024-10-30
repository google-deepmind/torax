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

"""Unit tests for torax.transport_model.quasilinear_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from torax import core_profile_setters
from torax import geometry
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.transport_model import quasilinear_utils
from torax.transport_model import runtime_params as runtime_params_lib


class QuasilinearUtilsTest(parameterized.TestCase):
  """Unit tests for the `torax.transport_model.quasilinear_utils` module."""

  def test_make_core_transport(self):
    """Tests that the model output is properly converted to core transport."""
    runtime_params = general_runtime_params.GeneralRuntimeParams()
    geo = geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()

    source_models = source_models_builder()
    dynamic_runtime_params_slice = (
        runtime_params_slice.DynamicRuntimeParamsSliceProvider(
            runtime_params=runtime_params,
            transport=runtime_params_lib.RuntimeParams(),
            sources=source_models_builder.runtime_params,
            torax_mesh=geo.torax_mesh,
        )(
            t=runtime_params.numerics.t_initial,
        )
    )

    core_profiles = core_profile_setters.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )
    quasilinear_inputs = quasilinear_utils.QuasilinearInputs(
        chiGB=jnp.array(1.0),
        Rmin=jnp.array(0.5),
        Rmaj=jnp.array(2.0),
        Ati=jnp.array(1.0),
        Ate=jnp.array(1.0),
        Ane=jnp.array(1.0),
    )
    expected_shape = (26,)
    qi = jnp.ones(expected_shape)
    qe = jnp.zeros(expected_shape)
    pfe = jnp.zeros(expected_shape)
    transport = quasilinear_utils.QuasilinearDynamicRuntimeParams(
        DVeff=False, An_min=0.05, **runtime_params_lib.RuntimeParams()
    )
    core_transport = quasilinear_utils.make_core_transport(
        qi=qi,
        qe=qe,
        pfe=pfe,
        quasilinear_inputs=quasilinear_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
