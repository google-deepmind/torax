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
from absl.testing import parameterized
import numpy as np
from torax import jax_utils
from torax.config import build_runtime_params
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.core_profiles import initialization
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.sources import generic_current_source
from torax.sources import pydantic_model as source_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.sources import source_profiles
from torax.stepper import pydantic_model as stepper_pydantic_model
from torax.tests.test_lib import torax_refs
from torax.transport_model import runtime_params as transport_params_lib


class InitializationTest(torax_refs.ReferenceValueTest):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)
    self.geo = geometry_pydantic_model.CircularConfig(n_rho=4).build_geometry()

  def test_update_psi_from_j(self):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = torax_refs.circular_references()

    runtime_params = references.runtime_params
    # Turn on the external current source.
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
            sources=source_pydantic_model.Sources.from_dict(
                {"generic_current_source": {}}
            ),
        )
    )
    bootstrap = source_profiles.BootstrapCurrentProfile.zero_profile(geo)
    external_current = generic_current_source.calculate_generic_current(
        mock.ANY,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
        unused_state=mock.ANY,
        unused_calculated_source_profiles=mock.ANY,
    )[0]
    currents = initialization._prescribe_currents(
        bootstrap_profile=bootstrap,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = initialization._update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip_tot,
        geo,
        currents.jtot_hires,
    ).value
    np.testing.assert_allclose(psi, references.psi.value)

  @parameterized.parameters(
      (
          {0.0: {0.0: 0.0, 1.0: 1.0}},
          np.array([0.125, 0.375, 0.625, 0.875]),
      ),
  )
  def test_initial_psi(
      self,
      psi,
      expected_psi,
  ):
    """Tests that runtime params validate boundary conditions."""
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            psi=psi,
        )
    )
    sources = source_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    provider = build_runtime_params.DynamicRuntimeParamsSliceProvider(
        runtime_params=runtime_params,
        transport=transport_params_lib.RuntimeParams(),
        sources=sources,
        stepper=stepper_pydantic_model.Stepper(),
        torax_mesh=self.geo.torax_mesh,
    )
    dynamic_runtime_params_slice = provider(
        t=1.0,
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=self.geo.torax_mesh,
    )
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        self.geo,
        source_models,
    )

    np.testing.assert_allclose(
        core_profiles.psi.value, expected_psi, atol=1e-6, rtol=1e-6
    )


if __name__ == "__main__":
  absltest.main()
