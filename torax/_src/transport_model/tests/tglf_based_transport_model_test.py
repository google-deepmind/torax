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
from collections.abc import Mapping
import dataclasses
from typing import Annotated, Any, Literal

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib


def _get_config_and_model_inputs(
    transport: Mapping[str, Any],
):
  """Returns the model inputs for testing."""
  config = default_configs.get_default_config_dict()
  config["transport"] = transport
  torax_config = model_config.ToraxConfig.from_dict(config)
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
      torax_config
  )(
      t=torax_config.numerics.t_initial,
  )
  geo = torax_config.geometry.build_provider(t=torax_config.numerics.t_initial)
  core_profiles = initialization.initial_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
  )
  return torax_config, (runtime_params, geo, core_profiles)


class TGLFTransportModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    model_config.ToraxConfig.model_fields[
        "transport"
    ].annotation |= TGLFBasedTransportModelConfig
    model_config.ToraxConfig.model_rebuild(force=True)

  def test_tglf_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        "model_name": "tglf_based",
    })
    transport_model = torax_config.transport.build_transport_model()
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_policy = torax_config.pedestal.set_pedestal.build_pedestal_policy()
    runtime_params, geo, core_profiles = model_inputs
    pedestal_policy_state = pedestal_policy.initial_state(
        t=torax_config.numerics.t_initial,
        runtime_params=runtime_params.pedestal_policy,
    )
    pedestal_model_outputs = pedestal_model(
        runtime_params, geo, core_profiles, pedestal_policy_state
    )

    core_transport = transport_model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_policy_state,
        pedestal_model_outputs,
    )
    expected_shape = model_inputs[1].rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  def test_tglf_based_transport_model_prepare_tglf_inputs_shapes(self):
    """Tests that the tglf inputs have the expected shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        "model_name": "tglf_based",
    })
    transport_model = torax_config.transport.build_transport_model()
    runtime_params, geo, core_profiles = model_inputs
    assert isinstance(
        runtime_params.transport,
        tglf_based_transport_model.RuntimeParams,
    )
    tglf_inputs = transport_model.prepare_tglf_inputs(
        transport=runtime_params.transport,
        geo=geo,
        core_profiles=core_profiles,
    )

    vector_keys = [
        "chiGB",
        "lref_over_lti",
        "lref_over_lte",
        "lref_over_lne",
        "lref_over_lni0",
        "lref_over_lni1",
        "Ti_over_Te",
        "r_minor",
        "dr_major",
        "q",
        "q_prime",
        "nu_ee",
        "kappa",
        "kappa_shear",
        "delta",
        "delta_shear",
        "beta_e",
        "Zeff",
    ]
    scalar_keys = ["Rmaj", "Rmin"]
    expected_vector_length = geo.rho_face_norm.shape[0]
    for key in vector_keys:
      self.assertEqual(
          getattr(tglf_inputs, key).shape, (expected_vector_length,)
      )
    for key in scalar_keys:
      self.assertEqual(getattr(tglf_inputs, key).shape, ())


@dataclasses.dataclass(frozen=True, eq=False)
class FakeTGLFBasedTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Fake TGLFBasedTransportModel for testing purposes."""

  # pylint: disable=invalid-name
  def prepare_tglf_inputs(
      self,
      transport: tglf_based_transport_model.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> tglf_based_transport_model.TGLFInputs:
    """Exposing prepare_tglf_inputs for testing."""
    return self._prepare_tglf_inputs(transport, geo, core_profiles)

  # pylint: enable=invalid-name

  def _call_implementation(
      self,
      transport_runtime_params: tglf_based_transport_model.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    # Assert required for pytype.
    assert isinstance(
        transport_runtime_params,
        tglf_based_transport_model.RuntimeParams,
    )

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )
    return self._make_core_transport(
        ion_heat_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 0.4,
        electron_heat_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 0.5,
        electron_particle_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 1.6,
        tglf_inputs=tglf_inputs,
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
    )


# pylint: disable=invalid-name
class TGLFBasedTransportModelConfig(
    transport_pydantic_model_base.TransportBase
):
  """Model for testing the TGLF-based transport model."""

  model_name: Annotated[Literal["tglf_based"], torax_pydantic.JAX_STATIC] = (
      "tglf_based"
  )

  # pylint: disable=undefined-variable
  def build_transport_model(
      self,
  ) -> FakeTGLFBasedTransportModel:
    return FakeTGLFBasedTransportModel()

  def build_runtime_params(self, t: chex.Numeric):
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return tglf_based_transport_model.RuntimeParams(
        # DV_effective and An_min are inherited from QuasilinearTransportModel
        DV_effective=False,
        An_min=0.05,
        **base_kwargs,
    )


# pylint: enable=undefined-variable


if __name__ == "__main__":
  absltest.main()
