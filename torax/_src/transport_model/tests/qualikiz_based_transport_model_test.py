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
from typing import Any, Literal

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax._src.transport_model import qualikiz_based_transport_model
from torax._src.transport_model import quasilinear_transport_model
from torax.tests.test_lib import default_configs


def _get_config_and_model_inputs(
    transport: Mapping[str, Any],
):
  """Returns the model inputs for testing."""
  config = default_configs.get_default_config_dict()
  config['transport'] = transport
  torax_config = model_config.ToraxConfig.from_dict(config)
  source_models = source_models_lib.SourceModels(
      sources=torax_config.sources, neoclassical=torax_config.neoclassical
  )
  dynamic_runtime_params_slice = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
          torax_config
      )(
          t=torax_config.numerics.t_initial,
      )
  )
  geo = torax_config.geometry.build_provider(t=torax_config.numerics.t_initial)
  static_slice = build_runtime_params.build_static_params_from_config(
      torax_config
  )
  core_profiles = initialization.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_slice,
      geo=geo,
      source_models=source_models,
  )
  return torax_config, (
      dynamic_runtime_params_slice, geo, core_profiles)


class QualikizTransportModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    model_config.ToraxConfig.model_fields['transport'].annotation |= (
        QualikizBasedTransportModelConfig
    )
    model_config.ToraxConfig.model_rebuild(force=True)

  def test_qualikiz_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs(
        {
            'model_name': 'qualikiz_based',
            'collisionality_multiplier': 1.0,
            'avoid_big_negative_s': True,
            'q_sawtooth_proxy': True,
        }
    )
    transport_model = torax_config.transport.build_transport_model()
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        *model_inputs
    )

    core_transport = transport_model(
        *model_inputs, pedestal_model_outputs
    )
    expected_shape = model_inputs[1].rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  def test_qualikiz_based_transport_model_prepare_qualikiz_inputs_shapes(self):
    """Tests that the qualikiz inputs have the expected shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs(
        {
            'model_name': 'qualikiz_based',
            'collisionality_multiplier': 1.0,
            'avoid_big_negative_s': True,
            'q_sawtooth_proxy': True,
            'smag_alpha_correction': True,
        }
    )
    transport_model = torax_config.transport.build_transport_model()
    dynamic_runtime_params_slice, geo, core_profiles = model_inputs
    assert isinstance(
        dynamic_runtime_params_slice.transport,
        qualikiz_based_transport_model.DynamicRuntimeParams,
    )
    qualikiz_inputs = transport_model.prepare_qualikiz_inputs(
        Z_eff_face=dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
        density_reference=dynamic_runtime_params_slice.numerics.density_reference,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
    )

    # 1D array qualikiz_inputs
    vector_keys = [
        'Z_eff_face',
        'Ati',
        'Ate',
        'Ane',
        'Ani0',
        'Ani1',
        'q',
        'smag',
        'x',
        'Ti_Te',
        'log_nu_star_face',
        'normni',
        'chiGB',
    ]
    scalar_keys = ['Rmaj', 'Rmin']
    expected_vector_length = geo.rho_face_norm.shape[0]
    for key in vector_keys:
      self.assertEqual(
          getattr(qualikiz_inputs, key).shape, (expected_vector_length,)
      )
    for key in scalar_keys:
      self.assertEqual(getattr(qualikiz_inputs, key).shape, ())


class FakeQualikizBasedTransportModel(
    qualikiz_based_transport_model.QualikizBasedTransportModel
):
  """Fake QualikizBasedTransportModel for testing purposes."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  # pylint: disable=invalid-name
  def prepare_qualikiz_inputs(
      self,
      Z_eff_face: chex.Array,
      density_reference: chex.Numeric,
      transport: qualikiz_based_transport_model.DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> qualikiz_based_transport_model.QualikizInputs:
    """Exposing prepare_qualikiz_inputs for testing."""
    return self._prepare_qualikiz_inputs(
        Z_eff_face, density_reference, transport, geo, core_profiles
    )

  # pylint: enable=invalid-name

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> state.CoreTransport:
    transport = dynamic_runtime_params_slice.transport
    # Assert required for pytype.
    assert isinstance(
        transport,
        qualikiz_based_transport_model.DynamicRuntimeParams,
    )
    qualikiz_inputs = self._prepare_qualikiz_inputs(
        Z_eff_face=dynamic_runtime_params_slice.plasma_composition.Z_eff_face,
        density_reference=dynamic_runtime_params_slice.numerics.density_reference,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
    )
    # Assert required for pytype.
    assert isinstance(
        transport,
        quasilinear_transport_model.DynamicRuntimeParams,
    )
    return self._make_core_transport(
        qi=jnp.ones(geo.rho_face_norm.shape) * 0.4,
        qe=jnp.ones(geo.rho_face_norm.shape) * 0.5,
        pfe=jnp.ones(geo.rho_face_norm.shape) * 1.6,
        quasilinear_inputs=qualikiz_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        gradient_reference_length=geo.R_major,
        gyrobohm_flux_reference_length=geo.a_minor,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))


# pylint: disable=invalid-name
class QualikizBasedTransportModelConfig(
    transport_pydantic_model_base.TransportBase
):
  """Model for the Qualikiz-based transport model.

  Attributes:
    transport_model: The transport model to use. Hardcoded to 'qualikiz'.
    collisionality_multiplier: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DV_effective: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """

  model_name: Literal['qualikiz_based'] = 'qualikiz_based'
  collisionality_multiplier: pydantic.PositiveFloat = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  # pylint: disable=undefined-variable
  def build_transport_model(
      self,
  ) -> FakeQualikizBasedTransportModel:
    return FakeQualikizBasedTransportModel()

  def build_dynamic_params(self, t: chex.Numeric):
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return qualikiz_based_transport_model.DynamicRuntimeParams(
        collisionality_multiplier=self.collisionality_multiplier,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        **base_kwargs,
    )
# pylint: enable=undefined-variable


if __name__ == '__main__':
  absltest.main()
