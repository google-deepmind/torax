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
import dataclasses
from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
import pydantic
from torax import state
from torax.config import build_runtime_params
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.transport_model import pydantic_model as transport_pydantic_model
from torax.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax.transport_model import qualikiz_based_transport_model
from torax.transport_model import quasilinear_transport_model


def _get_model_inputs(
    transport: transport_pydantic_model.Transport,
):
  """Returns the model inputs for testing."""
  runtime_params = general_runtime_params.GeneralRuntimeParams()
  geo = geometry_pydantic_model.CircularConfig().build_geometry()
  sources = sources_pydantic_model.Sources()
  source_models = source_models_lib.SourceModels(
      sources=sources.source_model_config
  )
  pedestal = pedestal_pydantic_model.Pedestal()
  dynamic_runtime_params_slice = (
      build_runtime_params.DynamicRuntimeParamsSliceProvider(
          runtime_params=runtime_params,
          transport=transport,
          sources=sources,
          pedestal=pedestal,
          torax_mesh=geo.torax_mesh,
      )(
          t=runtime_params.numerics.t_initial,
      )
  )
  static_slice = build_runtime_params.build_static_runtime_params_slice(
      runtime_params=runtime_params,
      sources=sources,
      torax_mesh=geo.torax_mesh,
  )
  core_profiles = initialization.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_runtime_params_slice,
      static_runtime_params_slice=static_slice,
      geo=geo,
      source_models=source_models,
  )
  return dynamic_runtime_params_slice, geo, core_profiles


class QualikizTransportModelTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    transport_pydantic_model.Transport.model_fields[
        'transport_model_config'
    ].annotation |= QualikizBasedTransportModelConfig
    transport_pydantic_model.Transport.model_rebuild(force=True)

  def test_qualikiz_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    transport = transport_pydantic_model.Transport.from_dict({
        'transport_model': 'qualikiz_based',
        'coll_mult': 1.0,
        'avoid_big_negative_s': True,
        'q_sawtooth_proxy': True,
    })
    transport_model = transport.build_transport_model()
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    pedestal = pedestal_pydantic_model.Pedestal()
    pedestal_model = pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )

    core_transport = transport_model(
        dynamic_runtime_params_slice, geo, core_profiles, pedestal_model_outputs
    )
    expected_shape = geo.rho_face_norm.shape
    self.assertEqual(core_transport.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.v_face_el.shape, expected_shape)

  def test_qualikiz_based_transport_model_prepare_qualikiz_inputs_shapes(self):
    """Tests that the qualikiz inputs have the expected shapes."""
    transport = transport_pydantic_model.Transport.from_dict({
        'transport_model': 'qualikiz_based',
        'coll_mult': 1.0,
        'avoid_big_negative_s': True,
        'q_sawtooth_proxy': True,
        'smag_alpha_correction': True,
    })
    dynamic_runtime_params_slice, geo, core_profiles = _get_model_inputs(
        transport
    )
    transport_model = transport.build_transport_model()
    assert isinstance(
        dynamic_runtime_params_slice.transport,
        qualikiz_based_transport_model.DynamicRuntimeParams,
    )
    qualikiz_inputs = transport_model.prepare_qualikiz_inputs(
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        nref=dynamic_runtime_params_slice.numerics.nref,
        transport=dynamic_runtime_params_slice.transport,
        geo=geo,
        core_profiles=core_profiles,
    )

    # 1D array qualikiz_inputs
    vector_keys = [
        'Zeff_face',
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
      Zeff_face: chex.Array,
      nref: chex.Numeric,
      transport: qualikiz_based_transport_model.DynamicRuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> qualikiz_based_transport_model.QualikizInputs:
    """Exposing prepare_qualikiz_inputs for testing."""
    return self._prepare_qualikiz_inputs(
        Zeff_face, nref, transport, geo, core_profiles
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
        Zeff_face=dynamic_runtime_params_slice.plasma_composition.Zeff_face,
        nref=dynamic_runtime_params_slice.numerics.nref,
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
        gradient_reference_length=geo.Rmaj,
        gyrobohm_flux_reference_length=geo.Rmin,
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
    coll_mult: Collisionality multiplier.
    avoid_big_negative_s: Ensure that smag - alpha > -0.2 always, to compensate
      for no slab modes.
    smag_alpha_correction: Reduce magnetic shear by 0.5*alpha to capture main
      impact of alpha.
    q_sawtooth_proxy: If q < 1, modify input q and smag as if q~1 as if there
      are sawteeth.
    DVeff: Effective D / effective V approach for particle transport.
    An_min: Minimum |R/Lne| below which effective V is used instead of effective
      D.
  """

  transport_model: Literal['qualikiz_based'] = 'qualikiz_based'
  coll_mult: pydantic.PositiveFloat = 1.0
  avoid_big_negative_s: bool = True
  smag_alpha_correction: bool = True
  q_sawtooth_proxy: bool = True
  DVeff: bool = False
  An_min: pydantic.PositiveFloat = 0.05

  # pylint: disable=undefined-variable
  def build_transport_model(
      self,
  ) -> FakeQualikizBasedTransportModel:
    return FakeQualikizBasedTransportModel()

  def build_dynamic_params(self, t: chex.Numeric):
    base_kwargs = dataclasses.asdict(super().build_dynamic_params(t))
    return qualikiz_based_transport_model.DynamicRuntimeParams(
        coll_mult=self.coll_mult,
        avoid_big_negative_s=self.avoid_big_negative_s,
        smag_alpha_correction=self.smag_alpha_correction,
        q_sawtooth_proxy=self.q_sawtooth_proxy,
        DVeff=self.DVeff,
        An_min=self.An_min,
        **base_kwargs,
    )
# pylint: enable=undefined-variable


if __name__ == '__main__':
  absltest.main()
