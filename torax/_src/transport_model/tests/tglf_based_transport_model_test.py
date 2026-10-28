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
import jax.numpy as jnp
import numpy as np
import pydantic
from torax._src import array_typing
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.fvm import cell_variable
from torax._src.geometry import geometry
from torax._src.pedestal_model import (
    pedestal_model_output as pedestal_model_output_lib,
)
from torax._src.test_utils import core_profile_helpers
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import register_model
from torax._src.transport_model import runtime_params as transport_params
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_coeffs
from torax._src.transport_model import transport_state as transport_state_lib
from torax._src.transport_model.tglf import tglf2py


def _get_config_and_model_inputs(
    transport: Mapping[str, Any],
):
  """Returns the model inputs for testing."""
  config = default_configs.get_default_config_dict()
  config["transport"] = transport
  torax_config = model_config.ToraxConfig.from_dict(config)
  core_profiles, runtime_params, geo = core_profile_helpers.make_initial_inputs(
      torax_config
  )
  two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
  transport_model = torax_config.transport.build_transport_model()
  return torax_config, (
      runtime_params,
      geo,
      core_profiles,
      pedestal_model_output_lib.IBCPedestalModelOutput.empty(),
      two_point_mask,
      transport_model.initial_state(runtime_params),
  )


def setUpModule():
  # This model isn't registered by default as it's actually a base class.
  register_model.register_transport_model(TGLFBasedTransportModelConfig)


class TGLFTransportModelTest(parameterized.TestCase):

  @absltest.skipIf(
      tglf2py.tglf2py_lib is None,
      "TGLF extension module 'tglf2py_lib' is not compiled/installed.",
  )
  def test_tglf_based_transport_model_output_shapes(self):
    """Tests that the core transport output has the right shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
            },
        },
    })
    transport_model = torax_config.transport.build_transport_model()

    core_transport, _ = transport_model(*model_inputs)
    expected_shape = model_inputs[1].rho_face_norm.shape
    self.assertEqual(core_transport.total.chi_face_ion.shape, expected_shape)
    self.assertEqual(core_transport.total.chi_face_el.shape, expected_shape)
    self.assertEqual(core_transport.total.d_face_el.shape, expected_shape)
    self.assertEqual(core_transport.total.v_face_el.shape, expected_shape)

  def test_tglf_based_transport_model_prepare_tglf_inputs_shapes(self):
    """Tests that the tglf inputs have the expected shapes."""
    torax_config, model_inputs = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
            },
        },
    })
    transport_model = (
        torax_config.transport.build_transport_model().core_transport_models[
            "tglf_based"
        ]
    )
    assert isinstance(
        transport_model, tglf_based_transport_model.TGLFBasedTransportModel
    )
    runtime_params, geo, core_profiles, _, two_point_mask, _ = model_inputs
    tglf_params = (
        runtime_params.transport.core_transport_model_params["tglf_based"]
    )
    assert isinstance(
        tglf_params, tglf_based_transport_model.RuntimeParams
    )
    tglf_inputs = transport_model._prepare_tglf_inputs(
        transport=tglf_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    expected_length = geo.rho_face_norm.shape[0]
    for key in tglf_inputs.__dict__.keys():
      self.assertEqual(getattr(tglf_inputs, key).shape, (expected_length,))

  def test_max_normalized_collisionality_caps_xnue(self):
    """Tests that max_normalized_collisionality clamps XNUE."""
    max_normalized_collisionality = 0.5
    # Get uncapped inputs (max_normalized_collisionality=inf).
    torax_config, uncapped_inputs = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
            },
        },
    })
    # Get capped inputs.
    _, capped_inputs = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
                "max_normalized_collisionality": max_normalized_collisionality,
            },
        },
    })
    transport_model = (
        torax_config.transport.build_transport_model().core_transport_models[
            "tglf_based"
        ]
    )
    assert isinstance(
        transport_model, tglf_based_transport_model.TGLFBasedTransportModel
    )
    runtime_uncapped, geo, core_profiles, _, two_point_mask, _ = uncapped_inputs
    runtime_capped = capped_inputs[0]

    tglf_params_uncapped = (
        runtime_uncapped.transport.core_transport_model_params["tglf_based"]
    )
    tglf_params_capped = (
        runtime_capped.transport.core_transport_model_params["tglf_based"]
    )
    assert isinstance(
        tglf_params_uncapped, tglf_based_transport_model.RuntimeParams
    )
    assert isinstance(
        tglf_params_capped, tglf_based_transport_model.RuntimeParams
    )

    uncapped = transport_model._prepare_tglf_inputs(
        transport=tglf_params_uncapped,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    capped = transport_model._prepare_tglf_inputs(
        transport=tglf_params_capped,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    # Precondition: some uncapped values must exceed the cap.
    above_cap = uncapped.XNUE > max_normalized_collisionality
    self.assertTrue(jnp.any(above_cap))
    # Where uncapped exceeds cap, capped should equal the cap.
    np.testing.assert_allclose(
        capped.XNUE[above_cap],
        jnp.full_like(capped.XNUE[above_cap], max_normalized_collisionality),
    )
    # Where uncapped is at or below cap, capped should be unchanged.
    np.testing.assert_allclose(
        capped.XNUE[~above_cap],
        uncapped.XNUE[~above_cap],
    )

  def test_vexb_shear_is_invariant_to_rigid_toroidal_rotation(self):
    # GACODE normalizes rotation to B_pol, so rigid toroidal rotation maps onto
    # a constant omega_0 = E_r / (R B_pol) and adds no ExB shear.
    _, (_, geo, core_profiles, _, two_point_mask, _) = (
        _get_config_and_model_inputs({
            "core_transport_models": {
                "tglf_based": {
                    "model_name": "tglf_based",
                },
            },
        })
    )
    tglf_config = TGLFBasedTransportModelConfig()
    transport_model = tglf_config.build_transport_model()
    tglf_params = tglf_config.build_runtime_params(t=0.0)
    omega = 1e5  # [rad/s]
    rigidly_rotating_core_profiles = dataclasses.replace(
        core_profiles,
        toroidal_angular_velocity=cell_variable.CellVariable(
            value=jnp.full_like(geo.rho_norm, omega),
            face_centers=geo.rho_face_norm,
            right_face_constraint=jnp.array(omega),
            right_face_grad_constraint=None,
        ),
    )

    reference = transport_model._prepare_tglf_inputs(
        transport=tglf_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    rotating = transport_model._prepare_tglf_inputs(
        transport=tglf_params,
        geo=geo,
        core_profiles=rigidly_rotating_core_profiles,
        two_point_mask=two_point_mask,
    )

    np.testing.assert_allclose(
        rotating.VEXB_SHEAR, reference.VEXB_SHEAR, rtol=1e-6, atol=1e-10
    )

  def test_tglf_based_transport_model_dv_effective(self):
    """Tests that DV_effective switches between effective D/V and scaled D."""
    torax_config_false, model_inputs_false = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
                "DV_effective": False,
            },
        },
    })
    transport_model_false = torax_config_false.transport.build_transport_model()
    core_transport_false, _ = transport_model_false(*model_inputs_false)

    torax_config_true, model_inputs_true = _get_config_and_model_inputs({
        "core_transport_models": {
            "tglf_based": {
                "model_name": "tglf_based",
                "DV_effective": True,
            },
        },
    })
    transport_model_true = torax_config_true.transport.build_transport_model()
    core_transport_true, _ = transport_model_true(*model_inputs_true)

    # With DV_effective=False (scaled D), d_face_el is set equal to chi_face_el.
    np.testing.assert_allclose(
        core_transport_false.total.d_face_el,
        core_transport_false.total.chi_face_el,
    )
    # With DV_effective=True, effective D/V differs from scaled D.
    self.assertFalse(
        np.allclose(
            core_transport_true.total.d_face_el,
            core_transport_false.total.d_face_el,
        )
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FakeTGLFBasedTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """Fake TGLFBasedTransportModel for testing purposes."""

  def call_implementation(
      self,
      transport_runtime_params: transport_params.ComponentRuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
      component_state: transport_state_lib.ComponentTransportState,
  ) -> tuple[
      transport_coeffs.TransportCoeffs,
      transport_state_lib.ComponentTransportState,
  ]:
    # Assert required for pytype.
    assert isinstance(
        transport_runtime_params,
        tglf_based_transport_model.RuntimeParams,
    )

    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport_runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )
    return (
        self._make_transport_coeffs(
            ion_heat_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 0.4,
            electron_heat_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 0.5,
            electron_particle_flux_GB=jnp.ones(geo.rho_face_norm.shape) * 1.6,
            tglf_inputs=tglf_inputs,
            transport=transport_runtime_params,
            geo=geo,
            core_profiles=core_profiles,
            two_point_mask=two_point_mask,
        ),
        component_state,
    )


# pylint: disable=invalid-name
class TGLFBasedTransportModelConfig(
    tglf_based_transport_model.TGLFBasedTransportModelConfig
):
  """Model for testing the TGLF-based transport model."""

  model_name: Annotated[Literal["tglf_based"], torax_pydantic.JAX_STATIC] = (
      "tglf_based"
  )
  DV_effective_smooth_width: pydantic.NonNegativeFloat = 0.01
  use_rotation: Annotated[bool, torax_pydantic.JAX_STATIC] = True

  def build_transport_model(
      self,
  ) -> FakeTGLFBasedTransportModel:
    return FakeTGLFBasedTransportModel()


if __name__ == "__main__":
  absltest.main()
