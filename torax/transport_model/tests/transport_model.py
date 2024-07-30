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

"""Unit tests for torax.transport_model."""

import dataclasses
from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import geometry
from torax import sim as sim_lib
from torax import state
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.sources import source_models as source_models_lib
from torax.time_step_calculator import fixed_time_step_calculator
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model as transport_model_lib


class TransportSmoothingTest(parameterized.TestCase):
  """Tests Gaussian smoothing in the `torax.transport_model` package."""

  def test_smoothing(self):
    """Tests that smoothing works as expected."""
    # Set up default config and geo
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=general_runtime_params.ProfileConditions(
            set_pedestal=False
        ),
    )
    geo = geometry.build_circular_geometry()
    source_models_builder = source_models_lib.SourceModelsBuilder()
    source_models = source_models_builder()
    transport_model_builder = FakeTransportModelBuilder(
        runtime_params=runtime_params_lib.RuntimeParams(
            apply_inner_patch=True,
            apply_outer_patch=True,
            rho_inner=0.3,
            rho_outer=0.8,
            smoothing_sigma=0.05,
        )
    )
    transport_model = transport_model_builder()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=source_models_builder.runtime_params,
            geo=geo,
        )
    )
    time_calculator = fixed_time_step_calculator.FixedTimeStepCalculator()
    input_state = sim_lib.get_initial_state(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        time_step_calculator=time_calculator,
        source_models=source_models,
    )
    transport_coeffs = transport_model(
        dynamic_runtime_params_slice, geo, input_state.core_profiles
    )
    chi_face_ion_orig = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_face_el_orig = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_face_el_orig = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_face_el_orig = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])
    inner_patch_idx = np.searchsorted(
        geo.rho_face_norm, dynamic_runtime_params_slice.transport.rho_inner
    )
    outer_patch_idx = np.searchsorted(
        geo.rho_face_norm, dynamic_runtime_params_slice.transport.rho_outer
    )

    # assert that the smoothing did not impact the zones inside/outside the
    # inner/outer transport patch locations
    np.testing.assert_allclose(
        transport_coeffs['chi_face_ion'][:inner_patch_idx],
        chi_face_ion_orig[:inner_patch_idx],
    )
    np.testing.assert_allclose(
        transport_coeffs['chi_face_el'][:inner_patch_idx],
        chi_face_el_orig[:inner_patch_idx],
    )
    np.testing.assert_allclose(
        transport_coeffs['d_face_el'][:inner_patch_idx],
        d_face_el_orig[:inner_patch_idx],
    )
    np.testing.assert_allclose(
        transport_coeffs['v_face_el'][:inner_patch_idx],
        v_face_el_orig[:inner_patch_idx],
    )
    np.testing.assert_allclose(
        transport_coeffs['chi_face_ion'][outer_patch_idx:],
        chi_face_ion_orig[outer_patch_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs['chi_face_el'][outer_patch_idx:],
        chi_face_el_orig[outer_patch_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs['d_face_el'][outer_patch_idx:],
        d_face_el_orig[outer_patch_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs['v_face_el'][outer_patch_idx:],
        v_face_el_orig[outer_patch_idx:],
    )
    # carry out smoothing by hand for a representative middle location.
    # Check that behaviour is as expected
    test_idx = 5
    eps = 1e-7
    lower_cutoff = 0.01
    r_reduced = geo.rho_face_norm[inner_patch_idx:outer_patch_idx]
    test_r = r_reduced[test_idx]
    smoothing_array = np.exp(
        -np.log(2)
        * (r_reduced - test_r) ** 2
        / (dynamic_runtime_params_slice.transport.smoothing_sigma**2 + eps)
    )
    smoothing_array /= np.sum(smoothing_array)
    smoothing_array = np.where(
        smoothing_array < lower_cutoff, 0.0, smoothing_array
    )
    smoothing_array /= np.sum(smoothing_array)
    chi_face_ion_orig_smoothed_test_r = (
        chi_face_ion_orig[inner_patch_idx:outer_patch_idx] * smoothing_array
    )
    chi_face_el_orig_smoothed_test_r = (
        chi_face_el_orig[inner_patch_idx:outer_patch_idx] * smoothing_array
    )
    d_face_el_orig_smoothed_test_r = (
        d_face_el_orig[inner_patch_idx:outer_patch_idx] * smoothing_array
    )
    v_face_el_orig_smoothed_test_r = (
        v_face_el_orig[inner_patch_idx:outer_patch_idx] * smoothing_array
    )

    self.assertAlmostEqual(
        transport_coeffs['chi_face_ion'][inner_patch_idx + test_idx],
        chi_face_ion_orig_smoothed_test_r.sum(),
        places=6,
    )
    self.assertAlmostEqual(
        transport_coeffs['chi_face_el'][inner_patch_idx + test_idx],
        chi_face_el_orig_smoothed_test_r.sum(),
        places=6,
    )
    self.assertAlmostEqual(
        transport_coeffs['d_face_el'][inner_patch_idx + test_idx],
        d_face_el_orig_smoothed_test_r.sum(),
        places=6,
    )
    self.assertAlmostEqual(
        transport_coeffs['v_face_el'][inner_patch_idx + test_idx],
        v_face_el_orig_smoothed_test_r.sum(),
        places=6,
    )


class FakeTransportModel(transport_model_lib.TransportModel):
  """Fake TransportModel for testing purposes."""

  def __init__(self):
    super().__init__()
    self._frozen = True

  def _call_implementation(
      self,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
  ) -> state.CoreTransport:
    del dynamic_runtime_params_slice, core_profiles  # these are unused
    chi_face_ion = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_face_el = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_face_el = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_face_el = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])
    return state.CoreTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )


def _default_fake_builder() -> FakeTransportModel:
  return FakeTransportModel()


@dataclasses.dataclass(kw_only=True)
class FakeTransportModelBuilder(transport_model_lib.TransportModelBuilder):
  """Builds a class FakeTransportModel."""

  builder: Callable[
      [],
      FakeTransportModel,
  ] = _default_fake_builder

  def __call__(
      self,
  ) -> FakeTransportModel:
    return self.builder()


if __name__ == '__main__':
  absltest.main()
