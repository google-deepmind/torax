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
from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import state
from torax.config import build_runtime_params
from torax.config import profile_conditions as profile_conditions_lib
from torax.config import runtime_params as general_runtime_params
from torax.config import runtime_params_slice
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.geometry import pydantic_model as geometry_pydantic_model
from torax.pedestal_model import pedestal_model as pedestal_model_lib
from torax.pedestal_model import pydantic_model as pedestal_pydantic_model
from torax.pedestal_model import set_tped_nped
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.transport_model import runtime_params as runtime_params_lib
from torax.transport_model import transport_model as transport_model_lib


class TransportSmoothingTest(parameterized.TestCase):
  """Tests Gaussian smoothing in the `torax.transport_model` package."""

  def test_smoothing(self):
    """Tests that smoothing works as expected."""
    # Set up default config and geo
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            set_pedestal=False,
            ne_bound_right=0.5,
        ),
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    sources = sources_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
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
    pedestal = pedestal_pydantic_model.Pedestal()
    pedestal_model = pedestal.build_pedestal_model()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
            pedestal=pedestal,
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
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    transport_coeffs = transport_model(
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        pedestal_model_outputs,
    )
    inner_patch_idx = np.searchsorted(
        geo.rho_face_norm, dynamic_runtime_params_slice.transport.rho_inner
    )
    outer_patch_idx = np.searchsorted(
        geo.rho_face_norm, dynamic_runtime_params_slice.transport.rho_outer
    )
    inner_patch_ones = np.ones(inner_patch_idx)
    outer_patch_ones = np.ones(geo.rho_face_norm.shape[0] - outer_patch_idx)
    chi_face_ion_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chii_inner,
        np.linspace(0.5, 2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chii_outer,
    ])
    chi_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chie_inner,
        np.linspace(0.25, 1, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chie_outer,
    ])
    d_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.De_inner,
        np.linspace(2, 3, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.De_outer,
    ])
    v_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.Ve_inner,
        np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.Ve_outer,
    ])

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

    np.testing.assert_allclose(
        transport_coeffs['chi_face_ion'][inner_patch_idx + test_idx],
        chi_face_ion_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['chi_face_el'][inner_patch_idx + test_idx],
        chi_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['d_face_el'][inner_patch_idx + test_idx],
        d_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['v_face_el'][inner_patch_idx + test_idx],
        v_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )

  def test_smoothing_everywhere(self):
    """Tests that smoothing everywhere works as expected."""
    # Set up default config and geo
    runtime_params = general_runtime_params.GeneralRuntimeParams(
        profile_conditions=profile_conditions_lib.ProfileConditions(
            ne_bound_right=0.5,
        ),
    )
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    sources = sources_pydantic_model.Sources()
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    transport_model_builder = FakeTransportModelBuilder(
        runtime_params=runtime_params_lib.RuntimeParams(
            apply_inner_patch=True,
            apply_outer_patch=True,
            rho_inner=0.3,
            rho_outer=0.8,
            smoothing_sigma=0.05,
            smooth_everywhere=True,
        )
    )
    transport_model = transport_model_builder()
    pedestal = pedestal_pydantic_model.Pedestal()
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider(
            runtime_params,
            transport=transport_model_builder.runtime_params,
            sources=sources,
            torax_mesh=geo.torax_mesh,
            pedestal=pedestal,
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
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )
    pedestal_model = set_tped_nped.SetTemperatureDensityPedestalModel()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    transport_coeffs = transport_model(
        dynamic_runtime_params_slice,
        geo,
        core_profiles,
        pedestal_model_outputs,
    )
    inner_patch_idx = np.searchsorted(
        geo.rho_face_norm, dynamic_runtime_params_slice.transport.rho_inner
    )
    # set to mimic pedestal zone minimization
    outer_patch_idx = np.searchsorted(
        geo.rho_face_norm,
        pedestal_model_outputs.rho_norm_ped_top,
    )
    inner_patch_ones = np.ones(inner_patch_idx)
    outer_patch_ones = np.ones(geo.rho_face_norm.shape[0] - outer_patch_idx)
    chi_face_ion_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chii_inner,
        np.linspace(0.5, 2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chimin,
    ])
    chi_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chie_inner,
        np.linspace(0.25, 1, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chimin,
    ])
    d_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.De_inner,
        np.linspace(2, 3, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.Demin,
    ])
    v_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.Ve_inner,
        np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.Vemin,
    ])

    # assert that the smoothing did impact the zones inside/outside the
    # inner/outer transport patch locations
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['chi_face_ion'][:inner_patch_idx],
        chi_face_ion_orig[:inner_patch_idx],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['chi_face_el'][:inner_patch_idx],
        chi_face_el_orig[:inner_patch_idx],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['d_face_el'][:inner_patch_idx],
        d_face_el_orig[:inner_patch_idx],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['v_face_el'][:inner_patch_idx],
        v_face_el_orig[:inner_patch_idx],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['chi_face_ion'][outer_patch_idx:],
        chi_face_ion_orig[outer_patch_idx:],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['chi_face_el'][outer_patch_idx:],
        chi_face_el_orig[outer_patch_idx:],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['d_face_el'][outer_patch_idx:],
        d_face_el_orig[outer_patch_idx:],
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs['v_face_el'][outer_patch_idx:],
        v_face_el_orig[outer_patch_idx:],
    )

    # carry out smoothing by hand for a representative middle location.
    # Check that behaviour is as expected
    test_idx = 12
    eps = 1e-7
    lower_cutoff = 0.01
    r = geo.rho_face_norm
    test_r = r[test_idx]
    smoothing_array = np.exp(
        -np.log(2)
        * (r - test_r) ** 2
        / (dynamic_runtime_params_slice.transport.smoothing_sigma**2 + eps)
    )
    smoothing_array /= np.sum(smoothing_array)
    smoothing_array = np.where(
        smoothing_array < lower_cutoff, 0.0, smoothing_array
    )
    smoothing_array /= np.sum(smoothing_array)
    chi_face_ion_orig_smoothed_test_r = chi_face_ion_orig * smoothing_array
    chi_face_el_orig_smoothed_test_r = chi_face_el_orig * smoothing_array
    d_face_el_orig_smoothed_test_r = d_face_el_orig * smoothing_array
    v_face_el_orig_smoothed_test_r = v_face_el_orig * smoothing_array

    np.testing.assert_allclose(
        transport_coeffs['chi_face_ion'][test_idx],
        chi_face_ion_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['chi_face_el'][test_idx],
        chi_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['d_face_el'][test_idx],
        d_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs['v_face_el'][test_idx],
        v_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
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
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
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
