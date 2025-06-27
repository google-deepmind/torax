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

from typing import Literal

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params_slice
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model as pedestal_model_lib
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax._src.transport_model import runtime_params as runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib


class TransportSmoothingTest(parameterized.TestCase):
  """Tests Gaussian smoothing in the `torax.transport_model` package."""

  def setUp(self):
    super().setUp()
    # Register the fake transport config.
    model_config.ToraxConfig.model_fields[
        'transport'
    ].annotation |= FakeTransportConfig
    model_config.ToraxConfig.model_rebuild(force=True)

  def test_smoothing(self):
    """Tests that smoothing works as expected."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'fake',
        'apply_inner_patch': True,
        'apply_outer_patch': True,
        'rho_inner': 0.3,
        'rho_outer': 0.8,
        'smoothing_width': 0.05,
    }
    config['profile_conditions'] = {
        'n_e_right_bc': 0.5e20,
    }
    config['geometry'] = {'geometry_type': 'circular'}
    torax_config = model_config.ToraxConfig.from_dict(config)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(
            t=torax_config.numerics.t_initial,
        )
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
        neoclassical_models,
    )
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    transport_model = torax_config.transport.build_transport_model()
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
        inner_patch_ones * dynamic_runtime_params_slice.transport.chi_i_inner,
        np.linspace(0.5, 2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chi_i_outer,
    ])
    chi_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chi_e_inner,
        np.linspace(0.25, 1, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chi_e_outer,
    ])
    d_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.D_e_inner,
        np.linspace(2, 3, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.D_e_outer,
    ])
    v_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.V_e_inner,
        np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.V_e_outer,
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
        / (dynamic_runtime_params_slice.transport.smoothing_width**2 + eps)
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
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'fake',
        'apply_inner_patch': True,
        'apply_outer_patch': True,
        'rho_inner': 0.3,
        'rho_outer': 0.8,
        'smoothing_width': 0.05,
        'smooth_everywhere': True,
    }
    config['profile_conditions'] = {
        'n_e_right_bc': 0.5e20,
    }
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
    }
    config['geometry'] = {'geometry_type': 'circular'}
    torax_config = model_config.ToraxConfig.from_dict(config)
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(t=torax_config.numerics.t_initial)
    )
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
        neoclassical_models,
    )
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        dynamic_runtime_params_slice, geo, core_profiles
    )
    transport_model = torax_config.transport.build_transport_model()
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
        inner_patch_ones * dynamic_runtime_params_slice.transport.chi_i_inner,
        np.linspace(0.5, 2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chi_min,
    ])
    chi_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.chi_e_inner,
        np.linspace(0.25, 1, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.chi_min,
    ])
    d_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.D_e_inner,
        np.linspace(2, 3, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.D_e_min,
    ])
    v_face_el_orig = np.concatenate([
        inner_patch_ones * dynamic_runtime_params_slice.transport.V_e_inner,
        np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])[
            inner_patch_idx:outer_patch_idx
        ],
        outer_patch_ones * dynamic_runtime_params_slice.transport.V_e_min,
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
        / (dynamic_runtime_params_slice.transport.smoothing_width**2 + eps)
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
      transport_dynamic_runtime_params: runtime_params_lib.DynamicRuntimeParams,
      dynamic_runtime_params_slice: runtime_params_slice.DynamicRuntimeParamsSlice,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    del dynamic_runtime_params_slice, core_profiles  # these are unused
    chi_face_ion = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_face_el = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_face_el = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_face_el = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])
    return transport_model_lib.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
    )

  def __hash__(self) -> int:
    return hash(self.__class__.__name__)

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self))


class FakeTransportConfig(transport_pydantic_model_base.TransportBase):
  """Fake transport config for a model that always returns zeros."""

  model_name: Literal['fake'] = 'fake'

  def build_transport_model(self) -> FakeTransportModel:
    return FakeTransportModel()


if __name__ == '__main__':
  absltest.main()
