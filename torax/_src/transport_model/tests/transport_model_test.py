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
from typing import Annotated, Literal

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import combined
from torax._src.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax._src.transport_model import register_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_model as transport_model_lib


@dataclasses.dataclass(frozen=True, eq=False)
class FixedTransportModel(transport_model_lib.TransportModel):
  """Fixed TransportModel for testing purposes."""

  def call_implementation(
      self,
      transport_runtime_params: transport_runtime_params_lib.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    chi_face_ion = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_face_el = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_face_el = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_face_el = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])
    # Add sub-components
    chi_face_ion_bohm = chi_face_ion * 0.3
    chi_face_ion_gyrobohm = chi_face_ion * 0.7

    return transport_model_lib.TurbulentTransport(
        chi_face_ion=chi_face_ion,
        chi_face_el=chi_face_el,
        d_face_el=d_face_el,
        v_face_el=v_face_el,
        chi_face_ion_bohm=chi_face_ion_bohm,
        chi_face_ion_gyrobohm=chi_face_ion_gyrobohm,
    )


class FixedTransportConfig(transport_pydantic_model_base.TransportBase):
  """Fixed transport config for a model that always returns fixed values."""

  model_name: Annotated[Literal['fixed'], torax_pydantic.JAX_STATIC] = 'fixed'

  def build_transport_model(self) -> FixedTransportModel:
    return FixedTransportModel()


def setUpModule():
  # Register the fixed transport config.
  register_model.register_transport_model(FixedTransportConfig)


class TransportSmoothingTest(parameterized.TestCase):
  """Tests Gaussian smoothing in the `torax.transport_model` package."""

  def test_smoothing(self):
    """Tests that smoothing works as expected."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'smoothing_width': 0.05,
        'transport_models': [{
            'model_name': 'fixed',
        }],
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
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=True,
    )
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )
    transport_model = torax_config.transport.build_transport_model()
    transport_coeffs = transport_model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_outputs,
    )

    pedestal_idx = np.searchsorted(
        geo.rho_face_norm, pedestal_model_outputs.rho_norm_ped_top
    )

    # Reconstruct original profiles before smoothing (but after masking and
    # clipping)
    assert isinstance(runtime_params.transport, combined.RuntimeParams)
    chi_min = runtime_params.transport.chi_min
    chi_max = runtime_params.transport.chi_max
    d_e_min = runtime_params.transport.D_e_min
    d_e_max = runtime_params.transport.D_e_max
    v_e_min = runtime_params.transport.V_e_min
    v_e_max = runtime_params.transport.V_e_max

    chi_i_raw = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_e_raw = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_e_raw = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_e_raw = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])

    core_mask = geo.rho_face_norm <= pedestal_model_outputs.rho_norm_ped_top

    chi_face_ion_orig = np.where(core_mask, chi_i_raw, 0.0)
    chi_face_el_orig = np.where(core_mask, chi_e_raw, 0.0)
    d_face_el_orig = np.where(core_mask, d_e_raw, 0.0)
    v_face_el_orig = np.where(core_mask, v_e_raw, 0.0)

    chi_face_ion_orig = np.clip(chi_face_ion_orig, chi_min, chi_max)
    chi_face_el_orig = np.clip(chi_face_el_orig, chi_min, chi_max)
    d_face_el_orig = np.clip(d_face_el_orig, d_e_min, d_e_max)
    v_face_el_orig = np.clip(v_face_el_orig, v_e_min, v_e_max)

    # assert that the smoothing did not impact the pedestal zone (masked out)
    np.testing.assert_allclose(
        transport_coeffs.chi_face_ion[pedestal_idx:],
        chi_face_ion_orig[pedestal_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs.chi_face_el[pedestal_idx:],
        chi_face_el_orig[pedestal_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs.d_face_el[pedestal_idx:],
        d_face_el_orig[pedestal_idx:],
    )
    np.testing.assert_allclose(
        transport_coeffs.v_face_el[pedestal_idx:],
        v_face_el_orig[pedestal_idx:],
    )

    # carry out smoothing by hand for a representative middle location in the
    # core.
    test_idx = 5
    eps = 1e-7
    lower_cutoff = 0.01
    r_reduced = geo.rho_face_norm[:pedestal_idx]
    test_r = r_reduced[test_idx]
    smoothing_array = np.exp(
        -np.log(2)
        * (r_reduced - test_r) ** 2
        / (runtime_params.transport.smoothing_width**2 + eps)
    )
    smoothing_array /= np.sum(smoothing_array)
    smoothing_array = np.where(
        smoothing_array < lower_cutoff, 0.0, smoothing_array
    )
    smoothing_array /= np.sum(smoothing_array)
    chi_face_ion_orig_smoothed_test_r = (
        chi_face_ion_orig[:pedestal_idx] * smoothing_array
    )
    chi_face_el_orig_smoothed_test_r = (
        chi_face_el_orig[:pedestal_idx] * smoothing_array
    )
    d_face_el_orig_smoothed_test_r = (
        d_face_el_orig[:pedestal_idx] * smoothing_array
    )
    v_face_el_orig_smoothed_test_r = (
        v_face_el_orig[:pedestal_idx] * smoothing_array
    )

    np.testing.assert_allclose(
        transport_coeffs.chi_face_ion[test_idx],
        chi_face_ion_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.chi_face_el[test_idx],
        chi_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.d_face_el[test_idx],
        d_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.v_face_el[test_idx],
        v_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )

  def test_smoothing_everywhere(self):
    """Tests that smoothing everywhere works as expected."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'smoothing_width': 0.05,
        'smooth_everywhere': True,
        'transport_models': [{
            'model_name': 'fixed',
        }],
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
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial,
    )
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=True,
    )
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )
    transport_model = torax_config.transport.build_transport_model()
    transport_coeffs = transport_model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_outputs,
    )

    # Apply the smoothing
    transport_coeffs = transport_model._smooth_coeffs(
        runtime_params, geo, transport_coeffs, pedestal_model_outputs
    )

    pedestal_idx = np.searchsorted(
        geo.rho_face_norm,
        pedestal_model_outputs.rho_norm_ped_top,
    )

    # Reconstruct original profiles before smoothing (but after masking and
    # clipping)
    assert isinstance(runtime_params.transport, combined.RuntimeParams)
    chi_min = runtime_params.transport.chi_min
    chi_max = runtime_params.transport.chi_max
    d_e_min = runtime_params.transport.D_e_min
    d_e_max = runtime_params.transport.D_e_max
    v_e_min = runtime_params.transport.V_e_min
    v_e_max = runtime_params.transport.V_e_max

    chi_i_raw = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_e_raw = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_e_raw = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_e_raw = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])

    core_mask = geo.rho_face_norm <= pedestal_model_outputs.rho_norm_ped_top

    chi_face_ion_orig = np.where(core_mask, chi_i_raw, 0.0)
    chi_face_el_orig = np.where(core_mask, chi_e_raw, 0.0)
    d_face_el_orig = np.where(core_mask, d_e_raw, 0.0)
    v_face_el_orig = np.where(core_mask, v_e_raw, 0.0)

    chi_face_ion_orig = np.clip(chi_face_ion_orig, chi_min, chi_max)
    chi_face_el_orig = np.clip(chi_face_el_orig, chi_min, chi_max)
    d_face_el_orig = np.clip(d_face_el_orig, d_e_min, d_e_max)
    v_face_el_orig = np.clip(v_face_el_orig, v_e_min, v_e_max)

    # assert that the smoothing did impact the pedestal zone
    # (smooth_everywhere=True)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs.chi_face_ion[pedestal_idx:],
        chi_face_ion_orig[pedestal_idx:],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs.chi_face_el[pedestal_idx:],
        chi_face_el_orig[pedestal_idx:],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs.d_face_el[pedestal_idx:],
        d_face_el_orig[pedestal_idx:],
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        transport_coeffs.v_face_el[pedestal_idx:],
        v_face_el_orig[pedestal_idx:],
    )

    # carry out smoothing by hand for a representative middle location.
    test_idx = 12
    eps = 1e-7
    lower_cutoff = 0.01
    r = geo.rho_face_norm
    test_r = r[test_idx]
    smoothing_array = np.exp(
        -np.log(2)
        * (r - test_r) ** 2
        / (runtime_params.transport.smoothing_width**2 + eps)
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
        transport_coeffs.chi_face_ion[test_idx],
        chi_face_ion_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.chi_face_el[test_idx],
        chi_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.d_face_el[test_idx],
        d_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        transport_coeffs.v_face_el[test_idx],
        v_face_el_orig_smoothed_test_r.sum(),
        rtol=1e-6,
    )


class TransportMaskingTest(parameterized.TestCase):
  """Tests for output masking in transport models."""

  def test_single_model_masking(self):
    """Tests that disabling a channel zeroes its output in a single model."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'chi_min': 0.0,
        'D_e_min': 0.0,
        'transport_models': [{
            'model_name': 'fixed',
            'disable_chi_i': True,
            'disable_D_e': False,
        }],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    # Build components
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        source_models,
        neoclassical_models,
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=source_models,
        neoclassical_models=neoclassical_models,
        explicit=True,
    )
    # We need a pedestal model even if unused by the fixed transport
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    pedestal_model_outputs = pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    transport_model = torax_config.transport.build_transport_model()
    coeffs = transport_model(
        runtime_params, geo, core_profiles, pedestal_model_outputs
    )

    # Verify chi_i is zeroed out
    np.testing.assert_allclose(coeffs.chi_face_ion, 0.0)
    if coeffs.chi_face_ion_bohm is not None:
      np.testing.assert_allclose(coeffs.chi_face_ion_bohm, 0.0)
    if coeffs.chi_face_ion_gyrobohm is not None:
      np.testing.assert_allclose(coeffs.chi_face_ion_gyrobohm, 0.0)

    # Verify D_e is non-zero (FixedTransportModel returns non-zero values)
    self.assertFalse(np.allclose(coeffs.d_face_el, 0.0))

  def test_combined_model_masking(self):
    """Tests that masking works correctly in a combined model."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [
            {
                'model_name': 'fixed',  # Base model
                'disable_chi_i': False,
                'disable_D_e': False,
            },
            {
                'model_name': 'fixed',  # Additive model with selective enable
                'disable_chi_i': True,  # Should NOT add to chi_i
                'disable_D_e': False,  # Should add to D_e
            },
        ],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        torax_config.sources.build_models(),
        torax_config.neoclassical.build_models(),
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=torax_config.sources.build_models(),
        neoclassical_models=torax_config.neoclassical.build_models(),
        explicit=True,
    )
    pedestal_model_outputs = pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )
    transport_model = torax_config.transport.build_transport_model()
    coeffs = transport_model(
        runtime_params, geo, core_profiles, pedestal_model_outputs
    )

    # Get reference values from a single fixed model
    single_fixed_config = model_config.ToraxConfig.from_dict({
        **config,
        'transport': {
            'model_name': 'combined',
            'transport_models': [{
                'model_name': 'fixed',
            }],
        },
    })
    single_model = single_fixed_config.transport.build_transport_model()
    single_runtime = build_runtime_params.RuntimeParamsProvider.from_config(
        single_fixed_config
    )(t=0.0)
    ref_coeffs = single_model(
        single_runtime, geo, core_profiles, pedestal_model_outputs
    )

    # chi_i should be approx equal to single model (1x contribution)
    # The first model adds it, the second model has it disabled (adds 0)
    np.testing.assert_allclose(
        coeffs.chi_face_ion, ref_coeffs.chi_face_ion, rtol=1e-5
    )

    # D_e should be approx double the single model (2x contribution)
    # Both models add to it.
    np.testing.assert_allclose(
        coeffs.d_face_el, 2 * ref_coeffs.d_face_el, rtol=1e-5
    )

  def test_preserves_none_enabled(self):
    """Tests that None values are preserved when channel is enabled."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [{
            'model_name': 'fixed',
        }],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)

    coeffs = transport_model_lib.TurbulentTransport(
        chi_face_ion=jnp.array([1.0]),
        chi_face_el=jnp.array([1.0]),
        d_face_el=jnp.array([1.0]),
        v_face_el=jnp.array([1.0]),
        chi_face_ion_bohm=None,
    )

    # Test preservation when enabled
    new_coeffs = model.zero_out_disabled_channels(
        runtime_params.transport, coeffs
    )
    self.assertIsNone(new_coeffs.chi_face_ion_bohm)

  def test_preserves_none_disabled(self):
    """Tests that None values are preserved when channel is disabled."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'transport_models': [{
            'model_name': 'fixed',
        }],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)

    coeffs = transport_model_lib.TurbulentTransport(
        chi_face_ion=jnp.array([1.0]),
        chi_face_el=jnp.array([1.0]),
        d_face_el=jnp.array([1.0]),
        v_face_el=jnp.array([1.0]),
        chi_face_ion_bohm=None,
    )

    # Test preservation when disabled
    disabled_params = dataclasses.replace(
        runtime_params.transport, disable_chi_i=True
    )
    new_coeffs_disabled = model.zero_out_disabled_channels(
        disabled_params, coeffs
    )
    self.assertIsNone(new_coeffs_disabled.chi_face_ion_bohm)

  def test_sub_channel_domain_restriction(self):
    """Tests that sub-channels are masked by domain restriction."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'model_name': 'combined',
        'chi_min': 0.0,
        'D_e_min': 0.0,
        'V_e_min': 0.0,
        'transport_models': [{
            'model_name': 'fixed',
            'rho_max': 0.8,
        }],
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    # We need a pedestal model even if unused by the fixed transport
    pedestal_model = torax_config.pedestal.build_pedestal_model()
    # Mock core profiles (not used by FixedTransportModel but needed for API)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        torax_config.sources.build_models(),
        torax_config.neoclassical.build_models(),
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=torax_config.sources.build_models(),
        neoclassical_models=torax_config.neoclassical.build_models(),
        explicit=True,
    )
    pedestal_outputs = pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        pedestal_transition_state=pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
    )

    transport_model = torax_config.transport.build_transport_model()
    coeffs = transport_model(
        runtime_params, geo, core_profiles, pedestal_outputs
    )

    # Find index where rho > 0.8
    cutoff_idx = np.searchsorted(geo.rho_face_norm, 0.8, side='right')

    # Verify main channel is zeroed
    np.testing.assert_allclose(coeffs.chi_face_ion[cutoff_idx:], 0.0)

    # Verify sub-channels are also zeroed
    # FixedTransportModel sets chi_face_ion_bohm = chi_face_ion * 0.3
    # If not masked, it would be non-zero because FixedTransportModel computes
    # it everywhere
    self.assertIsNotNone(coeffs.chi_face_ion_bohm)
    np.testing.assert_allclose(coeffs.chi_face_ion_bohm[cutoff_idx:], 0.0)
    self.assertIsNotNone(coeffs.chi_face_ion_gyrobohm)
    np.testing.assert_allclose(coeffs.chi_face_ion_gyrobohm[cutoff_idx:], 0.0)


if __name__ == '__main__':
  absltest.main()
