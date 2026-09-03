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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src import array_typing
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.pedestal_model import pedestal_transition_state as pedestal_transition_state_lib
from torax._src.pedestal_model import runtime_params as pedestal_runtime_params_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.torax_pydantic import model_config
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import component
from torax._src.transport_model import enums
from torax._src.transport_model import pereverzev
from torax._src.transport_model import pydantic_model_base as transport_pydantic_model_base
from torax._src.transport_model import register_model
from torax._src.transport_model import runtime_params as transport_runtime_params_lib
from torax._src.transport_model import transport_coefficients_builder
from torax._src.transport_model import transport_model


@dataclasses.dataclass(frozen=True, eq=False)
class FixedTransportModel(component.ComponentTransportModel):
  """Fixed TransportModel for testing purposes."""

  def call_implementation(
      self,
      transport_runtime_params: (
          transport_runtime_params_lib.ComponentRuntimeParams
      ),
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      two_point_mask: array_typing.BoolVectorFace,
  ) -> component.TurbulentTransport:
    chi_face_ion = np.linspace(0.5, 2, geo.rho_face_norm.shape[0])
    chi_face_el = np.linspace(0.25, 1, geo.rho_face_norm.shape[0])
    d_face_el = np.linspace(2, 3, geo.rho_face_norm.shape[0])
    v_face_el = np.linspace(-0.2, -2, geo.rho_face_norm.shape[0])
    # Add sub-components
    chi_face_ion_bohm = chi_face_ion * 0.3
    chi_face_ion_gyrobohm = chi_face_ion * 0.7

    return component.TurbulentTransport(
        chi_face_ion=chi_face_ion,  # pyrefly: ignore[bad-argument-type]
        chi_face_el=chi_face_el,  # pyrefly: ignore[bad-argument-type]
        d_face_el=d_face_el,  # pyrefly: ignore[bad-argument-type]
        v_face_el=v_face_el,  # pyrefly: ignore[bad-argument-type]
        chi_face_ion_bohm=chi_face_ion_bohm,  # pyrefly: ignore[bad-argument-type]
        chi_face_ion_gyrobohm=chi_face_ion_gyrobohm,  # pyrefly: ignore[bad-argument-type]
    )


class FixedTransportConfig(
    transport_pydantic_model_base.ComponentTransportBase
):
  """Fixed transport config for a model that always returns fixed values."""

  model_name: Annotated[Literal['fixed'], torax_pydantic.JAX_STATIC] = 'fixed'

  def build_transport_model(self) -> FixedTransportModel:
    return FixedTransportModel()


def setUpModule():
  # Register the fixed transport config.
  register_model.register_transport_model(FixedTransportConfig)


class TransportMaskingTest(parameterized.TestCase):
  """Tests for output masking in transport models."""

  def test_single_model_masking(self):
    """Tests that disabling a channel zeroes its output in a single model."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        # Set the min values to 0.0 to avoid clipping overriding the masking.
        'chi_min': 0.0,
        'D_e_min': 0.0,
        'core_transport_models': {
            'fixed': {
                'model_name': 'fixed',
                'disable_chi_i': True,  # Should be zeroed
                'disable_D_e': False,  # Should be present
            },
        },
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

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    model = torax_config.transport.build_transport_model()
    coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_outputs,
        two_point_mask,
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
        'core_transport_models': {
            'base': {
                'model_name': 'fixed',  # Base model
                'disable_chi_i': False,
                'disable_D_e': False,
            },
            'additive': {
                'model_name': 'fixed',  # Additive model with selective enable
                'disable_chi_i': True,  # Should NOT add to chi_i
                'disable_D_e': False,  # Should add to D_e
            },
        },
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
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    model = torax_config.transport.build_transport_model()
    coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        pedestal_model_outputs,
        two_point_mask,
    )

    single_fixed_config = model_config.ToraxConfig.from_dict({
        **config,
        'transport': {
            'core_transport_models': {'fixed': {'model_name': 'fixed'}},
        },
    })
    single_model = single_fixed_config.transport.build_transport_model()
    single_runtime = build_runtime_params.RuntimeParamsProvider.from_config(
        single_fixed_config
    )(t=0.0)
    ref_coeffs = single_model(
        single_runtime,
        geo,
        core_profiles,
        pedestal_model_outputs,
        two_point_mask,
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

  def test_preserves_none_channel_enabled(self):
    model = FixedTransportModel()
    runtime_params = mock.create_autospec(
        transport_runtime_params_lib.ComponentRuntimeParams,
        disable_chi_i=False,
        disable_chi_e=False,
        disable_D_e=False,
        disable_V_e=False,
    )

    coeffs = component.TurbulentTransport(
        chi_face_ion=jnp.array([1.0]),
        chi_face_el=jnp.array([1.0]),
        d_face_el=jnp.array([1.0]),
        v_face_el=jnp.array([1.0]),
        chi_face_ion_bohm=None,
    )
    new_coeffs = model.zero_out_disabled_channels(runtime_params, coeffs)
    self.assertIsNone(new_coeffs.chi_face_ion_bohm)

  def test_preserves_none_channel_disabled(self):
    model = FixedTransportModel()
    runtime_params = mock.create_autospec(
        transport_runtime_params_lib.ComponentRuntimeParams,
        disable_chi_i=True,
        disable_chi_e=False,
        disable_D_e=False,
        disable_V_e=False,
    )

    coeffs = component.TurbulentTransport(
        chi_face_ion=jnp.array([1.0]),
        chi_face_el=jnp.array([1.0]),
        d_face_el=jnp.array([1.0]),
        v_face_el=jnp.array([1.0]),
        chi_face_ion_bohm=None,
    )
    new_coeffs_disabled = model.zero_out_disabled_channels(
        runtime_params, coeffs
    )
    self.assertIsNone(new_coeffs_disabled.chi_face_ion_bohm)

  def test_sub_channel_domain_restriction(self):
    """Tests that sub-channels are masked by domain restriction."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'chi_min': 0.0,
        'D_e_min': 0.0,
        'V_e_min': 0.0,
        'smoothing_width': 0.0,
        'core_transport_models': {
            'fixed': {
                'model_name': 'fixed',
                'rho_max': 0.8,
            },
        },
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

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    transport_model_instance = torax_config.transport.build_transport_model()
    coeffs = transport_model_instance(
        runtime_params,
        geo,
        core_profiles,
        pedestal_outputs,
        two_point_mask,
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


class TransportModelTest(absltest.TestCase):

  def test_adaptive_transport_preserves_pereverzev_transport_in_h_mode(self):
    """H-mode adaptive suppression leaves numerical PC transport unchanged."""
    config = default_configs.get_default_config_dict()
    config['solver'] = {
        'chi_pereverzev': 30.0,
        'D_pereverzev': 15.0,
    }
    config['pedestal'] = {
        'model_name': 'set_T_ped_n_ped',
        'set_pedestal': True,
        'mode': 'ADAPTIVE_TRANSPORT',
    }
    config['sources'] = {
        'generic_heat': {
            'P_total': 1e9,
            'is_explicit': True,
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    models = torax_config.build_models()
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=0.0)
    geo = torax_config.geometry.build_provider(t=0.0)
    core_profiles = initialization.initial_core_profiles(
        runtime_params,
        geo,
        models.source_models,
        models.neoclassical_models,
    )
    source_profiles = source_profile_builders.build_source_profiles(
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        source_models=models.source_models,
        neoclassical_models=models.neoclassical_models,
        explicit=True,
    )
    transition_state = dataclasses.replace(
        pedestal_transition_state_lib.PedestalTransitionState.empty_L_mode(),
        confinement_mode=jnp.asarray(
            pedestal_transition_state_lib.ConfinementMode.H_MODE
        ),
    )
    pedestal_output = models.pedestal_model(
        runtime_params,
        geo,
        core_profiles,
        source_profiles,
        transition_state,
    )
    transition_state = dataclasses.replace(
        transition_state,
        pedestal_model_output=pedestal_output,
    )
    for multiplier in dataclasses.astuple(
        pedestal_output.transport_multipliers
    ):
      # check pedestal transport multiplier is on
      self.assertLess(multiplier, 1.0)

    coeffs = transport_coefficients_builder.calculate_all_transport_coeffs(
        models.transport_model,
        models.neoclassical_models,
        runtime_params,
        geo,
        core_profiles,
        transition_state,
        use_pereverzev=True,
    )

    two_point_mask = pedestal_output.get_two_point_face_mask(
        geo, set_pedestal=runtime_params.pedestal.set_pedestal
    )
    raw_pereverzev = pereverzev.calculate_pereverzev_transport(
        runtime_params,
        geo,
        core_profiles,
        two_point_mask,
    )
    for field in dataclasses.fields(pereverzev.PereverzevTransport):
      with self.subTest(field=field.name):
        np.testing.assert_allclose(
            getattr(coeffs, field.name),
            getattr(raw_pereverzev, field.name),
            rtol=1e-12,
        )

  def test_combining(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'core_transport_models': {
            'inner': {
                'model_name': 'prescribed',
                'rho_max': 0.2,
                'chi_i': 1.0,
            },
            'mid': {
                'model_name': 'prescribed',
                'rho_min': 0.2,
                'rho_max': 0.8,
                'chi_i': 2.0,
            },
            'outer': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 3.0,
            },
        },
        'pedestal_transport_models': {
            'pedestal_prescribed': {'model_name': 'prescribed', 'chi_i': 0.1}
        },
    }
    config['pedestal'] = {'set_pedestal': True}
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
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
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    transport_coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
        two_point_mask,
    )
    # Target:
    # - 0.1 for rho = [rho_ped_top, rho_max]
    # - 3 for rho = (0.8, rho_ped_top), to check pedestal overrides it
    # - 5 for rho = (0.5, 0.8], to check case where models overlap
    # - 2 for rho = (0.2, 0.5], to check case rho_min_1 == rho_max_2
    # - 1 for rho = [0, 0.2], to check case where rho_min = 0
    target = jnp.where(geo.rho_face_norm <= 0.91, 3.0, 0.1)
    target = jnp.where(geo.rho_face_norm <= 0.8, 5.0, target)
    target = jnp.where(geo.rho_face_norm <= 0.5, 2.0, target)
    target = jnp.where(geo.rho_face_norm <= 0.2, 1.0, target)
    np.testing.assert_allclose(transport_coeffs.chi_face_ion, target)

  def test_chi_min(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'core_transport_models': {
            'prescribed': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 2.0,
            }
        },
        'chi_min': 1.0,
    }
    config['pedestal'] = {'set_pedestal': True}
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
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
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    transport_coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
        two_point_mask,
    )
    # Target:
    # - 1.0 for rho = [rho_ped_top, rho_max], set by chi_min
    # - 2.0 for rho = (0.5, rho_ped_top), set by the model
    # - 1.0 for rho = [0.0, 0.5], set by chi_min
    target = jnp.where(geo.rho_face_norm <= 0.91, 2.0, 1.0)
    target = jnp.where(geo.rho_face_norm <= 0.5, 1.0, target)
    np.testing.assert_allclose(transport_coeffs.chi_face_ion, target)

  def test_build_smoothing_matrix_zero_width_is_identity(self):
    """Tests that a zero smoothing width produces an identity matrix."""
    config = {
        'smoothing_width': 0.0,
        'core_transport_models': {
            'prescribed': {'model_name': 'prescribed', 'chi_i': 1.0}
        },
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = transport_model._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )
    np.testing.assert_allclose(
        matrix, np.eye(len(geo.rho_face_norm)), atol=1e-7
    )

  def test_build_smoothing_matrix_row_sums_and_constant_invariance(self):
    """Tests that matrix rows sum to 1 and preserve constant profiles."""
    config = {
        'smoothing_width': 0.08,
        'core_transport_models': {
            'prescribed': {'model_name': 'prescribed', 'chi_i': 1.0}
        },
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = transport_model._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )
    row_sums = np.sum(matrix, axis=1)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-6)

    constant_profile = jnp.full_like(geo.rho_face_norm, 3.5)
    smoothed = jnp.dot(matrix, constant_profile)
    np.testing.assert_allclose(smoothed, constant_profile, atol=1e-6)

  def test_build_smoothing_matrix_zone_isolation(self):
    """Tests that matrix is identity for points outside defined smoothing zones."""
    config = {
        'smoothing_zones': [
            {'rho_min': 0.3, 'rho_max': 0.7, 'smoothing_width': 0.05},
        ],
        'core_transport_models': {
            'prescribed': {'model_name': 'prescribed', 'chi_i': 1.0}
        },
    }
    _, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.91,
    )
    matrix = transport_model._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )

    outside_mask = (geo.rho_face_norm < 0.3) | (geo.rho_face_norm > 0.7)
    outside_indices = np.where(outside_mask)[0]
    for idx in outside_indices:
      expected_row = np.zeros(len(geo.rho_face_norm))
      expected_row[idx] = 1.0
      np.testing.assert_allclose(matrix[idx], expected_row, atol=1e-7)

  def test_build_smoothing_matrix_pedestal_boundary_isolation(self):
    """Tests that smoothing matrix is identity at/above pedestal top in IBC mode."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'smoothing_width': 0.08,
        'core_transport_models': {
            'prescribed': {'model_name': 'prescribed', 'chi_i': 1.0}
        },
    }
    config['pedestal'] = {
        'set_pedestal': True,
        'mode': 'INTERNAL_BOUNDARY_CONDITION',
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.8,
    )
    matrix = transport_model._build_smoothing_matrix(
        runtime_params.transport,
        runtime_params,
        geo,
        mock_pedestal_outputs,
    )

    pedestal_mask = geo.rho_face_norm >= 0.8
    pedestal_indices = np.where(pedestal_mask)[0]
    for idx in pedestal_indices:
      expected_row = np.zeros(len(geo.rho_face_norm))
      expected_row[idx] = 1.0
      np.testing.assert_allclose(matrix[idx], expected_row, atol=1e-7)

  def test_smoothing_zones(self):
    """Tests that smoothing_zones smoothes transport coefficients in the specified region."""
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'smoothing_zones': [
            {'rho_min': 0.3, 'rho_max': 0.7, 'smoothing_width': 0.08},
        ],
        'core_transport_models': {
            'inner': {
                'model_name': 'prescribed',
                'rho_max': 0.5,
                'chi_i': 1.0,
            },
            'outer': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 5.0,
            },
        },
        'chi_min': 0.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(t=torax_config.numerics.t_initial)
    source_models = torax_config.sources.build_models()
    neoclassical_models = torax_config.neoclassical.build_models()
    core_profiles = initialization.initial_core_profiles(
        runtime_params, geo, source_models, neoclassical_models
    )
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=0.95,
    )
    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs = model(
        runtime_params,
        geo,
        core_profiles,
        mock_pedestal_outputs,
        two_point_mask,
    )
    self.assertEqual(coeffs.chi_face_ion.shape, geo.rho_face_norm.shape)

    # 1. Unsmoothed regions: rho < 0.25 should equal 1.0,
    # rho > 0.75 should equal 5.0
    unsmoothed_left = geo.rho_face_norm < 0.25
    unsmoothed_right = geo.rho_face_norm > 0.75
    np.testing.assert_allclose(
        coeffs.chi_face_ion[unsmoothed_left], 1.0, atol=1e-5
    )
    np.testing.assert_allclose(
        coeffs.chi_face_ion[unsmoothed_right], 5.0, atol=1e-5
    )

    # 2. Inside smoothing zone: left of step (rho in (0.4, 0.5))
    # smoothed upwards (> 1.0), right of step (rho in (0.5, 0.6))
    # smoothed downwards (< 5.0).
    near_step_left = np.where(
        (geo.rho_face_norm > 0.4) & (geo.rho_face_norm < 0.5)
    )[0]
    near_step_right = np.where(
        (geo.rho_face_norm > 0.5) & (geo.rho_face_norm < 0.6)
    )[0]
    self.assertTrue(np.all(coeffs.chi_face_ion[near_step_left] > 1.0))
    self.assertTrue(np.all(coeffs.chi_face_ion[near_step_right] < 5.0))

  def test_smoothing_width_shortcut(self):
    """Tests that setting smoothing_width applies full-domain smoothing."""
    config_small = {
        'smoothing_width': 0.02,
        'core_transport_models': {
            'inner': {
                'model_name': 'prescribed',
                'rho_max': 0.5,
                'chi_i': 1.0,
            },
            'outer': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 5.0,
            },
        },
        'chi_min': 0.0,
    }
    config_large = {
        'smoothing_width': 0.12,
        'core_transport_models': {
            'inner': {
                'model_name': 'prescribed',
                'rho_max': 0.5,
                'chi_i': 1.0,
            },
            'outer': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 5.0,
            },
        },
        'chi_min': 0.0,
    }

    model_small, params_small, geo = self._build_model_and_params(config_small)
    model_large, params_large, _ = self._build_model_and_params(config_large)

    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs_small = model_small(
        params_small,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
        two_point_mask,
    )
    coeffs_large = model_large(
        params_large,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
        two_point_mask,
    )

    idx_left = np.where(
        (geo.rho_face_norm > 0.35) & (geo.rho_face_norm < 0.45)
    )[0]
    self.assertTrue(
        np.all(
            coeffs_large.chi_face_ion[idx_left]
            > coeffs_small.chi_face_ion[idx_left]
        )
    )

  def test_error_if_pedestal_model_defines_rho_min(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'core_transport_models': {'prescribed': {'model_name': 'prescribed'}},
        'pedestal_transport_models': {
            'pedestal_prescribed': {
                'model_name': 'prescribed',
                'rho_min': 0.1,
            }
        },
    }
    with self.assertRaisesRegex(
        ValueError, 'rho_min and rho_max not supported'
    ):
      model_config.ToraxConfig.from_dict(config)

  def test_error_if_pedestal_model_defines_rho_max(self):
    config = default_configs.get_default_config_dict()
    config['transport'] = {
        'core_transport_models': {'prescribed': {'model_name': 'prescribed'}},
        'pedestal_transport_models': {
            'pedestal_prescribed': {
                'model_name': 'prescribed',
                'rho_max': 0.9,
            }
        },
    }
    with self.assertRaisesRegex(
        ValueError, 'rho_min and rho_max not supported'
    ):
      model_config.ToraxConfig.from_dict(config)

  def _build_model_and_params(self, transport_config):
    config = default_configs.get_default_config_dict()
    config['transport'] = transport_config
    torax_config = model_config.ToraxConfig.from_dict(config)
    model = torax_config.transport.build_transport_model()
    geo = torax_config.geometry.build_provider(
        t=torax_config.numerics.t_initial
    )
    runtime_params = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )(
        t=torax_config.numerics.t_initial,
    )
    return model, runtime_params, geo

  def test_merge_mode_overwrite(self):
    """Tests that OVERWRITE mode wipes previous values in active region."""
    # Model 1: Value 1.0 everywhere.
    # Model 2: Value 2.0 in rho > 0.5. MergeMode = OVERWRITE.
    config = {
        'core_transport_models': {
            'model_1': {'model_name': 'prescribed', 'chi_i': 1.0},
            'model_2': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 2.0,
                'merge_mode': 'overwrite',
            },
        },
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,  # No pedestal restriction for this test
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs = model(
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
        two_point_mask,
    )

    # Expected: 1.0 for rho <= 0.5, 2.0 for rho > 0.5
    target = jnp.where(geo.rho_face_norm <= 0.5, 1.0, 2.0)
    np.testing.assert_allclose(coeffs.chi_face_ion, target)

  def test_overwrite_locks_subsequent(self):
    """Tests that OVERWRITE mode prevents subsequent ADD models from modifying the region."""
    # Model 1: Overwrite, Value 1.0 in rho > 0.5.
    # Model 2: Add, Value 2.0 everywhere.
    config = {
        'core_transport_models': {
            'model_1': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 1.0,
                'merge_mode': 'overwrite',
            },
            'model_2': {
                'model_name': 'prescribed',
                'chi_i': 2.0,
                'merge_mode': 'add',
            },
        },
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs = model(
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
        two_point_mask,
    )

    expected = jnp.where(geo.rho_face_norm <= 0.5, 2.0, 1.0)
    # Model 2 is locked out of rho > 0.5 by Model 1's OVERWRITE.
    np.testing.assert_allclose(coeffs.chi_face_ion, expected)

  def test_disable_channel_transparency(self):
    """Tests that disabling a channel makes the overwrite transparent for that channel."""
    # Model 1: Value 1.0 for chi_i and chi_e.
    # Model 2: Overwrite, Value 2.0 in rho > 0.5. BUT disable_chi_e = True.
    # Result: chi_i should be 1.0 then 2.0. chi_e should be 1.0 everywhere
    # (transparent overwrite).
    config = {
        'core_transport_models': {
            'model_1': {
                'model_name': 'prescribed',
                'chi_i': 1.0,
                'chi_e': 1.0,
            },
            'model_2': {
                'model_name': 'prescribed',
                'rho_min': 0.5,
                'chi_i': 2.0,
                'chi_e': 2.0,
                'merge_mode': 'overwrite',
                'disable_chi_e': True,
            },
        },
        'chi_min': 0.0,
    }
    model, runtime_params, geo = self._build_model_and_params(config)
    mock_pedestal_outputs = mock.create_autospec(
        pedestal_model_output_lib.PedestalModelOutput,
        instance=True,
        rho_norm_ped_top=1.0,
    )

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs = model(
        runtime_params,
        geo,
        mock.ANY,
        mock_pedestal_outputs,
        two_point_mask,
    )

    # chi_i: Overwritten in outer half.
    target_i = jnp.where(geo.rho_face_norm <= 0.5, 1.0, 2.0)
    np.testing.assert_allclose(coeffs.chi_face_ion, target_i)

    # chi_e: Not overwritten (transparent), so Model 1 value remains.
    # Model 2 contributes 0 because disable_chi_e=True, and it doesn't wipe
    # because it's disabled.
    target_e = jnp.ones_like(geo.rho_face_norm) * 1.0
    np.testing.assert_allclose(coeffs.chi_face_el, target_e)

  def test_none_handling_in_combine(self):
    """Tests that None values are preserved as None if no model writes to them."""
    # We use a mock model to return None for clear isolation.
    mock_model = mock.create_autospec(
        component.ComponentTransportModel, instance=True
    )
    geo = mock.Mock(spec=geometry.Geometry)
    geo.rho_face_norm = jnp.linspace(0, 1, 10)

    # Return a structure with some None fields
    mock_coeffs = component.TurbulentTransport(
        chi_face_ion=jnp.ones_like(geo.rho_face_norm),
        chi_face_el=jnp.ones_like(geo.rho_face_norm),
        d_face_el=jnp.ones_like(geo.rho_face_norm),
        v_face_el=jnp.ones_like(geo.rho_face_norm),
        # Optional fields as None
        chi_face_el_bohm=None,
        chi_face_el_gyrobohm=None,
        chi_face_ion_bohm=None,
        chi_face_ion_gyrobohm=None,
    )
    mock_model.return_value = mock_coeffs

    # Manually instantiate TransportModel with our mock
    combined_model = transport_model.TransportModel(
        core_transport_models={'mock': mock_model},
        pedestal_transport_models={},
    )

    # We need dummy params for the mock model
    mock_params = mock.Mock()
    mock_params.disable_chi_i = False
    mock_params.disable_chi_e = False
    mock_params.disable_D_e = False
    mock_params.disable_V_e = False
    mock_params.merge_mode = enums.MergeMode.ADD
    mock_params.rho_min = 0.0
    mock_params.rho_max = 1.0

    # We need a RuntimeParams for combined model
    combined_params = mock.create_autospec(
        transport_runtime_params_lib.RuntimeParams, instance=True
    )
    combined_params.core_transport_model_params = {'mock': mock_params}
    combined_params.pedestal_transport_model_params = {}
    # Set clipping and smoothing params so __call__ doesn't crash on mocks.
    combined_params.chi_min = 0.0
    combined_params.chi_max = 100.0
    combined_params.D_e_min = 0.0
    combined_params.D_e_max = 100.0
    combined_params.V_e_min = -100.0
    combined_params.V_e_max = 100.0
    combined_params.smoothing_width = 0.0
    combined_params.smoothing_zones = ()

    pedestal_output = mock.Mock(
        spec=pedestal_model_output_lib.PedestalModelOutput
    )
    pedestal_output.rho_norm_ped_top = 1.0
    pedestal_output.get_two_point_face_mask.return_value = jnp.zeros_like(
        geo.rho_face_norm, dtype=jnp.bool_
    )

    runtime_params = mock.Mock()
    runtime_params.transport = combined_params
    runtime_params.pedestal.set_pedestal = False
    runtime_params.pedestal.mode = (
        pedestal_runtime_params_lib.Mode.INTERNAL_BOUNDARY_CONDITION
    )
    runtime_params.profile_conditions.internal_boundary_conditions = None
    core_profiles = mock.Mock()

    two_point_mask = np.zeros_like(geo.rho_face_norm, dtype=bool)
    coeffs = combined_model(
        runtime_params, geo, core_profiles, pedestal_output, two_point_mask
    )

    # Check that output has None for optional fields
    self.assertIsNone(coeffs.chi_face_ion_bohm)
    self.assertIsNone(coeffs.chi_face_el_bohm)
    # Check that main fields are arrays
    self.assertIsNotNone(coeffs.chi_face_ion)


if __name__ == '__main__':
  absltest.main()
