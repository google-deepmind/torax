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
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src import jax_utils
from torax._src import math_utils
from torax._src import state
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.core_profiles import profile_conditions as profile_conditions_lib
from torax._src.geometry import geometry
from torax._src.geometry import standard_geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.physics import psi_calculations
from torax._src.sources import generic_current_source
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.test_utils import torax_refs
from torax._src.torax_pydantic import model_config

# pylint: disable=invalid-name


@dataclasses.dataclass
class _Currents:
  """Container for the various currents used in tests."""

  j_toroidal_total: jax.Array
  j_toroidal_total_face: jax.Array
  j_toroidal_external: jax.Array
  j_toroidal_bootstrap: jax.Array
  j_toroidal_ohmic: jax.Array


class InitializationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)

  def test_update_psi_from_j(self):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = torax_refs.circular_references()

    # Turn on the external current source.
    runtime_params, geo = references.get_runtime_params_and_geo()
    bootstrap = bootstrap_current_base.BootstrapCurrent.zeros(geo)
    j_parallel_external = generic_current_source.calculate_generic_current(
        runtime_params=runtime_params,
        geo=geo,
        source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
        unused_state=mock.ANY,
        unused_calculated_source_profiles=mock.ANY,
        unused_conductivity=mock.ANY,
    )[0]
    j_toroidal_external = psi_calculations.j_parallel_to_j_toroidal(
        j_parallel_external, geo, runtime_params.numerics.min_rho_norm
    )
    j_total_hires = (
        initialization.get_j_toroidal_total_hires_with_external_sources(
            bootstrap_current=bootstrap,
            runtime_params=runtime_params,
            geo=geo,
            j_toroidal_external=j_toroidal_external,
        )
    )
    psi = initialization.update_psi_from_j(
        runtime_params.profile_conditions.Ip,
        geo,
        j_total_hires,
    ).value
    np.testing.assert_allclose(psi, references.psi.value)

  def test_initial_core_profiles_toroidal_velocity(self):
    config = default_configs.get_default_config_dict()
    # Test default initialization (zeros)
    torax_config = model_config.ToraxConfig.from_dict(config)
    core_profiles, geo, _ = _get_initial_state(torax_config)
    np.testing.assert_allclose(
        core_profiles.toroidal_velocity.value, np.zeros_like(geo.rho)
    )

  def test_initial_toroidal_velocity_from_profile_conditions(self):
    config = default_configs.get_default_config_dict()
    toroidal_velocity_test = np.array([10.0, 20.0, 30.0, 40.0])
    _, geo, _ = _get_initial_state(model_config.ToraxConfig.from_dict(config))
    config['profile_conditions']['toroidal_velocity'] = {
        0.0: {
            rho: value
            for rho, value in zip(geo.rho_norm, toroidal_velocity_test)
        }
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    core_profiles, _, _ = _get_initial_state(torax_config)
    np.testing.assert_allclose(
        core_profiles.toroidal_velocity.value, toroidal_velocity_test
    )

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
    config = default_configs.get_default_config_dict()
    config['profile_conditions']['psi'] = psi
    torax_config = model_config.ToraxConfig.from_dict(config)
    core_profiles, _, _ = _get_initial_state(torax_config)

    np.testing.assert_allclose(
        core_profiles.psi.value, expected_psi, atol=1e-6, rtol=1e-6
    )

  def test_initial_psi_from_geo(self):
    """Tests that psi is correctly loaded from a CHEASE file (Case 2)."""
    config = default_configs.get_default_config_dict()
    config['geometry']['geometry_type'] = 'chease'
    config['profile_conditions']['initial_psi_from_j'] = False
    torax_config = model_config.ToraxConfig.from_dict(config)

    core_profiles, geo, _ = _get_initial_state(torax_config)

    # The `psi_from_Ip` attribute of the geometry is the ground truth here.
    self.assertIsInstance(geo, standard_geometry.StandardGeometry)
    np.testing.assert_allclose(core_profiles.psi.value, geo.psi_from_Ip)

  @parameterized.parameters([
      dict(initial_psi_from_j=False),
      dict(initial_psi_from_j=True),
  ])
  def test_initial_psi_from_config_overrides_initial_psi_from_j(
      self, initial_psi_from_j: bool
  ):
    """Tests that providing psi directly in config takes precedence."""
    config = default_configs.get_default_config_dict()
    config['geometry']['geometry_type'] = 'chease'
    # Set both flags that would normally trigger Case 3
    psi_profile = {0: {0: 42.0, 1: 43.0}}
    config['profile_conditions']['initial_psi_from_j'] = initial_psi_from_j
    config['profile_conditions']['psi'] = psi_profile
    torax_config = model_config.ToraxConfig.from_dict(config)

    core_profiles, geo, _ = _get_initial_state(torax_config)

    # Check that the final psi matches the one provided in the config.
    expected_psi = np.interp(
        geo.rho_norm, np.array([0.0, 1.0]), np.array([42.0, 43.0])
    )
    np.testing.assert_allclose(core_profiles.psi.value, expected_psi)

  @parameterized.parameters([
      dict(geometry_name='circular'),
      dict(geometry_name='chease'),
  ])
  def test_compare_initial_currents_with_different_initial_j_ohmic(
      self,
      geometry_name: str,
  ):
    _CURRENT_PROFILE_NU = 2
    _FRACTION_OF_TOTAL_CURRENT = 0.25
    _NRHO = 100
    _TOL = 3e-2

    config = default_configs.get_default_config_dict()
    config['geometry']['geometry_type'] = geometry_name
    config['geometry']['n_rho'] = _NRHO
    config['sources'] = {
        'generic_current': {
            'fraction_of_total_current': _FRACTION_OF_TOTAL_CURRENT
        },
    }
    torax_config = model_config.ToraxConfig.from_dict(config)

    profile_conditions1 = dict(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )
    profile_conditions2 = dict(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )

    torax_config.update_fields({'profile_conditions': profile_conditions1})
    _, _, currents1 = _get_initial_state(torax_config)

    torax_config.update_fields({'profile_conditions': profile_conditions2})
    _, geo, currents2 = _get_initial_state(torax_config)

    # calculate total and Ohmic current profile references
    jformula = (1 - geo.rho_norm**2) ** _CURRENT_PROFILE_NU
    denom = jax.scipy.integrate.trapezoid(jformula * geo.spr, geo.rho_norm)
    ctot = torax_config.profile_conditions.Ip.value[0] / denom
    j_total1_expected = jformula * ctot
    j_ohmic2_expected = j_total1_expected * (1 - _FRACTION_OF_TOTAL_CURRENT)

    # Due to approximation errors in psi-->j_total conversions, as well as
    # modifications to j_total on axis, we only compare the current profile
    # mean values up to relatively loose tolerance.

    # Both total currents should be equal, even if the relative contribution
    # of Ohmic and external current is different.
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        np.mean(currents1.j_toroidal_total),
        np.mean(currents2.j_toroidal_total),
        rtol=_TOL,
    )

    # Check that the total current agrees with the expected reference formula.
    np.testing.assert_allclose(
        np.mean(currents1.j_toroidal_total),
        np.mean(j_total1_expected),
        rtol=_TOL,
    )

    # The only non-inductive current is the external current. Therefore the
    # sum of Ohmic + external current should be equal to the total current.
    np.testing.assert_allclose(
        np.mean(currents1.j_toroidal_external + currents1.j_toroidal_ohmic),
        np.mean(currents1.j_toroidal_total),
        rtol=_TOL,
    )

    # j_ohmic2_expected is the expected formula for j_ohmic when setting
    # initial_j_is_total_current=False as in Case 2. It is the "nu" formula
    # scaled down to compensate for the external current.
    np.testing.assert_allclose(
        np.mean(currents2.j_toroidal_ohmic),
        np.mean(j_ohmic2_expected),
        rtol=_TOL,
    )

    # Check that the face conversions agree with the expected reference.
    np.testing.assert_allclose(
        np.mean(currents1.j_toroidal_total_face),
        np.mean(
            math_utils.cell_to_face(
                j_total1_expected,
                geo,
                preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
            )
        ),
        rtol=_TOL,
    )

    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        np.mean(currents2.j_toroidal_total_face),
        np.mean(
            math_utils.cell_to_face(
                j_total1_expected,
                geo,
                preserved_quantity=math_utils.IntegralPreservationQuantity.SURFACE,
            )
        ),
        rtol=_TOL,
    )

  def test_initial_psi_from_j_with_bootstrap_is_consistent_with_case_without_bootstrap(
      self,
  ):
    _CURRENT_PROFILE_NU = 2
    _NRHO = 100
    # TODO(b/439047730): Investigate whether tolerance can be reduced by doing a
    # better comparison, e.g. by integrating the j profile.
    _TOL = 4e-2
    config = default_configs.get_default_config_dict()
    config['geometry']['geometry_type'] = 'chease'
    config['geometry']['n_rho'] = _NRHO
    torax_config = model_config.ToraxConfig.from_dict(config)

    profile_conditions1 = dict(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )
    sources1 = {
        'generic_current': {
            'fraction_of_total_current': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    neoclassical_1 = {
        'bootstrap_current': {
            'model_name': 'zeros',
        },
    }
    torax_config.update_fields({
        'profile_conditions': profile_conditions1,
        'sources': sources1,
        'neoclassical': neoclassical_1,
    })
    _, _, currents1 = _get_initial_state(torax_config)

    profile_conditions2 = dict(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )
    sources2 = {
        'generic_current': {
            'fraction_of_total_current': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    neoclassical_2 = {
        'bootstrap_current': {
            'model_name': 'sauter',
            'bootstrap_multiplier': 1.0,
        },
    }
    torax_config.update_fields({
        'profile_conditions': profile_conditions2,
        'sources': sources2,
        'neoclassical': neoclassical_2,
    })
    _, _, currents2 = _get_initial_state(torax_config)

    # In Case 1, all the current should be Ohmic current.
    np.testing.assert_allclose(
        np.mean(currents1.j_toroidal_ohmic),
        np.mean(currents1.j_toroidal_total),
        rtol=_TOL,
    )

    # In Case 2, some of the current is bootstrap, so the Ohmic currents should
    # be different between the two cases.
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        np.mean(currents1.j_toroidal_ohmic),
        np.mean(currents2.j_toroidal_ohmic),
        rtol=_TOL,
    )

    # The only non-inductive current in Case 2 is the bootstrap current.
    # Thus, the sum of Ohmic and booststrap currents should be equal to the
    # total (ohmic) current in Case 1.
    np.testing.assert_allclose(
        np.mean(currents1.j_toroidal_total),
        np.mean(currents2.j_toroidal_ohmic + currents2.j_toroidal_bootstrap),
        rtol=_TOL,
    )

  def test_initial_psi_from_geo_noop_circular(self):
    """Tests expected behaviour of initial psi and current options."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_from_j': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    _, _, currents1 = _get_initial_state(torax_config)
    jtotal1 = currents1.j_toroidal_total

    torax_config.update_fields({'profile_conditions.initial_psi_from_j': True})
    _, _, currents2 = _get_initial_state(torax_config)
    jtotal2 = currents2.j_toroidal_total

    np.testing.assert_allclose(jtotal1, jtotal2)

  def test_get_initial_psi_mode_geometry(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'geometry',
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        mock.ANY,
    )
    self.assertEqual(psi_source, profile_conditions_lib.InitialPsiMode.GEOMETRY)

  def test_get_initial_psi_mode_j(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'j',
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        mock.ANY,
    )
    self.assertEqual(psi_source, profile_conditions_lib.InitialPsiMode.J)

  def test_get_initial_psi_mode_profile_conditions(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'profile_conditions',
        'psi': 15.0,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        mock.ANY,
    )
    self.assertEqual(
        psi_source, profile_conditions_lib.InitialPsiMode.PROFILE_CONDITIONS
    )

  def test_get_initial_psi_mode_legacy_initial_psi_from_j(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'profile_conditions',
        'initial_psi_from_j': True,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        torax_config.geometry.build_provider(t=0.0),
    )
    self.assertEqual(psi_source, profile_conditions_lib.InitialPsiMode.J)

  def test_get_initial_psi_mode_legacy_initial_psi_from_j_circular_geo(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'profile_conditions',
        'initial_psi_from_j': False,
    }
    config['geometry']['geometry_type'] = 'circular'
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        torax_config.geometry.build_provider(t=0.0),
    )
    self.assertEqual(psi_source, profile_conditions_lib.InitialPsiMode.J)

  def test_get_initial_psi_mode_legacy_init_from_geo(self):
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_mode': 'profile_conditions',
        'initial_psi_from_j': False,
    }
    config['geometry']['geometry_type'] = 'chease'
    torax_config = model_config.ToraxConfig.from_dict(config)
    params_provider = build_runtime_params.RuntimeParamsProvider.from_config(
        torax_config
    )
    runtime_params = params_provider(t=0.0)
    psi_source = initialization._get_initial_psi_mode(
        runtime_params,
        torax_config.geometry.build_provider(t=0.0),
    )
    self.assertEqual(psi_source, profile_conditions_lib.InitialPsiMode.GEOMETRY)


def _get_initial_state(
    torax_config: model_config.ToraxConfig,
) -> tuple[
    state.CoreProfiles,
    geometry.Geometry,
    _Currents,
]:
  """Returns initial core profiles, sources, geometry and currents for a config."""
  source_models = torax_config.sources.build_models()
  neoclassical_models = torax_config.neoclassical.build_models()
  runtime_params, geo = (
      build_runtime_params.get_consistent_runtime_params_and_geometry(
          t=torax_config.numerics.t_initial,
          runtime_params_provider=build_runtime_params.RuntimeParamsProvider.from_config(
              torax_config
          ),
          geometry_provider=torax_config.geometry.build_provider,
      )
  )
  core_profiles = initialization.initial_core_profiles(
      runtime_params=runtime_params,
      geo=geo,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
  )
  conductivity = neoclassical_models.conductivity.calculate_conductivity(
      geo, core_profiles
  )
  core_sources = source_profile_builders.get_all_source_profiles(
      runtime_params=runtime_params,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      neoclassical_models=neoclassical_models,
      conductivity=conductivity,
  )
  j_toroidal_total = core_profiles.j_total
  j_toroidal_total_face = core_profiles.j_total_face
  j_toroidal_external = psi_calculations.j_parallel_to_j_toroidal(
      sum(core_sources.psi.values()), geo, runtime_params.numerics.min_rho_norm
  )
  j_toroidal_bootstrap = psi_calculations.j_parallel_to_j_toroidal(
      core_sources.bootstrap_current.j_parallel_bootstrap,
      geo,
      runtime_params.numerics.min_rho_norm,
  )
  j_toroidal_ohmic = (
      j_toroidal_total - j_toroidal_external - j_toroidal_bootstrap
  )
  currents = _Currents(
      j_toroidal_total=j_toroidal_total,
      j_toroidal_total_face=j_toroidal_total_face,
      j_toroidal_external=j_toroidal_external,
      j_toroidal_bootstrap=j_toroidal_bootstrap,
      j_toroidal_ohmic=j_toroidal_ohmic,
  )
  return core_profiles, geo, currents


if __name__ == '__main__':
  absltest.main()
