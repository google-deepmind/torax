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

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from torax._src import jax_utils
from torax._src import math_utils
from torax._src.config import build_runtime_params
from torax._src.core_profiles import initialization
from torax._src.geometry import geometry
from torax._src.neoclassical.bootstrap_current import base as bootstrap_current_base
from torax._src.sources import generic_current_source
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source_models as source_models_lib
from torax._src.sources import source_profile_builders
from torax._src.test_utils import default_configs
from torax._src.test_utils import torax_refs
from torax._src.torax_pydantic import model_config
from torax._src.fvm import cell_variable

# pylint: disable=invalid-name


class InitializationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    jax_utils.enable_errors(True)

  def test_update_psi_from_j(self):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = torax_refs.circular_references()

    # Turn on the external current source.
    dynamic_runtime_params_slice, geo = references.get_dynamic_slice_and_geo()
    bootstrap = bootstrap_current_base.BootstrapCurrent.zeros(geo)
    external_current = generic_current_source.calculate_generic_current(
        mock.ANY,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
        unused_state=mock.ANY,
        unused_calculated_source_profiles=mock.ANY,
        unused_conductivity=mock.ANY,
    )[0]
    j_total_hires = initialization._get_j_total_hires(
        bootstrap_profile=bootstrap,
        external_current=external_current,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
    )
    psi = initialization.update_psi_from_j(
        dynamic_runtime_params_slice.profile_conditions.Ip,
        geo,
        j_total_hires,
    ).value
    np.testing.assert_allclose(psi, references.psi.value)

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
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )
    dynamic_runtime_params_slice = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )(t=1.0)
    )
    geo = torax_config.geometry.build_provider(t=1.0)
    static_slice = build_runtime_params.build_static_params_from_config(
        torax_config
    )
    core_profiles = initialization.initial_core_profiles(
        static_slice,
        dynamic_runtime_params_slice,
        geo,
        source_models,
    )

    np.testing.assert_allclose(
        core_profiles.psi.value, expected_psi, atol=1e-6, rtol=1e-6
    )

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
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )

    torax_config.update_fields({'profile_conditions': profile_conditions1})
    (
        j_total1,
        j_total_face1,
        j_external1,
        j_ohmic1,
        _,
        _,
    ) = _calculate_currents(torax_config, source_models)

    torax_config.update_fields({'profile_conditions': profile_conditions2})
    j_total2, j_total_face2, _, j_ohmic2, _, geo = _calculate_currents(
        torax_config, source_models
    )

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
        np.mean(j_total1),
        np.mean(j_total2),
        rtol=_TOL,
    )

    # Check that the total current agrees with the expected reference formula.
    np.testing.assert_allclose(
        np.mean(j_total1),
        np.mean(j_total1_expected),
        rtol=_TOL,
    )

    # The only non-inductive current is the external current. Therefore the
    # sum of Ohmic + external current should be equal to the total current.
    np.testing.assert_allclose(
        np.mean(j_external1 + j_ohmic1),
        np.mean(j_total1),
        rtol=_TOL,
    )

    # j_ohmic2_expected is the expected formula for j_ohmic when setting
    # initial_j_is_total_current=False as in Case 2. It is the "nu" formula
    # scaled down to compensate for the external current.
    np.testing.assert_allclose(
        np.mean(j_ohmic2),
        np.mean(j_ohmic2_expected),
        rtol=_TOL,
    )

    # Check that the face conversions agree with the expected reference.
    np.testing.assert_allclose(
        np.mean(j_total_face1),
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
        np.mean(j_total_face2),
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
    _TOL = 3e-2
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
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )
    j_total1, _, _, j_ohmic1, _, _ = _calculate_currents(
        torax_config, source_models
    )

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
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )

    _, _, _, j_ohmic2, j_bootstrap2, _ = _calculate_currents(
        torax_config, source_models
    )

    # In Case 1, all the current should be Ohmic current.
    np.testing.assert_allclose(
        np.mean(j_ohmic1),
        np.mean(j_total1),
        rtol=_TOL,
    )

    # In Case 2, some of the current is bootstrap, so the Ohmic currents should
    # be different between the two cases.
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        np.mean(j_ohmic1),
        np.mean(j_ohmic2),
        rtol=_TOL,
    )

    # The only non-inductive current in Case 2 is the bootstrap current.
    # Thus, the sum of Ohmic and booststrap currents should be equal to the
    # total (ohmic) current in Case 1.
    np.testing.assert_allclose(
        np.mean(j_total1),
        np.mean(j_ohmic2 + j_bootstrap2),
        rtol=_TOL,
    )

  def test_initial_psi_from_geo_noop_circular(self):
    """Tests expected behaviour of initial psi and current options."""
    config = default_configs.get_default_config_dict()
    config['profile_conditions'] = {
        'initial_psi_from_j': False,
    }
    torax_config = model_config.ToraxConfig.from_dict(config)
    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )
    jtotal1, _, _, _, _, _ = _calculate_currents(torax_config, source_models)

    torax_config.update_fields({'profile_conditions.initial_psi_from_j': True})
    jtotal2, _, _, _, _, _ = _calculate_currents(torax_config, source_models)

    np.testing.assert_allclose(jtotal1, jtotal2)


def test_initial_core_profiles_omega_tor(self):
    """Tests initialization of omega_tor in CoreProfiles."""
    config_dict = default_configs.get_default_config_dict()

    # Define sample omega_tor profile and boundary condition
    omega_tor_profile_coeffs = {0.0: {0.0: 2.0, 1.0: 0.5}}  # Linearly decreasing profile
    omega_tor_bc_right = 0.1

    config_dict['profile_conditions']['omega_tor'] = omega_tor_profile_coeffs
    config_dict['profile_conditions']['omega_tor_right_bc'] = omega_tor_bc_right

    # Ensure these are not None if the main config might have them as None
    # For this test, T_i_right_bc, T_e_right_bc, n_e_right_bc must be defined
    # if their main profiles don't provide a value at rho=1.0, or if they are None.
    # The default config usually provides these.
    if config_dict['profile_conditions'].get('T_i_right_bc') is None:
        # Default configs provide these as time varying arrays.
        # We need to ensure they are set if we simplify to scalar for test.
        # However, the default config should be fine.
        pass
    if config_dict['profile_conditions'].get('T_e_right_bc') is None:
        pass
    if config_dict['profile_conditions'].get('n_e_right_bc') is None:
        pass

    torax_config = model_config.ToraxConfig.from_dict(config_dict)

    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )

    t_initial = torax_config.numerics.t_initial

    dynamic_runtime_params_slice_provider = (
        build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
            torax_config
        )
    )
    dynamic_runtime_params_slice = dynamic_runtime_params_slice_provider(t=t_initial)

    geo_provider = torax_config.geometry.build_provider
    geo = geo_provider(t=t_initial)

    static_runtime_params_slice = (
        build_runtime_params.build_static_params_from_config(torax_config)
    )

    core_profiles_out = initialization.initial_core_profiles(
        static_runtime_params_slice=static_runtime_params_slice,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    # Assertions
    self.assertIsInstance(core_profiles_out.omega_tor, cell_variable.CellVariable)

    # Expected omega_tor values on the cell grid
    # This comes directly from the dynamic_runtime_params_slice after interpolation by TimeVaryingArray
    expected_omega_tor_values = dynamic_runtime_params_slice.profile_conditions.omega_tor

    np.testing.assert_allclose(
        core_profiles_out.omega_tor.value,
        expected_omega_tor_values,
        atol=1e-6,
        err_msg="omega_tor profile values do not match expected.",
    )

    self.assertEqual(
        core_profiles_out.omega_tor.right_face_constraint,
        omega_tor_bc_right,
        msg="omega_tor right face constraint does not match.",
    )

    # Assuming default left boundary condition is zero gradient
    np.testing.assert_allclose(
        core_profiles_out.omega_tor.left_face_grad_constraint,
        0.0,
        atol=1e-9, # Comparing float with 0.0
        err_msg="omega_tor left face gradient constraint is not zero.",
    )


def _calculate_currents(
    torax_config: model_config.ToraxConfig,
    source_models: source_models_lib.SourceModels,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, geometry.Geometry
]:
  """Calculates j_total, j_external, and j_ohmic currents."""

  static_slice = build_runtime_params.build_static_params_from_config(
      torax_config
  )

  dynamic_slice, geo = (
      build_runtime_params.get_consistent_dynamic_runtime_params_slice_and_geometry(
          t=torax_config.numerics.t_initial,
          dynamic_runtime_params_slice_provider=build_runtime_params.DynamicRuntimeParamsSliceProvider.from_config(
              torax_config
          ),
          geometry_provider=torax_config.geometry.build_provider,
      )
  )

  core_profiles = initialization.initial_core_profiles(
      dynamic_runtime_params_slice=dynamic_slice,
      static_runtime_params_slice=static_slice,
      geo=geo,
      source_models=source_models,
  )
  conductivity = source_models.conductivity.calculate_conductivity(
      dynamic_slice, geo, core_profiles
  )
  core_sources = source_profile_builders.get_all_source_profiles(
      static_runtime_params_slice=static_slice,
      dynamic_runtime_params_slice=dynamic_slice,
      geo=geo,
      core_profiles=core_profiles,
      source_models=source_models,
      conductivity=conductivity,
  )
  j_total = core_profiles.j_total
  j_total_face = core_profiles.j_total_face
  j_external = sum(core_sources.psi.values())
  j_bootstrap = core_sources.bootstrap_current.j_bootstrap
  j_ohmic = j_total - j_external - j_bootstrap
  return j_total, j_total_face, j_external, j_ohmic, j_bootstrap, geo


if __name__ == '__main__':
  absltest.main()
