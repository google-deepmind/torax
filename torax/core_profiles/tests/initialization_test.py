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
from torax import jax_utils
from torax import math_utils
from torax.config import build_runtime_params
from torax.core_profiles import initialization
from torax.geometry import geometry
from torax.sources import generic_current_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source_models as source_models_lib
from torax.sources import source_profile_builders
from torax.sources import source_profiles
from torax.tests.test_lib import default_configs
from torax.tests.test_lib import torax_refs
from torax.torax_pydantic import model_config

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
    bootstrap = source_profiles.BootstrapCurrentProfile.zero_profile(geo)
    external_current = generic_current_source.calculate_generic_current(
        mock.ANY,
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_name=generic_current_source.GenericCurrentSource.SOURCE_NAME,
        unused_state=mock.ANY,
        unused_calculated_source_profiles=mock.ANY,
        unused_conductivity=mock.ANY,
    )[0]
    _, j_total_hires = initialization._prescribe_currents(
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
        'j_bootstrap': {
            'bootstrap_multiplier': 0.0,
        },
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

    source_models = source_models_lib.SourceModels(
        sources=torax_config.sources, neoclassical=torax_config.neoclassical
    )

    profile_conditions1 = dict(
        initial_j_is_total_current=True,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )
    sources1 = {
        'j_bootstrap': {
            'bootstrap_multiplier': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
        'generic_current': {
            'fraction_of_total_current': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    torax_config.update_fields({
        'profile_conditions': profile_conditions1,
        'sources': sources1,
    })
    j_total1, _, _, j_ohmic1, _, _ = _calculate_currents(
        torax_config, source_models
    )

    profile_conditions2 = dict(
        initial_j_is_total_current=False,
        initial_psi_from_j=True,
        current_profile_nu=_CURRENT_PROFILE_NU,
    )
    sources2 = {
        'j_bootstrap': {
            'bootstrap_multiplier': 1.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
        'generic_current': {
            'fraction_of_total_current': 0.0,
            'mode': runtime_params_lib.Mode.MODEL_BASED,
        },
    }
    torax_config.update_fields({
        'profile_conditions': profile_conditions2,
        'sources': sources2,
    })

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
  j_total = core_profiles.currents.j_total
  j_total_face = core_profiles.currents.j_total_face
  j_external = sum(core_sources.psi.values())
  j_bootstrap = core_sources.j_bootstrap.j_bootstrap
  j_ohmic = j_total - j_external - j_bootstrap
  return j_total, j_total_face, j_external, j_ohmic, j_bootstrap, geo


if __name__ == '__main__':
  absltest.main()
