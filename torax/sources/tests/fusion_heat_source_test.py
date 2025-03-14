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
from typing import Callable, Mapping

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import constants
from torax import interpolated_param
from torax.config import build_runtime_params
from torax.config import plasma_composition
from torax.core_profiles import initialization
from torax.sources import fusion_heat_source
from torax.sources import pydantic_model as sources_pydantic_model
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.tests.test_lib import torax_refs


class FusionHeatSourceTest(test_lib.IonElSourceTestCase):
  """Tests for FusionHeatSource."""

  def setUp(self):
    super().setUp(
        source_name=fusion_heat_source.FusionHeatSource.SOURCE_NAME,
        source_config_class=fusion_heat_source.FusionHeatSourceConfig,
        needs_source_models=True,
    )

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(
          references_getter=torax_refs.chease_references_Ip_from_runtime_params
      ),
  ])
  def test_calc_fusion(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_fusion` function to a reference implementation."""
    references = references_getter()

    runtime_params = references.runtime_params

    sources = sources_pydantic_model.Sources.from_dict({
        fusion_heat_source.FusionHeatSource.SOURCE_NAME: {},
    })
    dynamic_runtime_params_slice, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
            sources=sources,
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    torax_fusion_power, _, _ = fusion_heat_source.calc_fusion(
        geo,
        core_profiles,
        static_slice,
        dynamic_runtime_params_slice,
    )

    reference_fusion_power = reference_calc_fusion(
        runtime_params, geo, core_profiles
    )

    np.testing.assert_allclose(torax_fusion_power, reference_fusion_power)

  @parameterized.named_parameters(
      ('no_fusion_D', 'D', 0.0),
      ('no_fusion_T', 'T', 0.0),
      ('no_fusion_HT', {'H': 0.5, 'T': 0.5}, 0.0),
      ('50-50-DT', {'D': 0.5, 'T': 0.5}, 1.0),
      ('25-75-DT', {'D': 0.25, 'T': 0.75}, 0.75),
  )
  def test_calc_fusion_different_ion_mixtures(
      self,
      main_ion_input: (
          str | Mapping[str, interpolated_param.TimeInterpolatedInput]
      ),
      expected_fusion_factor: float,
  ):
    """Compare `calc_fusion` function to a reference implementation for various ion mixtures."""
    references = torax_refs.chease_references_Ip_from_chease()

    runtime_params = references.runtime_params

    runtime_params.plasma_composition.main_ion_mixture = (
        plasma_composition.IonMixture.from_config(main_ion_input)
    )

    sources = sources_pydantic_model.Sources.from_dict({
        fusion_heat_source.FusionHeatSource.SOURCE_NAME: {},
    })
    dynamic_runtime_params_slice_t, geo = (
        torax_refs.build_consistent_dynamic_runtime_params_slice_and_geometry(
            runtime_params,
            references.geometry_provider,
            sources=sources,
            t=0.0,  # arbitrary since we don't use time-dependent params
        )
    )
    static_slice = build_runtime_params.build_static_runtime_params_slice(
        runtime_params=runtime_params,
        sources=sources,
        torax_mesh=geo.torax_mesh,
    )
    source_models = source_models_lib.SourceModels(
        sources=sources.source_model_config
    )
    core_profiles = initialization.initial_core_profiles(
        dynamic_runtime_params_slice=dynamic_runtime_params_slice_t,
        static_runtime_params_slice=static_slice,
        geo=geo,
        source_models=source_models,
    )

    torax_fusion_power, _, _ = fusion_heat_source.calc_fusion(
        geo,
        core_profiles,
        static_slice,
        dynamic_runtime_params_slice_t,
    )

    reference_fusion_power = reference_calc_fusion(
        runtime_params, geo, core_profiles
    )

    np.testing.assert_allclose(
        torax_fusion_power, expected_fusion_factor * reference_fusion_power
    )


def reference_calc_fusion(runtime_params, geo, core_profiles):
  """Reference implementation from PINT. We still use TORAX state here."""
  # PINT doesn't follow Google style
  # pylint:disable=invalid-name
  T = core_profiles.temp_ion.face_value()
  consts = constants.CONSTANTS

  # P [W/m^3] = Efus *1/4 * n^2 * <sigma*v>.
  # <sigma*v> for DT calculated with the Bosch-Hale parameterization
  # NF 1992.
  # T is in keV for the formula

  Efus = 17.6 * 1e3 * consts.keV2J
  mrc2 = 1124656
  BG = 34.3827
  C1 = 1.17302e-9
  C2 = 1.51361e-2
  C3 = 7.51886e-2
  C4 = 4.60643e-3
  C5 = 1.35e-2
  C6 = -1.0675e-4
  C7 = 1.366e-5

  theta = T / (
      1 - (T * (C2 + T * (C4 + T * C6))) / (1 + T * (C3 + T * (C5 + T * C7)))
  )
  xi = (BG**2 / (4 * theta)) ** (1 / 3)
  sigmav = (
      C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi) / 1e6
  )  # units of m^3/s

  Pfus = (
      Efus
      * 0.25
      * (core_profiles.ni.face_value() * runtime_params.numerics.nref) ** 2
      * sigmav
  )  # [W/m^3]
  Ptot = np.trapz(Pfus * geo.vpr_face, geo.rho_face_norm) / 1e6  # [MW]

  return Ptot


if __name__ == '__main__':
  absltest.main()
