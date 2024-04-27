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

"""Tests for fusion_heat_source."""

from typing import Callable

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax import constants
from torax import core_profile_setters
from torax.config import runtime_params_slice
from torax.sources import fusion_heat_source
from torax.sources import runtime_params as runtime_params_lib
from torax.sources import source
from torax.sources import source_models as source_models_lib
from torax.sources.tests import test_lib
from torax.tests.test_lib import torax_refs


class FusionHeatSourceTest(test_lib.IonElSourceTestCase):
  """Tests for FusionHeatSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=fusion_heat_source.FusionHeatSource,
        unsupported_modes=[
            runtime_params_lib.Mode.FORMULA_BASED,
        ],
        expected_affected_core_profiles=(
            source.AffectedCoreProfile.TEMP_ION,
            source.AffectedCoreProfile.TEMP_EL,
        ),
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
    geo = references.geo
    nref = runtime_params.numerics.nref

    source_models = source_models_lib.SourceModels()
    dynamic_runtime_params_slice = (
        runtime_params_slice.build_dynamic_runtime_params_slice(
            runtime_params,
            sources=source_models.runtime_params,
        )
    )
    core_profiles = core_profile_setters.initial_core_profiles(
        static_runtime_params_slice=runtime_params_slice.build_static_runtime_params_slice(
            runtime_params
        ),
        dynamic_runtime_params_slice=dynamic_runtime_params_slice,
        geo=geo,
        source_models=source_models,
    )

    fusion_jax, _, _ = fusion_heat_source.calc_fusion(
        geo,
        core_profiles,
        nref,
    )

    def calculate_fusion(runtime_params, geo, core_profiles):
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
          1
          - (T * (C2 + T * (C4 + T * C6))) / (1 + T * (C3 + T * (C5 + T * C7)))
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
      Ptot = np.trapz(Pfus * geo.vpr_face, geo.r_face) / 1e6  # [MW]

      return Ptot

    fusion_pint = calculate_fusion(runtime_params, geo, core_profiles)

    np.testing.assert_allclose(fusion_jax, fusion_pint)


if __name__ == '__main__':
  absltest.main()
