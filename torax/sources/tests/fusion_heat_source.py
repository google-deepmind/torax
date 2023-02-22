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
from torax import initial_states
from torax.sources import fusion_heat_source
from torax.sources import source
from torax.sources import source_config
from torax.sources import source_profiles
from torax.sources.tests import test_lib
from torax.tests.test_lib import pint_ref


class FusionHeatSourceTest(test_lib.IonElSourceTestCase):
  """Tests for FusionHeatSource."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass(
        source_class=fusion_heat_source.FusionHeatSource,
        unsupported_types=[
            source_config.SourceType.FORMULA_BASED,
        ],
        expected_affected_mesh_states=(
            source.AffectedMeshStateAttribute.TEMP_ION,
            source.AffectedMeshStateAttribute.TEMP_EL,
        ),
    )

  @parameterized.parameters([
      dict(references_getter=pint_ref.circular_references),
      dict(references_getter=pint_ref.chease_references_Ip_from_chease),
      dict(references_getter=pint_ref.chease_references_Ip_from_config),
  ])
  def test_calc_fusion(
      self, references_getter: Callable[[], pint_ref.References]
  ):
    """Compare `calc_fusion` function to a reference implementation."""
    references = references_getter()

    config = references.config
    geo = references.geo
    nref = config.nref

    state = initial_states.initial_state(
        config,
        geo,
        sources=source_profiles.Sources(),
    )

    fusion_jax, _, _ = fusion_heat_source.calc_fusion(
        geo,
        state,
        nref,
    )

    def calculate_fusion(config, geo, profiles):
      """Reference implementation from pyntegrated_model."""
      # pyntegrated_model doesn't follow Google style
      # pylint:disable=invalid-name
      T = profiles.Ti.faceValue()
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
          Efus * 0.25 * (profiles.ni.faceValue() * config.nref) ** 2 * sigmav
      )  # [W/m^3]
      # Modification from raw pyntegrated_model: we use geo.r_face here,
      # rather than a call to geo.rface(), which in pyntegrated_model is FiPy
      # FaceVariable.
      Ptot = np.trapz(Pfus * geo.vpr_face, geo.r_face) / 1e6  # [MW]

      return Ptot

    profiles = pint_ref.state_to_profiles(state)

    fusion_pyntegrated = calculate_fusion(config, geo, profiles)

    np.testing.assert_allclose(fusion_jax, fusion_pyntegrated)


if __name__ == '__main__':
  absltest.main()
