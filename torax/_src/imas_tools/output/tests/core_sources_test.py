# Copyright 2026 DeepMind Technologies Limited
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

"""Tests for IMAS core_sources output."""

import types
from unittest import mock

from absl.testing import absltest
import imas
import numpy as np
from torax._src.imas_tools.output import core_sources
from torax._src.sources import electron_cyclotron_source


class CoreSourcesTest(absltest.TestCase):

  @mock.patch.object(core_sources, "_fill_grid_coordinates")
  @mock.patch.object(core_sources, "_fill_profiles_1d")
  @mock.patch.object(core_sources, "_fill_global_quantities")
  def test_output_uses_complete_imas_source_identifiers(
      self, mock_global_quantities, mock_profiles_1d, mock_grid_coordinates
  ):
    del mock_global_quantities, mock_profiles_1d, mock_grid_coordinates
    ec_source_name = (
        electron_cyclotron_source.ElectronCyclotronSource.SOURCE_NAME
    )
    source_profiles = types.SimpleNamespace(
        T_e={ec_source_name: np.ones(1)},
        T_i={},
        n_e={},
        psi={},
        fast_ions={},
    )

    ids = core_sources.core_sources_to_IMAS(
        core_sources=[source_profiles],
        geometry=[mock.sentinel.geometry],
        times=np.array([0.0]),
    )

    identifiers = {
        str(source.identifier.name): source.identifier for source in ids.source
    }
    for imas_name in ("ec", "collisional_equipartition"):
      expected = imas.identifiers.core_source_identifier[imas_name]
      self.assertEqual(identifiers[imas_name].index, expected.index)
      self.assertEqual(identifiers[imas_name].description, expected.description)


if __name__ == "__main__":
  absltest.main()
