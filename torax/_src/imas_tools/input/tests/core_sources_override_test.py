# Copyright 2026 Google LLC
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

"""Regression tests for explicit IMAS core source identifier overrides."""

import pathlib

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.imas_tools.input import core_sources
from torax._src.imas_tools.input import loader


class IMASSourceOverridesTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    directory = pathlib.Path(__file__).parent
    self.ids = loader.load_imas_data(
        "core_sources_ddv4.nc", "core_sources", directory=directory
    )

  def test_default_skips_unmapped_source_types(self):
    result = core_sources.sources_from_IMAS(self.ids)
    self.assertNotIn("impurity_radiation", result)
    self.assertNotIn("cyclotron_radiation", result)
    self.assertIn("ecrh", result)
    self.assertIn("bremsstrahlung", result)

  def test_explicit_line_radiation_alias(self):
    result = core_sources.sources_from_IMAS(
        self.ids,
        source_name_overrides={"line_radiation": "impurity_radiation"},
    )
    self.assertIn("impurity_radiation", result)
    imas_source = next(
        src
        for src in self.ids.source
        if str(src.identifier.name) == "line_radiation"
    )
    expected = imas_source.profiles_1d[0].electrons.energy
    actual = result["impurity_radiation"]["prescribed_values"][0][2][0]
    np.testing.assert_allclose(actual, expected)
    self.assertEqual(result["impurity_radiation"]["mode"], "PRESCRIBED")

  def test_explicit_synchrotron_alias_only_if_requested(self):
    result = core_sources.sources_from_IMAS(
        self.ids,
        source_name_overrides={"synchrotron_radiation": "cyclotron_radiation"},
    )
    self.assertIn("cyclotron_radiation", result)
    imas_source = next(
        src
        for src in self.ids.source
        if str(src.identifier.name) == "synchrotron_radiation"
    )
    np.testing.assert_allclose(
        result["cyclotron_radiation"]["prescribed_values"][0][2][0],
        imas_source.profiles_1d[0].electrons.energy,
    )

  def test_explicit_alias_cannot_double_count_native_source(self):
    with self.assertRaisesRegex(ValueError, "refusing to double count"):
      core_sources.sources_from_IMAS(
          self.ids,
          source_name_overrides={"line_radiation": "bremsstrahlung"},
      )

  def test_existing_multi_launcher_ec_sources_are_still_combined(self):
    result = core_sources.sources_from_IMAS(self.ids)
    sources = [
        source
        for source in self.ids.source
        if str(source.identifier.name) == "ec"
    ]
    self.assertLen(sources, 2)
    expected = (
        sources[0].profiles_1d[0].electrons.energy
        + sources[1].profiles_1d[0].electrons.energy
    )
    np.testing.assert_allclose(
        result["ecrh"]["prescribed_values"][0][2][0], expected
    )

  def test_unknown_target_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "Unknown TORAX source"):
      core_sources.sources_from_IMAS(
          self.ids,
          source_name_overrides={"line_radiation": "not_a_torax_source"},
      )

  @parameterized.parameters(
      "total", "auxiliary", "radiation", "cyclotron_synchrotron_radiation"
  )
  def test_composite_source_cannot_be_mapped_to_one_model(self, name):
    with self.assertRaisesRegex(ValueError, "Aggregate IMAS source"):
      core_sources.sources_from_IMAS(
          self.ids, source_name_overrides={name: "impurity_radiation"}
      )

  def test_external_only_does_not_include_internal_overrides(self):
    result = core_sources.sources_from_IMAS(
        self.ids,
        load_only_external_sources=True,
        source_name_overrides={"line_radiation": "impurity_radiation"},
    )
    self.assertNotIn("impurity_radiation", result)
    self.assertNotIn("bremsstrahlung", result)
    self.assertIn("ecrh", result)
    self.assertIn("icrh", result)

  def test_empty_mapping_is_backwards_compatible(self):
    old = core_sources.sources_from_IMAS(self.ids)
    new = core_sources.sources_from_IMAS(self.ids, source_name_overrides={})
    self.assertEqual(old.keys(), new.keys())
    np.testing.assert_allclose(
        old["ecrh"]["prescribed_values"][0][2][0],
        new["ecrh"]["prescribed_values"][0][2][0],
    )


if __name__ == "__main__":
  absltest.main()
