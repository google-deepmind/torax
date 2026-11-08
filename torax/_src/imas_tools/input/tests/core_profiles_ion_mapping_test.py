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

"""Tests for configurable IMAS-to-TORAX ion aliases."""

from absl.testing import absltest
import numpy as np
from torax._src.imas_tools.input import core_profiles
from torax._src.imas_tools.input import loader


class IMASIonAliasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ids = loader.load_imas_data(
        "core_profiles_ddv4_iterhybrid_rampup_conditions.nc", "core_profiles"
    )
    self._set_ions(("D", "T", "He5"))

  def _set_ions(self, names):
    for profile in self.ids.profiles_1d:
      profile.ion.resize(len(names))
      for idx, symbol in enumerate(names):
        profile.ion[idx].name = symbol
        if idx == 0:
          profile.ion[idx].density = [9e19, 3e19]
        elif idx == 1:
          profile.ion[idx].density = [9e19, 3e19]
        else:
          profile.ion[idx].density = profile.electrons.density / 100

  def test_nonstandard_impurity_without_mapping_remains_invalid(self):
    with self.assertRaises(KeyError):
      core_profiles.plasma_composition_from_IMAS(self.ids)

  def test_nonstandard_impurity_can_be_mapped_to_supported_species(self):
    result = core_profiles.plasma_composition_from_IMAS(
        self.ids, imas_to_torax_ions={"He5": "He"}
    )
    self.assertEqual(tuple(result["impurity"]["species"]), ("He",))
    ratios = result["impurity"]["species"]["He"][2]
    for ratio in ratios:
      np.testing.assert_allclose(ratio, 0.01)
    np.testing.assert_allclose(result["main_ion"]["D"][1], [0.5, 0.5])
    np.testing.assert_allclose(result["main_ion"]["T"][1], [0.5, 0.5])

  def test_main_ion_aliases_are_used_for_presence_checks(self):
    self._set_ions(("Deuterium", "Tritium", "He5"))
    result = core_profiles.plasma_composition_from_IMAS(
        self.ids,
        main_ions_symbols=("D", "T"),
        imas_to_torax_ions={
            "Deuterium": "D",
            "Tritium": "T",
            "He5": "He",
        },
    )
    self.assertEqual(set(result["main_ion"]), {"D", "T"})
    self.assertEqual(set(result["impurity"]["species"]), {"He"})

  def test_explicit_main_ion_missing_after_mapping_raises(self):
    with self.assertRaisesRegex(ValueError, "expected main ion"):
      core_profiles.plasma_composition_from_IMAS(
          self.ids,
          main_ions_symbols=("H",),
          imas_to_torax_ions={"He5": "He"},
      )

  def test_invalid_alias_target_fails_validation(self):
    with self.assertRaises(KeyError):
      core_profiles.plasma_composition_from_IMAS(
          self.ids, imas_to_torax_ions={"He5": "NotARealIon"}
      )

  def test_excluded_ion_uses_original_imas_name(self):
    result = core_profiles.plasma_composition_from_IMAS(
        self.ids,
        excluded_impurities=("He5",),
        imas_to_torax_ions={"He5": "He"},
    )
    self.assertEqual(result["impurity"]["species"], {})

  def test_two_source_species_cannot_silently_overwrite_one_species(self):
    self._set_ions(("D", "He", "He5"))
    with self.assertRaisesRegex(ValueError, "distinct populations"):
      core_profiles.plasma_composition_from_IMAS(
          self.ids, imas_to_torax_ions={"He5": "He"}
      )

  def test_mixed_dt_cannot_be_misrepresented_as_pure_ion(self):
    self._set_ions(("DT", "T", "He"))
    with self.assertRaisesRegex(ValueError, "mixed DT population"):
      core_profiles.plasma_composition_from_IMAS(
          self.ids, imas_to_torax_ions={"DT": "D"}
      )

  def test_identity_mappings_preserve_existing_output(self):
    self._set_ions(("D", "T", "He"))
    plain = core_profiles.plasma_composition_from_IMAS(self.ids)
    mapped = core_profiles.plasma_composition_from_IMAS(
        self.ids, imas_to_torax_ions={"D": "D", "He": "He"}
    )
    self.assertEqual(plain["main_ion"].keys(), mapped["main_ion"].keys())
    self.assertEqual(
        plain["impurity"]["species"].keys(),
        mapped["impurity"]["species"].keys(),
    )
    for ion in ("D", "T"):
      np.testing.assert_array_equal(
          plain["main_ion"][ion][1], mapped["main_ion"][ion][1]
      )


if __name__ == "__main__":
  absltest.main()
