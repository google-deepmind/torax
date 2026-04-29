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
"""Tests for global ion_cyclotron_source infrastructure (model-agnostic)."""

from absl.testing import absltest
import jax
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.physics import fast_ion as fast_ion_lib
from torax._src.sources import runtime_params as runtime_params_lib
from torax._src.sources import source as source_lib
from torax._src.sources.ion_cyclotron_source import base as icrh_base
from torax._src.sources.tests import test_lib

# Most of the checks and computations in TORAX require float64.
jax.config.update('jax_enable_x64', True)


class _DummyConfig(icrh_base.IonCyclotronSourceConfig):

  def build_runtime_params(self, t):
    return runtime_params_lib.RuntimeParams(
        prescribed_values=(),
        mode=self.mode,
        is_explicit=self.is_explicit,
    )

  @property
  def model_func(self):
    return lambda *args, **kwargs: None


class IonCyclotronSourceBaseTest(test_lib.SourceTestCase):
  """Tests for model-agnostic IonCyclotronSource infrastructure."""

  # TODO(b/507871842): Refactor SourceTestCase to avoid pytype disabling.
  # pytype: disable=signature-mismatch
  def setUp(self):
    super().setUp(
        source_name=icrh_base.IonCyclotronSource.SOURCE_NAME,
        source_config_class=_DummyConfig,
    )
  # pytype: enable=signature-mismatch

  def test_source_name(self):
    self.assertEqual(icrh_base.IonCyclotronSource.SOURCE_NAME, 'icrh')

  def test_affected_core_profiles(self):
    expected = (
        source_lib.AffectedCoreProfile.TEMP_ION,
        source_lib.AffectedCoreProfile.TEMP_EL,
        source_lib.AffectedCoreProfile.FAST_IONS,
    )
    self.assertEqual(
        icrh_base.IonCyclotronSource.AFFECTED_CORE_PROFILES, expected
    )

  def test_default_model_function_name(self):
    self.assertEqual(icrh_base.DEFAULT_MODEL_FUNCTION_NAME, 'toric_nn')

  def test_build_fast_ions_all_zeros(self):
    """build_fast_ions with no input returns zeros for all species."""
    geo = circular_geometry.CircularConfig().build_geometry()
    fast_ions = icrh_base.build_fast_ions(source_name='icrh', geo=geo)
    self.assertLen(fast_ions, len(fast_ion_lib.FAST_ION_SPECIES))
    for fi, species in zip(fast_ions, fast_ion_lib.FAST_ION_SPECIES):
      self.assertEqual(fi.species, species)
      self.assertEqual(fi.source, 'icrh')
      np.testing.assert_allclose(fi.n.value, 0.0, atol=1e-15)
      np.testing.assert_allclose(fi.T.value, 0.0, atol=1e-15)

  def test_zero_fast_ions(self):
    """IonCyclotronSource.zero_fast_ions returns all-zero fast ions."""
    geo = circular_geometry.CircularConfig().build_geometry()
    fast_ions = icrh_base.IonCyclotronSource.zero_fast_ions(geo)
    self.assertLen(fast_ions, len(fast_ion_lib.FAST_ION_SPECIES))
    for fi in fast_ions:
      np.testing.assert_allclose(fi.n.value, 0.0, atol=1e-15)
      np.testing.assert_allclose(fi.T.value, 0.0, atol=1e-15)

  def test_build_source_returns_correct_type(self):
    """IonCyclotronSourceConfig.build_source returns IonCyclotronSource."""
    config = _DummyConfig()
    source = config.build_source()
    self.assertIsInstance(source, icrh_base.IonCyclotronSource)

  def test_config_default_values(self):
    """Verify default config values are correct."""
    config = _DummyConfig()
    np.testing.assert_allclose(config.P_total.get_value(0.0), 10e6)
    np.testing.assert_allclose(
        config.absorption_fraction.get_value(0.0), 1.0
    )


if __name__ == '__main__':
  absltest.main()
