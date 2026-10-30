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

"""Unit tests for fast_ion.py."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from torax._src.fvm import cell_variable
from torax._src.geometry import circular_geometry
from torax._src.physics import fast_ion as fast_ion_lib


# pylint: disable=invalid-name
class FastIonTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geo = circular_geometry.CircularConfig(
        n_rho=10, a_minor=1.0
    ).build_geometry()

  @parameterized.parameters(*fast_ion_lib.FAST_ION_SPECIES)
  def test_fast_ion_valid_species(self, species):
    zeros = np.zeros(len(self.geo.rho))
    cv = cell_variable.CellVariable(
        value=zeros,
        face_centers=self.geo.rho_face_norm,
        right_face_grad_constraint=None,
        right_face_constraint=np.float64(0.0),
    )
    fi = fast_ion_lib.FastIon(
        species=species,
        source='test_source',
        n=cv,
        T=cv,
    )
    self.assertEqual(fi.species, species)

  def test_fast_ion_invalid_species_raises(self):
    zeros = np.zeros(len(self.geo.rho))
    cv = cell_variable.CellVariable(
        value=zeros,
        face_centers=self.geo.rho_face_norm,
        right_face_grad_constraint=None,
        right_face_constraint=np.float64(0.0),
    )
    with self.assertRaisesRegex(ValueError, 'not a supported species'):
      fast_ion_lib.FastIon(
          species='INVALID',
          source='test_source',
          n=cv,
          T=cv,
      )

  def test_fast_ion_from_arrays(self):
    n = jnp.ones_like(self.geo.rho_norm) * 2.0
    T = jnp.ones_like(self.geo.rho_norm) * 5.0
    fi = fast_ion_lib.FastIon.from_arrays(
        species='He3',
        source_name='icrh',
        geo=self.geo,
        n=n,
        T=T,
        n_right_face_constraint=jnp.asarray(0.5),
        T_right_face_constraint=jnp.asarray(1.5),
    )
    self.assertEqual(fi.species, 'He3')
    self.assertEqual(fi.source, 'icrh')
    np.testing.assert_allclose(fi.n.value, n)
    np.testing.assert_allclose(fi.T.value, T)
    self.assertIsNotNone(fi.n.right_face_constraint)
    self.assertIsNotNone(fi.T.right_face_constraint)
    np.testing.assert_allclose(fi.n.right_face_constraint, 0.5)
    np.testing.assert_allclose(fi.T.right_face_constraint, 1.5)

  def test_fast_ion_from_arrays_mismatched_shape_raises(self):
    face_sized = jnp.ones_like(self.geo.rho_face_norm)
    cell_sized = jnp.ones_like(self.geo.rho_norm)
    with self.assertRaisesRegex(ValueError, 'consistent with face_centers'):
      fast_ion_lib.FastIon.from_arrays(
          species='He3',
          source_name='icrh',
          geo=self.geo,
          n=face_sized,
          T=cell_sized,
      )
    with self.assertRaisesRegex(ValueError, 'consistent with face_centers'):
      fast_ion_lib.FastIon.from_arrays(
          species='He3',
          source_name='icrh',
          geo=self.geo,
          n=cell_sized,
          T=face_sized,
      )


if __name__ == '__main__':
  absltest.main()
