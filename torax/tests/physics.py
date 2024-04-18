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

"""Unit tests for torax.physics."""

from typing import Callable
from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
import numpy as np
from torax import config_slice
from torax import constants
from torax import core_profile_setters
from torax import geometry
from torax import physics
from torax.sources import source_models as source_models_lib
from torax.tests.test_lib import torax_refs


class PhysicsTest(torax_refs.ReferenceValueTest):
  """Unit tests for the `torax.physics` module."""

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_calc_q_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_q_from_psi` function to a reference implementation."""
    references = references_getter()

    config = references.config
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    geo = references.geo

    # Dummy value for jtot for unit testing purposes.
    jtot = jnp.ones(config.numerics.nr)

    q_face_jax, q_cell_jax = physics.calc_q_from_jtot_psi(
        geo,
        jtot,
        references.psi,
        dynamic_config_slice.numerics.q_correction_factor,
    )

    # Make ground truth
    def calc_q_from_psi(config, geo):
      """Reference implementation from PINT."""
      consts = constants.CONSTANTS
      iota = np.zeros(config.numerics.nr + 1)  # on face grid
      q = np.zeros(config.numerics.nr + 1)  # on face grid
      # We use the reference value of psi here because the original code
      # for calculating psi depends on FiPy, and we don't want to install that
      iota[1:] = np.abs(
          references.psi_face_grad[1:]
          / geo.rmax
          / (2 * np.pi * geo.B0 * geo.r_face[1:])
      )
      q[1:] = 1 / iota[1:]
      # Change from PINT: we don't read jtot from `geo`
      q[0] = (
          2 * geo.B0 / (consts.mu0 * jtot[0] * geo.Rmaj)
      )  # use on-axis definition of q (Wesson 2004, Eq 3.48)
      q *= config.numerics.q_correction_factor

      def face_to_cell(config, face):
        cell = np.zeros(config.numerics.nr)
        cell[:] = 0.5 * (face[1:] + face[:-1])
        return cell

      q_cell = face_to_cell(config, q)
      return q, q_cell

    q_face_np, q_cell_np = calc_q_from_psi(config, geo)

    np.testing.assert_allclose(q_face_jax, q_face_np)
    np.testing.assert_allclose(q_cell_jax, q_cell_np)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_update_psi_from_j(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `update_psi_from_j` function to a reference implementation."""
    references = references_getter()

    config = references.config
    dynamic_config_slice = config_slice.build_dynamic_config_slice(config)
    geo = references.geo

    # pylint: disable=protected-access
    if isinstance(geo, geometry.CircularGeometry):
      currents = core_profile_setters._prescribe_currents_no_bootstrap(
          dynamic_config_slice,
          geo,
          source_models=source_models_lib.SourceModels(),
      )
      psi = core_profile_setters._update_psi_from_j(
          dynamic_config_slice, geo, currents
      ).value
    elif isinstance(geo, geometry.CHEASEGeometry):
      psi = geo.psi_from_chease_Ip
    else:
      raise ValueError(f'Unknown geometry type: {geo.geometry_type}')
    # pylint: enable=protected-access

    np.testing.assert_allclose(psi, references.psi.value)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_calc_jtot_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_jtot_from_psi` to a reference value."""
    references = references_getter()

    j, _ = physics.calc_jtot_from_psi(
        references.geo,
        references.psi,
    )

    np.testing.assert_allclose(j, references.jtot)

  @parameterized.parameters([
      dict(references_getter=torax_refs.circular_references),
      dict(references_getter=torax_refs.chease_references_Ip_from_chease),
      dict(references_getter=torax_refs.chease_references_Ip_from_config),
  ])
  def test_calc_s_from_psi(
      self, references_getter: Callable[[], torax_refs.References]
  ):
    """Compare `calc_s_from_psi` to a reference value."""
    references = references_getter()

    s = physics.calc_s_from_psi(
        references.geo,
        references.psi,
    )

    np.testing.assert_allclose(s, references.s)


if __name__ == '__main__':
  absltest.main()
