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
"""Tests for AdaptiveTGLFTransportModel."""

from concurrent import futures
import glob
import os
import shutil
import tempfile
from typing import Any

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from torax._src.data_harvesting import StagingSink
from torax._src.transport_model import adaptive_tglf_transport_model
from torax._src.transport_model import pydantic_model
from torax._src.transport_model import tglfnn_ukaea_transport_model
from torax._src.transport_model.tests import tglf_based_transport_model_test


class AdaptiveTGLFTransportModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dir = tempfile.mkdtemp()
    self.executor = futures.ThreadPoolExecutor(max_workers=2)

  def tearDown(self):
    self.executor.shutdown(wait=False)
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_adaptive_tglf_config_validation(self):
    config = pydantic_model.AdaptiveTGLFModelConfig(
        machine="multimachine",
        uncertainty_threshold=0.15,
        fallback_mode="per_face",
        smoothing_sigma=0.08,
        enable_data_harvesting=True,
        harvest_output_dir=self.test_dir,
    )
    self.assertEqual(config.model_name, "adaptive_tglf")
    self.assertEqual(config.uncertainty_threshold, 0.15)
    self.assertEqual(config.fallback_mode, "per_face")

    runtime_params = config.build_runtime_params(t=0.0)
    self.assertEqual(runtime_params.uncertainty_threshold, 0.15)
    self.assertEqual(runtime_params.fallback_mode, "per_face")

  def test_adaptive_tglf_no_fallback_when_uncertainty_low(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            "core_transport_models": {
                "bohm-gyrobohm": {"model_name": "bohm-gyrobohm"},
            },
        })
    )
    surrogate = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine="multimachine"
    )
    sink = StagingSink(output_dir=self.test_dir, run_id="no_fallback_run")

    called_solver = []

    def mock_solver(i: int, local_dict: dict[str, Any], global_dict: dict[str, Any]):
      called_solver.append(i)
      return i, 1.0, 2.0, 3.0

    model = adaptive_tglf_transport_model.AdaptiveTGLFTransportModel(
        surrogate_model=surrogate,
        executor=self.executor,
        sink=sink,
        high_fidelity_solver_fn=mock_solver,
    )

    # Set threshold very high so fallback is never triggered
    config = pydantic_model.AdaptiveTGLFModelConfig(
        machine="multimachine",
        uncertainty_threshold=1000.0,
        harvest_output_dir=self.test_dir,
    )
    transport_params = config.build_runtime_params(t=0.0)

    coeffs = model(
        transport_runtime_params=transport_params,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    self.assertEqual(coeffs.chi_face_ion.shape, geo.rho_face_norm.shape)
    # The high-fidelity mock solver must NOT have been called
    self.assertEqual(len(called_solver), 0)

  def test_adaptive_tglf_full_profile_fallback_and_harvesting(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            "core_transport_models": {
                "bohm-gyrobohm": {"model_name": "bohm-gyrobohm"},
            },
        })
    )
    surrogate = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine="multimachine"
    )
    sink = StagingSink(output_dir=self.test_dir, run_id="fallback_run")

    called_faces = []

    def mock_solver(i: int, local_dict: dict[str, Any], global_dict: dict[str, Any]):
      called_faces.append(i)
      # Return synthetic high-fidelity values: (index, pfi, efe, efi)
      return i, 0.5, 4.0, 5.0

    model = adaptive_tglf_transport_model.AdaptiveTGLFTransportModel(
        surrogate_model=surrogate,
        executor=self.executor,
        sink=sink,
        high_fidelity_solver_fn=mock_solver,
    )

    # Set threshold to 0.0 to force fallback on all faces
    config = pydantic_model.AdaptiveTGLFModelConfig(
        machine="multimachine",
        uncertainty_threshold=0.0,
        fallback_mode="full_profile",
        enable_data_harvesting=True,
        harvest_output_dir=self.test_dir,
    )
    transport_params = config.build_runtime_params(t=0.0)

    coeffs = model(
        transport_runtime_params=transport_params,
        runtime_params=runtime_params,
        geo=geo,
        core_profiles=core_profiles,
        two_point_mask=two_point_mask,
    )

    n_faces = len(geo.rho_face_norm)
    self.assertEqual(coeffs.chi_face_ion.shape, geo.rho_face_norm.shape)
    # Full profile fallback must have evaluated all faces
    self.assertEqual(len(called_faces), n_faces)

    # Flush the sink to verify harvested data
    flushed_path = sink.flush()
    self.assertIsNotNone(flushed_path)
    self.assertTrue(os.path.exists(flushed_path))

  def test_adaptive_tglf_per_face_fallback(self):
    _, (runtime_params, geo, core_profiles, _, two_point_mask) = (
        tglf_based_transport_model_test._get_config_and_model_inputs({
            "core_transport_models": {
                "bohm-gyrobohm": {"model_name": "bohm-gyrobohm"},
            },
        })
    )
    surrogate = tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel(
        machine="multimachine"
    )
    sink = StagingSink(output_dir=self.test_dir, run_id="per_face_run")

    called_faces = []

    def mock_solver(i: int, local_dict: dict[str, Any], global_dict: dict[str, Any]):
      called_faces.append(i)
      return i, 0.5, 4.0, 5.0

    model = adaptive_tglf_transport_model.AdaptiveTGLFTransportModel(
        surrogate_model=surrogate,
        executor=self.executor,
        sink=sink,
        high_fidelity_solver_fn=mock_solver,
    )

    n_faces = len(geo.rho_face_norm)
    # Mock compute_relative_uncertainty so only face 2 exceeds threshold
    synthetic_unc = np.zeros(n_faces)
    synthetic_unc[2] = 0.50

    from unittest import mock
    with mock.patch.object(
        tglfnn_ukaea_transport_model.TGLFNNukaeaTransportModel,
        "compute_relative_uncertainty",
        return_value=jnp.array(synthetic_unc),
    ):
      config = pydantic_model.AdaptiveTGLFModelConfig(
          machine="multimachine",
          uncertainty_threshold=0.20,
          fallback_mode="per_face",
          smoothing_sigma=0.05,
          enable_data_harvesting=False,
          harvest_output_dir=self.test_dir,
      )
      transport_params = config.build_runtime_params(t=0.0)

      coeffs = model(
          transport_runtime_params=transport_params,
          runtime_params=runtime_params,
          geo=geo,
          core_profiles=core_profiles,
          two_point_mask=two_point_mask,
      )

      self.assertEqual(coeffs.chi_face_ion.shape, geo.rho_face_norm.shape)
      # Only face 2 should have been evaluated!
      self.assertEqual(called_faces, [2])


if __name__ == "__main__":
  absltest.main()
