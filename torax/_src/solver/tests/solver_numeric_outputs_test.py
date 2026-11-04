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

"""Unit tests for SolverNumericOutputs container methods and conversions."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.geometry import circular_geometry
from torax._src.output_tools import output_grid_context
from torax._src.output_tools import output_keys
from torax._src.solver import solver_numeric_outputs


class SolverNumericOutputsTest(parameterized.TestCase):

  def test_solver_numeric_outputs_to_output_dict(self):
    geo = (
        circular_geometry.CircularConfig(n_rho=10)
        .build_runtime_params(0.0)
        .geometry
    )
    times = np.array([0.0, 1.0])
    context = output_grid_context.OutputGridContext(
        times=times,
        rho_face_norm=geo.rho_face_norm,
        rho_cell_norm=geo.rho_norm,
        rho_cell_plus_boundaries_norm=np.concatenate(
            [[0.0], geo.rho_norm, [1.0]]
        ),
    )
    outputs = solver_numeric_outputs.SolverNumericOutputs(
        outer_solver_iterations=np.array([3, 4]),
        inner_solver_iterations=np.array([10, 12]),
        solver_error_state=np.array([0, 0]),
        sawtooth_crash=np.array([False, True]),
    )
    out_dict = outputs.to_output_dict(context)

    expected_keys = {
        output_keys.OUTER_SOLVER_ITERATIONS,
        output_keys.INNER_SOLVER_ITERATIONS,
        output_keys.SOLVER_ERROR_STATE,
        output_keys.SAWTOOTH_CRASH,
    }
    self.assertEqual(out_dict.keys(), expected_keys)

    for key in expected_keys:
      dims, _, attrs = out_dict[key]
      self.assertEqual(dims, (output_keys.TIME,))
      self.assertEqual(attrs, {})

    np.testing.assert_array_equal(
        out_dict[output_keys.OUTER_SOLVER_ITERATIONS][1], np.array([3, 4])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.INNER_SOLVER_ITERATIONS][1], np.array([10, 12])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.SOLVER_ERROR_STATE][1], np.array([0, 0])
    )
    np.testing.assert_array_equal(
        out_dict[output_keys.SAWTOOTH_CRASH][1], np.array([False, True])
    )


if __name__ == '__main__':
  absltest.main()
