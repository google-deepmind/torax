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

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from torax._src.output_tools import output
from torax._src.output_tools import output_keys
from torax._src.torax_pydantic import file_restart
import xarray as xr


class StitchStateFilesTest(parameterized.TestCase):
  """Tests metadata and overlap handling when stitching restart output."""

  @staticmethod
  def _tree(
      times: list[float], values: list[int], attrs: dict[str, str]
  ) -> xr.DataTree:
    root = xr.Dataset(
        coords={
            output_keys.TIME: times,
            output_keys.RHO_CELL_NORM: (
                output_keys.RHO_CELL_NORM,
                [0.25, 0.75],
            ),
        },
        attrs=attrs,
    )
    profiles = xr.Dataset(
        {"value": ((output_keys.TIME,), values)},
        coords={output_keys.TIME: times},
    )
    return xr.DataTree(
        dataset=root,
        children={output_keys.PROFILES: xr.DataTree(dataset=profiles)},
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="normal_prior",
          prior_attrs={output_keys.CONFIG: '{"name": "A"}'},
          expected_previous_config='{"name": "A"}',
      ),
      dict(
          testcase_name="legacy_prior",
          prior_attrs={},
          expected_previous_config=None,
      ),
      dict(
          testcase_name="prior_has_older_previous_config",
          prior_attrs={
              output_keys.CONFIG: '{"name": "B"}',
              "previous_config": '{"name": "A"}',
          },
          expected_previous_config='{"name": "B"}',
      ),
      dict(
          testcase_name="legacy_prior_has_stale_previous_config",
          prior_attrs={"previous_config": '{"name": "A"}'},
          expected_previous_config=None,
      ),
  )
  def test_stitch_preserves_immediate_previous_config(
      self,
      prior_attrs: dict[str, str],
      expected_previous_config: str | None,
  ):
    prior = self._tree([0.0, 1.0, 2.0, 3.0], [10, 11, 12, 13], prior_attrs)
    current_attrs = {output_keys.CONFIG: '{"name": "C"}'}
    current = self._tree([2.0, 3.0, 4.0], [20, 21, 22], current_attrs)
    prior_before = prior.copy(deep=True)
    current_before = current.copy(deep=True)
    prior_path = os.path.join(self.create_tempdir().full_path, "prior.nc")
    prior.to_netcdf(prior_path)

    result = output.stitch_state_files(
        file_restart.FileRestartConfig(
            filename=prior_path, time=2.0, do_restart=True, stitch=True
        ),
        current,
    )

    self.assertEqual(result.attrs[output_keys.CONFIG], current_attrs["config"])
    if expected_previous_config is None:
      self.assertNotIn("previous_config", result.attrs)
    else:
      self.assertEqual(
          result.attrs["previous_config"], expected_previous_config
      )
    np.testing.assert_array_equal(
        result.children[output_keys.PROFILES].dataset["value"].values,
        [10, 11, 12, 21, 22],
    )
    np.testing.assert_array_equal(result.time.values, [0, 1, 2, 3, 4])
    xr.testing.assert_identical(
        output.load_state_file(prior_path), prior_before
    )
    xr.testing.assert_identical(current, current_before)

    result_path = os.path.join(self.create_tempdir().full_path, "result.nc")
    result.to_netcdf(result_path)
    xr.testing.assert_identical(output.load_state_file(result_path), result)

  def test_stitch_chain_keeps_only_immediate_previous_config(self):
    tempdir = self.create_tempdir().full_path
    prior_path = os.path.join(tempdir, "prior.nc")
    first_path = os.path.join(tempdir, "first.nc")
    self._tree(
        [0.0, 1.0, 2.0, 3.0],
        [10, 11, 12, 13],
        {output_keys.CONFIG: '{"name": "A"}'},
    ).to_netcdf(prior_path)

    first = output.stitch_state_files(
        file_restart.FileRestartConfig(
            filename=prior_path, time=2.0, do_restart=True, stitch=True
        ),
        self._tree(
            [2.0, 3.0, 4.0],
            [20, 21, 22],
            {output_keys.CONFIG: '{"name": "B"}'},
        ),
    )
    first.to_netcdf(first_path)
    result = output.stitch_state_files(
        file_restart.FileRestartConfig(
            filename=first_path, time=4.0, do_restart=True, stitch=True
        ),
        self._tree(
            [4.0, 5.0],
            [30, 31],
            {output_keys.CONFIG: '{"name": "C"}'},
        ),
    )

    self.assertEqual(result.attrs[output_keys.CONFIG], '{"name": "C"}')
    self.assertEqual(result.attrs["previous_config"], '{"name": "B"}')
    np.testing.assert_array_equal(
        result.children[output_keys.PROFILES].dataset["value"].values,
        [10, 11, 12, 21, 22, 31],
    )


if __name__ == "__main__":
  absltest.main()
