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

from absl.testing import absltest
from absl.testing import parameterized
from torax import interpolated_param
from torax.config import numerics
from torax.geometry import pydantic_model as geometry_pydantic_model


class NumericsTest(parameterized.TestCase):

  def test_numerics_make_provider(self):
    nums = numerics.Numerics()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = nums.make_provider(geo.torax_mesh)
    provider.build_dynamic_params(t=0.0)

  def test_interpolated_vars_are_only_constructed_once(
      self,
  ):
    """Tests that interpolated vars are only constructed once."""
    nums = numerics.Numerics()
    geo = geometry_pydantic_model.CircularConfig().build_geometry()
    provider = nums.make_provider(geo.torax_mesh)
    interpolated_params = {}
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        interpolated_params[field] = value

    # Check we don't make any additional calls to construct interpolated vars.
    provider.build_dynamic_params(t=1.0)
    for field in provider:
      value = getattr(provider, field)
      if isinstance(value, interpolated_param.InterpolatedParamBase):
        self.assertIs(value, interpolated_params[field])


if __name__ == "__main__":
  absltest.main()
