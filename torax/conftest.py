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

"""Pytest fixture for working around UnparsedFlagAccessError when running tests."""

import sys

import pytest
from absl import flags
# Need to import absltest to get --test_srcdir defined.
from absl.testing import absltest  # pylint: disable=unused-import


@pytest.fixture(scope='session', autouse=True)
def parse_flags():
  # Only pass the first item, because pytest flags shouldn't be parsed as absl
  # flags.
  flags.FLAGS(sys.argv[:1])
