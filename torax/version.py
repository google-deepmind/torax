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
"""Torax version information."""

from typing import Final

# b/404741308: This is currently a closest tag that could be made more
# fine-grained in the future, eg. based on an actual commit.
TORAX_VERSION: Final[str] = "0.3.2"


def _version_as_tuple(version_str: str) -> tuple[int, int, int]:
  return tuple(int(i) for i in version_str.split(".") if i.isdigit())


TORAX_VERSION_INFO: Final[tuple[int, int, int]] = _version_as_tuple(
    TORAX_VERSION
)
