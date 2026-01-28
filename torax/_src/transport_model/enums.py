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

"""Enums for the transport model."""

import enum


class MergeMode(enum.StrEnum):
  """Defines how a transport model's output is combined with previous models.

  Only impacts models used in the `combined` transport model.

  Attributes:
    ADD: The model's output will be added to all other transport models for the
      regions and channels it is enabled for, unless another model uses the
      `OVERWRITE` mode in which case it will be ignored in that region.
    OVERWRITE: This model's output will be used instead of any other models,
      within the domain and over the channels that it is enabled for.
  """
  ADD = 'add'
  OVERWRITE = 'overwrite'
