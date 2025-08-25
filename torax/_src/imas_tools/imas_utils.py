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

"""Useful functions for handling of IMAS IDSs."""
import imas
from imas import ids_toplevel

def load_imas_data(
    uri: str,
    ids_name: str,
) -> ids_toplevel.IDSToplevel:
  """Loads a full IDS of type ids_name for a given full uri or path_name."""
  with imas.DBEntry(uri=uri, mode="r") as db:
    ids = db.get(ids_name=ids_name)
  return ids
